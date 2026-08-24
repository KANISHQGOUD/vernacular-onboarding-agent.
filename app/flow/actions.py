"""The wiring: tools -> providers -> facts -> transitions.

Every method here follows the same five steps, in the same order:

    1. ask the machine whether this tool is legal in this state
    2. call the provider
    3. write what came back onto the session
    4. turn the provider's answer into a fact event
    5. fire it

Step 1 before step 2 is the important ordering. Checking the binding after the
call would mean an out-of-state tool had already moved money or opened an
account by the time we noticed.

THE INVARIANT THIS FILE EXISTS TO HOLD
--------------------------------------
`actions.py` fires FACT events only. Never a user-intent event.

That is the other half of the fence in intents.py. Put together:

    a language model may emit only USER-INTENT events   (intents.py)
    this layer may emit only FACT events                (here)
    and a fact event is minted only from a real provider response

So every fact in a session -- KYC verified, account opened, payment sent -- is
traceable to something a provider actually said. Nothing in the system can
produce one by reasoning, phrasing, or being persuaded. A test asserts this
file never fires anything from USER_INTENT_EVENTS.

That is also why `record_consent`, `set_language` and `save_applicant_field`
write to the session but do NOT move the machine. What those record is a user's
intent, and moving the machine on an intent is the classifier's job, not
the provider layer's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.flow.eligibility import EligibilityInput, Residency, evaluate
from app.audit import AuditKind
from app.flow.machine import Machine
from app.flow.states import Event, ToolName
from app.providers.core_banking import CoreBankingProvider
from app.providers.kyc import KycProvider, KycStatus
from app.providers.payments import (
    FIRST_TRANSACTION_CEILING_PAISE,
    PaymentRail,
    PaymentResult,
)
from app.session import Session


class ActionRefused(Exception):
    """The action could not be attempted. Distinct from a provider saying no.

    A refusal here means we would not even ask -- the amount was over our own
    ceiling, or a required reference was missing. A provider failure is a
    normal outcome that produces a fact event; this is a caller mistake.
    """


@dataclass
class Actions:
    """Holds the providers. One instance per process, shared across sessions."""

    machine: Machine
    kyc: KycProvider
    core_banking: CoreBankingProvider
    payments: PaymentRail

    # -- Recording tools: write, but do not move the machine -----------------

    def set_language(self, session: Session, language: str) -> Session:
        self.machine.assert_tool_allowed(session, ToolName.SET_LANGUAGE)
        session.language = language
        return session

    def record_consent(self, session: Session) -> Session:
        self.machine.assert_tool_allowed(session, ToolName.RECORD_CONSENT)
        session.consent_recorded = True
        return session

    def save_applicant_field(self, session: Session, field_name: str, value: str) -> Session:
        """Saving one field is an intent; having them all is a fact.

        So this writes without firing, and fires only when the required set is
        complete -- and that completeness is decided by comparing against
        REQUIRED_APPLICANT_FIELDS, never by asking the model whether it thinks
        it has enough.
        """
        self.machine.assert_tool_allowed(session, ToolName.SAVE_APPLICANT_FIELD)
        session.applicant[field_name] = value

        if session.required_data_complete:
            return self.machine.fire(session, Event.REQUIRED_DATA_COMPLETE)
        return session

    def record_lapse(self, session: Session) -> Session:
        """Nobody has said anything for a while.

        Note what this does NOT do: fire an event. The session does not move.
        Somebody who stops answering in COLLECT_DATA is still in COLLECT_DATA,
        and coming back needs no transition because they never left.

        That is not a shortcut. Resuming a state you moved OUT of means a way
        back IN to it, and the way back in to ACCOUNT_CREATE is the one thing
        this whole design exists to keep singular.
        """
        session.notes["lapses"] = str(session.lapses + 1)
        session.notes["lapsed_in"] = str(session.state)
        self.machine.record(
            session,
            AuditKind.SESSION_LAPSED,
            detail=f"no answer; lapse {session.lapses}",
        )
        return session

    def record_return(self, session: Session) -> Session:
        """They came back. There is nothing to undo.

        The conversation layer re-runs whatever the current state requires, so
        a returning caller is asked the question they were already on. A
        session reloaded from storage does not know it was ever away -- which
        is the same sentence `Conversation.resume` was written under, for the
        video KYC handoff, before there was anything else to resume.
        """
        self.machine.record(
            session,
            AuditKind.SESSION_RETURNED,
            detail=f"back after lapse {session.lapses}",
        )
        return session

    def escalate_to_human(self, session: Session, reason: str) -> Session:
        self.machine.assert_tool_allowed(session, ToolName.ESCALATE_TO_HUMAN)
        session.notes["escalation_reason"] = reason
        return session

    # -- Provider tools: call out, then mint a fact --------------------------

    def check_eligibility(
        self,
        session: Session,
        *,
        date_of_birth: date,
        residency: Residency,
        has_pan: bool,
        has_form_60: bool,
        mobile: str,
        mobile_linked_to_aadhaar: bool,
        as_of: date,
    ) -> Session:
        """Screening: one fact from the provider, the rest from the applicant.

        Note the division of labour. Core banking supplies whether they already
        bank here; the applicant supplies the rest; the ruleset decides. The
        rules themselves never make a network call, which is exactly what keeps
        them pure and the decision replayable.

        `as_of` is passed in rather than read from a clock, for the same reason.
        """
        self.machine.assert_tool_allowed(session, ToolName.CHECK_ELIGIBILITY)

        lookup = self.core_banking.find_customer(mobile)

        applicant = EligibilityInput(
            date_of_birth=date_of_birth,
            residency=residency,
            has_pan=has_pan,
            has_form_60=has_form_60,
            has_existing_active_account=lookup.has_active_account,
            mobile_linked_to_aadhaar=mobile_linked_to_aadhaar,
            as_of=as_of,
        )

        # Both halves stored: the answers that went in and the verdict that
        # came out. Without the input, "why was this person declined in March"
        # is unanswerable once the ruleset moves on.
        session.eligibility_input = applicant
        session.eligibility = evaluate(applicant)
        session.applicant["mobile"] = mobile

        return self.machine.fire(session, Event.ELIGIBILITY_CHECKED)

    def start_video_kyc(self, session: Session) -> Session:
        """Hand off. This is where the agent stops being in charge."""
        self.machine.assert_tool_allowed(session, ToolName.START_VIDEO_KYC)

        kyc_session = self.kyc.start_session(
            applicant_reference=session.session_id,
            language=session.language or "en-IN",
        )
        session.kyc_reference = kyc_session.reference

        # Counted at dispatch, not at failure. A session that was started and
        # never completed still consumed an attempt, and counting only failures
        # would let an applicant loop forever by abandoning each call.
        session.kyc_attempts += 1

        return self.machine.fire(session, Event.KYC_DISPATCHED)

    def check_kyc_status(self, session: Session) -> Session:
        """Poll. The only question we are allowed to ask about identity.

        Still PENDING is a real answer, not a failure -- the applicant may not
        have joined yet, or may be mid-call. It moves nothing, and the session
        stays suspended, possibly for hours.
        """
        self.machine.assert_tool_allowed(session, ToolName.CHECK_KYC_STATUS)

        if not session.kyc_reference:
            raise ActionRefused("no KYC reference on this session; nothing to poll")

        result = self.kyc.get_status(session.kyc_reference)

        if result.status is KycStatus.PENDING:
            return session

        if result.status is KycStatus.VERIFIED:
            return self.machine.fire(session, Event.KYC_VERIFIED)

        # FAILED and EXPIRED are the same event; the map's guard decides whether
        # that means another attempt or a human.
        session.notes["last_kyc_failure"] = str(result.failure_reason)
        return self.machine.fire(session, Event.KYC_FAILED)

    def open_account(self, session: Session) -> Session:
        self.machine.assert_tool_allowed(session, ToolName.OPEN_ACCOUNT)

        if not session.kyc_reference:
            # Unreachable through the map -- ACCOUNT_CREATE is only enterable
            # via KYC_VERIFIED. Kept because core banking requires the
            # reference anyway, and failing here gives a clearer error than a
            # rejection from the far side.
            raise ActionRefused("cannot open an account without a KYC reference")

        result = self.core_banking.open_account(
            full_name=session.applicant.get("full_name", ""),
            mobile=session.applicant.get("mobile", ""),
            kyc_reference=session.kyc_reference,
        )

        if not result.opened:
            session.notes["account_failure"] = str(result.failure_reason)
            return self.machine.fire(session, Event.ACCOUNT_CREATION_FAILED)

        session.account_number = result.account_number
        return self.machine.fire(session, Event.ACCOUNT_OPENED)

    def initiate_deposit(self, session: Session, amount_paise: int) -> Session:
        """First money in. Amount is paise, an integer, always."""
        self.machine.assert_tool_allowed(session, ToolName.INITIATE_DEPOSIT)

        if not session.account_number:
            raise ActionRefused("no account to deposit into")

        result = self.payments.initiate_deposit(session.account_number, amount_paise)

        if not result.succeeded:
            # No event: a failed deposit leaves the session in FUNDING so the
            # applicant can simply try again. The map has no "deposit failed"
            # edge because failing to add money is not a change of state.
            session.notes["deposit_failure"] = str(result.failure_reason)
            return session

        session.deposit_reference = result.reference
        return self.machine.fire(session, Event.DEPOSIT_CONFIRMED)

    def make_first_payment(self, session: Session, to_vpa: str, amount_paise: int) -> Session:
        """The success metric. Everything before this was setup.

        The ceiling checked here is OUR policy, not the rail's. The rail
        enforces UPI's per-transaction limit; this enforces that a person's
        very first payment, guided by a voice they have known for four minutes,
        should be small.
        """
        self.machine.assert_tool_allowed(session, ToolName.MAKE_FIRST_PAYMENT)

        if not session.account_number:
            raise ActionRefused("no account to pay from")

        if amount_paise > FIRST_TRANSACTION_CEILING_PAISE:
            raise ActionRefused(
                f"first transaction is capped at {FIRST_TRANSACTION_CEILING_PAISE} paise"
            )

        result: PaymentResult = self.payments.send_payment(
            from_account=session.account_number, to_vpa=to_vpa, amount_paise=amount_paise
        )

        if not result.succeeded:
            session.notes["payment_failure"] = str(result.failure_reason)
            return session

        session.first_payment_reference = result.reference
        return self.machine.fire(session, Event.PAYMENT_CONFIRMED)
