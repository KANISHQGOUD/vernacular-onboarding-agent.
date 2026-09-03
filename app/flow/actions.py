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

import logging
from dataclasses import dataclass, replace
from datetime import date

from app.flow.eligibility import EligibilityInput, Residency, evaluate
from app.flow.hearing import needs_confirming
from app.flow.fields import (
    MAX_FIELD_ATTEMPTS,
    FieldProblem,
    as_date,
    is_year_only,
    normalised,
    validate,
)
from app.audit import AuditKind
from app.flow.machine import Machine
from app.flow.states import Event, State, ToolName
from app.providers.core_banking import CoreBankingProvider
from app.providers.kyc import KycProvider, KycStatus
from app.providers.payments import (
    FIRST_TRANSACTION_CEILING_PAISE,
    format_rupees,
    PaymentRail,
    PaymentResult,
)
from app.session import Session

log = logging.getLogger(__name__)


class FieldRejected(Exception):
    """What we heard cannot be what they said.

    Not a caller mistake and not a provider failure -- a recognition failure,
    which is a third thing. The caller did nothing wrong and should not be
    told they did; `attempts` lets the conversation escalate its wording, and
    `problem` is phrased to be repeatable to them.
    """

    def __init__(self, field: str, problem: FieldProblem, attempts: int) -> None:
        self.field = field
        self.problem = problem
        self.attempts = attempts
        super().__init__(f"{field}: {problem}")


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

    def save_applicant_field(
        self,
        session: Session,
        field_name: str,
        value: str,
        *,
        as_of: date,
    ) -> Session:
        """Saving one field is an intent; having them all is a fact.

        So this writes without firing, and fires only when the required set is
        complete -- and that completeness is decided by comparing against
        REQUIRED_APPLICANT_FIELDS, never by asking the model whether it thinks
        it has enough.

        VALIDATED FIRST, BUT NOT FOREVER
        --------------------------------
        A value that cannot be what they said is refused and re-asked, because
        the provider reports no confidence and a rule that says "impossible"
        is better evidence than a number that says "unsure".

        After MAX_FIELD_ATTEMPTS it is stored anyway, flagged. Every rule in
        fields.py is a guess about what a real answer looks like, and each is
        wrong about somebody -- an unusual name, a number that does not fit
        the Indian mobile shape. A flow whose entire purpose is not turning
        people away must not invent a new way to turn them away, so the last
        word belongs to the applicant and the flag belongs to whoever reviews
        it.
        """
        self.machine.assert_tool_allowed(session, ToolName.SAVE_APPLICANT_FIELD)

        # Recorded before validating, and kept even when the value is
        # refused: "what did the machine hear on the attempt that failed" is
        # exactly the question worth being able to answer later.
        session.heard[field_name] = value

        problem = validate(field_name, value, as_of=as_of)
        if problem is not None:
            attempts = session.field_attempts(field_name) + 1
            session.notes[f"attempts:{field_name}"] = str(attempts)

            if attempts < MAX_FIELD_ATTEMPTS:
                raise FieldRejected(field_name, problem, attempts)

            # Out of attempts. Their answer wins, and the trail says why.
            log.warning(
                "storing %s unvalidated after %s attempts: %s",
                field_name,
                attempts,
                problem,
            )
            session.notes[f"unvalidated:{field_name}"] = str(problem)
            self.machine.record(
                session,
                AuditKind.FIELD_UNVALIDATED,
                detail=f"{field_name} stored after {attempts} attempts: {problem}",
            )

        # A bare year became 1 January. Recorded, because "was this day real
        # or assumed" is a question a reviewer will eventually ask, and the
        # stored date cannot answer it on its own.
        if field_name == "date_of_birth" and is_year_only(value):
            session.notes["inferred:date_of_birth"] = "year only; day assumed 1 January"

        clean = normalised(field_name, value)

        # High-stakes fields are read back before they are written down. See
        # hearing.CONFIRMED_FIELDS: the provider reports no confidence, so the
        # only party who actually knows whether we heard it right is asked.
        if needs_confirming(field_name):
            session.pending_field = field_name
            session.pending_value = clean
            return session

        return self._commit_field(session, field_name, clean, as_of=as_of)

    def _commit_field(
        self, session: Session, field_name: str, value: str, *, as_of: date
    ) -> Session:
        """Write it down, re-screen if it changes the verdict, then finish.

        Shared by the confirmed and unconfirmed paths, because "was that the
        final field" must be answered the same way regardless of how the value
        arrived -- and a session that hit the budget on its last field would
        otherwise sit in COLLECT_DATA forever.
        """
        session.applicant[field_name] = value

        if field_name == "date_of_birth":
            session = self._rescreen_eligibility(session, value, as_of=as_of)
            # Declined. Nothing further is collected, and firing completion
            # from a terminal state would be illegal anyway.
            if session.state is not State.COLLECT_DATA:
                return session

        if session.required_data_complete:
            return self.machine.fire(session, Event.REQUIRED_DATA_COMPLETE)
        return session

    def _rescreen_eligibility(
        self, session: Session, value: str, *, as_of: date
    ) -> Session:
        """Ask the ruleset again, using the date they actually gave us.

        WHY THIS HAS TO EXIST
        ---------------------
        ELIGIBILITY runs before COLLECT_DATA, on pre-screen answers. That
        ordering is right -- there is no point collecting five fields from
        somebody the rules will decline -- but it meant the age rule was
        evaluated against an answer given before the real date of birth was
        ever spoken, and `Conversation._screen` supplies a placeholder. The
        engine has always been able to decline a minor; it was simply never
        shown one.

        So the pre-screen stays a screen, and this is the check. The field
        rules in fields.py are about RECOGNITION -- could we have heard that
        -- and they rightly give up after three attempts and let the applicant
        win. Eligibility is about LAW, and does not give up. Keeping those
        separate is what lets the escape hatch stay generous without it
        becoming a way through the age gate.

        Silent when there was no pre-screen to revise: a session that reached
        COLLECT_DATA without one has no other answers to combine this with,
        and inventing them here would be deciding on data nobody supplied.
        """
        prior = session.eligibility_input
        born = as_date(value)
        if prior is None or born is None:
            return session

        applicant = replace(prior, date_of_birth=born, as_of=as_of)
        session.eligibility_input = applicant
        session.eligibility = evaluate(applicant)
        return self.machine.fire(session, Event.ELIGIBILITY_CHECKED)

    def confirm_pending_field(self, session: Session, *, as_of: date) -> Session:
        """They said yes. Write it down.

        Does NOT fire FIELD_CONFIRMED, and that is the rule this file lives
        by: what a caller meant is an intent, and moving the machine on an
        intent belongs to the layer that read it, not to the provider layer.
        `record_consent` and `save_applicant_field` are the same shape, and a
        test parses this module to keep it that way.
        """
        field_name = session.pending_field
        value = session.pending_value
        session.pending_field = None
        session.pending_value = None

        if field_name is None or value is None:
            return session
        return self._commit_field(session, field_name, value, as_of=as_of)

    def reject_pending_field(self, session: Session) -> Session:
        """They said no. Forget it and count the miss.

        Fires nothing, for the reason above. The value is discarded rather
        than kept as a suggestion: a read-back they have already denied is not
        a starting point for a guess -- it is the wrong answer, and offering
        it back would be arguing with them.
        """
        field_name = session.pending_field
        session.pending_field = None
        session.pending_value = None

        if field_name is not None:
            session.notes[f"corrections:{field_name}"] = str(
                session.corrections(field_name) + 1
            )
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

    def record_interruption(self, session: Session) -> Session:
        """They talked over the agent, somewhere it matters.

        Not a refusal and not a transition -- barge-in is allowed everywhere,
        and the session does not move. It goes in the trail because at the
        consent gate "did this person actually receive the notice" is a
        question somebody may have to answer, and four seconds into a
        twenty-second explanation is a weak yes.
        """
        session.notes["interrupted_in"] = str(session.state)
        self.machine.record(
            session,
            AuditKind.CALLER_INTERRUPTED,
            detail=f"caller spoke over the agent in {session.state}",
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
        # Normalised, because this is a second door into `applicant` that
        # does not go through `save_applicant_field` -- so it gets neither
        # validation nor a read-back. Defensible for a pre-screen answer,
        # which was given before the conversation reached COLLECT_DATA and is
        # therefore never asked for again; worth knowing about all the same.
        session.applicant["mobile"] = normalised("mobile", mobile)

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
        session.deposit_paise = amount_paise
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
            # Rupees, not paise. This string is repeated to the caller by
            # `_amount_refused_reply`, and paise is a unit for the ledger --
            # telling somebody their payment is "capped at 50000 paise" is
            # the machine's internals leaking into a conversation.
            raise ActionRefused(
                "a first payment is capped at "
                f"{format_rupees(FIRST_TRANSACTION_CEILING_PAISE)}"
            )

        result: PaymentResult = self.payments.send_payment(
            from_account=session.account_number, to_vpa=to_vpa, amount_paise=amount_paise
        )

        if not result.succeeded:
            session.notes["payment_failure"] = str(result.failure_reason)
            return session

        session.first_payment_reference = result.reference
        session.first_payment_paise = amount_paise
        return self.machine.fire(session, Event.PAYMENT_CONFIRMED)
