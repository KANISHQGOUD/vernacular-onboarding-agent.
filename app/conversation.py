"""Text in, text out. The loop that finally joins the halves.

One turn, in order:

    1. read the utterance            -> an intent, and maybe a value
    2. apply what the intent records -> language, consent, a field
    3. fire the intent               -> the machine moves, or refuses
    4. run whatever the new state requires, deterministically
    5. say what the new state says

Steps 4 and 5 are the ones worth staring at, because they are where a normal
agent would hand the model the wheel.

WHO DECIDES WHAT HAPPENS: NOT THE MODEL
---------------------------------------
`REQUIRED_ACTION` maps a state to the thing that must happen there -- poll KYC
in KYC_PENDING, open the account in ACCOUNT_CREATE. It is a table, like the
transitions and the tool bindings, for the same reason: you can read it, diff
it, and enumerate what it does not contain.

The model is never asked "what should we do next?". It is asked "what did this
person mean?", and the answer to that question is the only thing it supplies.

WHO DECIDES WHAT IS SAID: ALSO NOT THE MODEL
--------------------------------------------
`STATE_SPEECH` maps a state to the speech act belonging to it. The model
receives that act and chooses the wording, in the caller's language. It cannot
raise a subject the machine did not pick, because a SpeechAct is the only
thing the phrasing layer accepts.

So a turn where the model is entirely wrong produces, at worst, a
misunderstanding -- never an unrequested action and never an unrequested topic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping

from app.flow.actions import ActionRefused, Actions
from app.flow.eligibility import DeclineReason, Residency
from app.flow.intents import UntrustedEventError
from app.flow.machine import FlowError
from app.flow.states import Event, State
from app.languages import DEFAULT_LANGUAGE, normalise
from app.llm.classifier import Classification, GeminiIntentClassifier
from app.llm.phrasing import Phraser
from app.llm.speech_acts import SpeechAct, SpeechActKind
from app.providers.payments import format_rupees
from app.session import Session

log = logging.getLogger(__name__)

# What the agent asks for, in the order it asks. A list rather than the
# frozenset so the questions arrive in a sensible order instead of whatever
# order the set happens to iterate in.
FIELD_ORDER: tuple[str, ...] = (
    "full_name",
    "date_of_birth",
    "address",
    "occupation",
    "mobile",
)

FIELD_LABELS: Mapping[str, str] = {
    "full_name": "full name",
    "date_of_birth": "date of birth",
    "address": "address",
    "occupation": "occupation",
    "mobile": "mobile number",
}

# What a decline is explained as. Derived from the rule id, never composed by
# the model -- an invented decline reason is an invented bank policy.
DECLINE_TEXT: Mapping[DeclineReason, str] = {
    DeclineReason.MINOR_NEEDS_GUARDIAN: (
        "This account is for people aged eighteen and over. A guardian can "
        "open one with you."
    ),
    DeclineReason.NRI_DIFFERENT_PRODUCT: (
        "This account is for residents of India. There is a different account "
        "for people living abroad."
    ),
    DeclineReason.TAX_ID_MISSING: (
        "A PAN card, or Form 60 instead of one, is needed to open this account."
    ),
    DeclineReason.ALREADY_A_CUSTOMER: (
        "There is already an account with this mobile number. Branch staff can "
        "help you with it."
    ),
    DeclineReason.MOBILE_NOT_LINKED: (
        "The mobile number needs to be linked to your Aadhaar so we can send "
        "you a one-time code."
    ),
}

# Amounts the demo uses. In a real deployment these come from the caller.
DEMO_DEPOSIT_PAISE = 50_000
DEMO_PAYMENT_PAISE = 12_550
DEMO_PAYEE = "kirana@upi"


@dataclass(frozen=True, slots=True)
class Reply:
    """One turn's outcome. `text` is always speakable."""

    text: str
    state: State
    moved: bool
    understood: bool

    def __str__(self) -> str:
        return self.text


@dataclass
class Conversation:
    """Drives one session. Holds no per-turn state of its own."""

    actions: Actions
    classifier: GeminiIntentClassifier
    phraser: Phraser
    as_of: date = field(default_factory=date.today)

    # -- The turn -----------------------------------------------------------

    def open(self, session: Session) -> Reply:
        """Start the conversation. No utterance yet, so no model call."""
        self._advance(session)
        return self._reply(session, moved=True, understood=True)

    def resume(self, session: Session) -> Reply:
        """Pick a suspended session back up. No utterance, no model call.

        This is the regulatory handoff's other half. The applicant left to do
        video KYC and has come back -- possibly hours later, possibly on a
        different device -- and something has to re-poll the provider and carry
        on from wherever the session actually is.

        Deliberately identical to `open`: resuming is not a special mode, it is
        just running the state's required action again. A session reloaded from
        storage does not know it was ever away.
        """
        before = session.state
        self._advance(session)
        return self._reply(session, moved=session.state is not before, understood=True)

    def respond(self, session: Session, utterance: str) -> Reply:
        before = session.state

        try:
            reading = self.classifier.read(utterance, session.state)
        except UntrustedEventError:
            # The classifier asserted a fact -- KYC passed, money moved. That
            # is a privilege violation, and it is logged at ERROR so monitoring
            # sees it. But it must not end the call: the person on the phone
            # did nothing wrong, and a stack trace is not an answer.
            log.error(
                "classifier asserted a fact in %s -- refusing the turn",
                before,
                exc_info=True,
            )
            return self._reply(session, moved=False, understood=False)

        if reading is None:
            # Not understood, rate-limited, or the model was unreachable. All
            # three are the same to the caller, and none of them move anything.
            return self._reply(session, moved=False, understood=False)

        self._record_before_moving(session, reading)

        try:
            self.actions.machine.fire(session, reading.intent)
        except FlowError:
            # A legal intent that this state cannot act on. Routine: the
            # caller said something reasonable at an unreasonable moment.
            log.info("refused %s in %s", reading.intent, before)
            return self._reply(session, moved=False, understood=False)

        self._record_after_moving(session, reading)
        self._advance(session)
        return self._reply(session, moved=session.state is not before, understood=True)

    # -- Step 2: what the intent records ------------------------------------
    #
    # Split around the transition, because the tool bindings demand it and the
    # split is not arbitrary:
    #
    #   RECORD_CONSENT is bound to CONSENT, and CONSENT_GIVEN leaves CONSENT.
    #   So consent must be written before the machine moves, or the tool is
    #   no longer legal by the time we try.
    #
    #   SAVE_APPLICANT_FIELD is bound to COLLECT_DATA, and DATA_PROVIDED is a
    #   self-loop that stays there. Saving must happen after the move, because
    #   saving the LAST field fires REQUIRED_DATA_COMPLETE itself -- and doing
    #   that first would leave the machine in KYC_HANDOFF while we still had a
    #   DATA_PROVIDED to fire, which it would rightly refuse.

    def _record_before_moving(self, session: Session, reading: Classification) -> None:
        if reading.language:
            session.language = normalise(reading.language)

        if reading.intent is Event.CONSENT_GIVEN and session.state is State.CONSENT:
            self.actions.record_consent(session)

    def _record_after_moving(self, session: Session, reading: Classification) -> None:
        if reading.carries_data and session.state is State.COLLECT_DATA:
            try:
                self.actions.save_applicant_field(session, reading.field, reading.value)
            except FlowError:
                log.info("could not save %s in %s", reading.field, session.state)

    # -- Step 4: what the state requires, deterministically ------------------

    def _advance(self, session: Session, limit: int = 6) -> None:
        """Run the automatic actions until the machine needs a person again.

        Bounded rather than `while True`. A state whose required action leaves
        the machine where it started would otherwise spin forever, and a loop
        that cannot terminate is a worse failure than one that gives up.
        """
        for _ in range(limit):
            action = REQUIRED_ACTION.get(session.state)
            if action is None:
                return

            before = session.state
            try:
                action(self, session)
            except (ActionRefused, FlowError) as exc:
                # Expected: the action could not run here. Stop and let the
                # state speak for itself.
                log.warning("required action for %s failed: %s", before, exc)
                return
            except Exception:
                # Unexpected: a provider raised something we do not model --
                # an unknown reference, a client bug, a library blowing up.
                # Caught deliberately broadly, because the alternative is a
                # caller mid-application receiving a stack trace instead of a
                # sentence. Logged at ERROR with a traceback so nothing hides.
                log.error("required action for %s raised", before, exc_info=True)
                return

            if session.state is before:
                return

    # -- Step 5: what the state says ----------------------------------------

    def _reply(self, session: Session, *, moved: bool, understood: bool) -> Reply:
        act = (
            self._speech_for(session)
            if understood
            else SpeechAct(
                kind=SpeechActKind.DID_NOT_UNDERSTAND,
                language=session.language or DEFAULT_LANGUAGE,
            )
        )
        return Reply(
            text=self.phraser.say(act),
            state=session.state,
            moved=moved,
            understood=understood,
        )

    def _speech_for(self, session: Session) -> SpeechAct:
        kind = STATE_SPEECH.get(session.state, SpeechActKind.DID_NOT_UNDERSTAND)
        language = session.language or DEFAULT_LANGUAGE
        return SpeechAct(kind=kind, language=language, data=self._data_for(session, kind))

    @staticmethod
    def _data_for(session: Session, kind: SpeechActKind) -> dict[str, str]:
        if kind is SpeechActKind.ASK_FIELD:
            missing = next(
                (f for f in FIELD_ORDER if f in session.missing_fields),
                None,
            )
            return {"field_label": FIELD_LABELS.get(missing, "details")} if missing else {}

        if kind is SpeechActKind.ANNOUNCE_DECLINE:
            reasons = session.eligibility.decline_reasons if session.eligibility else ()
            text = " ".join(DECLINE_TEXT[r] for r in reasons if r in DECLINE_TEXT)
            return {"reason_text": text or "The details given do not meet the requirements."}

        if kind is SpeechActKind.ANNOUNCE_ACCOUNT_OPENED and session.account_number:
            return {"account_number": session.account_number}

        if kind is SpeechActKind.CONFIRM_COMPLETED:
            return {"amount_text": format_rupees(DEMO_PAYMENT_PAISE)}

        return {}

    # -- The automatic actions themselves ------------------------------------

    def _start(self, session: Session) -> None:
        self.actions.machine.fire(session, Event.SESSION_STARTED)

    def _screen(self, session: Session) -> None:
        """Run eligibility from the pre-screening answers.

        The demo supplies a qualifying applicant. In a deployment these come
        from a short spoken pre-screen, which is why `check_eligibility` takes
        them as arguments rather than reading them off the session: they are
        answers to questions, not yet an application record.
        """
        self.actions.check_eligibility(
            session,
            date_of_birth=date(1990, 1, 1),
            residency=Residency.RESIDENT,
            has_pan=True,
            has_form_60=False,
            mobile=session.applicant.get("mobile", "9812345678"),
            mobile_linked_to_aadhaar=True,
            as_of=self.as_of,
        )

    def _dispatch_kyc(self, session: Session) -> None:
        self.actions.start_video_kyc(session)

    def _poll_kyc(self, session: Session) -> None:
        self.actions.check_kyc_status(session)

    def _open_account(self, session: Session) -> None:
        self.actions.open_account(session)

    def _deposit(self, session: Session) -> None:
        self.actions.initiate_deposit(session, DEMO_DEPOSIT_PAISE)

    def _pay(self, session: Session) -> None:
        self.actions.make_first_payment(session, DEMO_PAYEE, DEMO_PAYMENT_PAISE)


# ---------------------------------------------------------------------------
# THE TWO TABLES
# ---------------------------------------------------------------------------
# Data, for the same reasons the transition table is data: readable, diffable,
# and enumerable by what they leave out. A state absent from REQUIRED_ACTION
# does nothing on its own -- it waits for a person.
# ---------------------------------------------------------------------------

REQUIRED_ACTION: Mapping[State, Callable[[Conversation, Session], None]] = {
    State.START: Conversation._start,
    State.ELIGIBILITY: Conversation._screen,
    State.KYC_HANDOFF: Conversation._dispatch_kyc,
    State.KYC_PENDING: Conversation._poll_kyc,
    State.ACCOUNT_CREATE: Conversation._open_account,
    State.FUNDING: Conversation._deposit,
    State.FIRST_TRANSACTION: Conversation._pay,
}

STATE_SPEECH: Mapping[State, SpeechActKind] = {
    State.START: SpeechActKind.ASK_LANGUAGE,
    State.LANGUAGE_SELECT: SpeechActKind.ASK_LANGUAGE,
    State.CONSENT: SpeechActKind.EXPLAIN_CONSENT,
    State.ELIGIBILITY: SpeechActKind.EXPLAIN_CONSENT,
    State.COLLECT_DATA: SpeechActKind.ASK_FIELD,
    State.KYC_HANDOFF: SpeechActKind.EXPLAIN_KYC_HANDOFF,
    State.KYC_PENDING: SpeechActKind.KYC_STILL_WAITING,
    State.ACCOUNT_CREATE: SpeechActKind.EXPLAIN_KYC_HANDOFF,
    State.FUNDING: SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
    State.FIRST_TRANSACTION: SpeechActKind.GUIDE_FIRST_PAYMENT,
    State.COMPLETED: SpeechActKind.CONFIRM_COMPLETED,
    State.DECLINED: SpeechActKind.ANNOUNCE_DECLINE,
    State.CONSENT_REFUSED: SpeechActKind.ACKNOWLEDGE_NO_CONSENT,
    State.ABANDONED: SpeechActKind.HANDING_TO_HUMAN,
    State.HUMAN_HANDOFF: SpeechActKind.HANDING_TO_HUMAN,
}
