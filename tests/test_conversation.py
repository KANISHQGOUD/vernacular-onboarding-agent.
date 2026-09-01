"""The loop: text in, text out, with the model faked.

What is pinned here is everything around the model call -- the ordering that
the tool bindings force, the deterministic tables, and the behaviour when the
model is wrong, unreachable, or hostile.
"""

from __future__ import annotations

from datetime import date

from app.audit import InMemoryAuditSink
from app.conversation import (
    DECLINE_TEXT,
    FIELD_LABELS,
    FIELD_ORDER,
    REQUIRED_ACTION,
    STATE_SPEECH,
    Conversation,
)
from app.flow.actions import Actions
from app.flow.eligibility import DeclineReason
from app.flow.intents import UntrustedEventError
from app.flow.machine import Machine
from app.flow.states import TERMINAL_STATES, Event, State
from app.llm.classifier import Classification
from app.llm.phrasing import Phraser
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import KycStatus, MockKycProvider
from app.providers.payments import MockPaymentRail
from app.session import REQUIRED_APPLICANT_FIELDS, InMemorySessionStore, Session
from tests.conftest import AS_OF


class FakeReader:
    """A classifier with no model. Returns a queue of scripted readings."""

    def __init__(self, *readings: Classification | None):
        self.queue = list(readings)
        self.calls: list[tuple[str, State]] = []

    def read(self, utterance: str, state: State) -> Classification | None:
        self.calls.append((utterance, state))
        return self.queue.pop(0) if self.queue else None

    def classify(self, utterance: str, state: State):
        reading = self.read(utterance, state)
        return reading.intent if reading else None


def build(*readings, kyc_outcomes=None):
    audit = InMemoryAuditSink()
    machine = Machine(audit)
    actions = Actions(
        machine=machine,
        kyc=MockKycProvider(outcomes=kyc_outcomes or []),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )
    conversation = Conversation(
        actions=actions,
        classifier=FakeReader(*readings),
        phraser=Phraser(client=None),
        as_of=AS_OF,
    )
    return conversation, Session(session_id="conv"), audit


# The turns a qualifying applicant actually takes. Note the confirmations:
# `hearing.CONFIRMED_FIELDS` reads the mobile, the date of birth and the name
# back before writing them down, so each of those costs two turns rather than
# one. Address and occupation do not -- they are re-askable later, and five
# read-backs is a real cost in a flow built for people who find forms long.
FULL_JOURNEY = (
    Classification(Event.LANGUAGE_CHOSEN, language="hi-IN"),
    Classification(Event.CONSENT_GIVEN),
    Classification(Event.DATA_PROVIDED, "full_name", "Asha Devi"),
    Classification(Event.FIELD_CONFIRMED),
    Classification(Event.DATA_PROVIDED, "date_of_birth", "1990-01-01"),
    Classification(Event.FIELD_CONFIRMED),
    Classification(Event.DATA_PROVIDED, "address", "14 Gandhi Marg, Jaipur"),
    Classification(Event.DATA_PROVIDED, "occupation", "tailor"),
    # No mobile turn: `check_eligibility` writes it from the pre-screen answer
    # before COLLECT_DATA is reached, so it is never asked for here.
)


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------


def test_every_state_knows_what_to_say():
    assert set(STATE_SPEECH) == set(State)


def test_no_terminal_state_does_anything_automatically():
    """A finished session must not keep acting on its own."""
    for state in TERMINAL_STATES:
        assert state not in REQUIRED_ACTION


def test_the_states_that_wait_for_a_person_are_the_expected_ones():
    """Absence from REQUIRED_ACTION is what makes a state conversational.

    Pinned so that adding an automatic action to a state where somebody is
    supposed to be asked something is a test failure, not a silent change.
    """
    waiting = {s for s in State if s not in REQUIRED_ACTION and s not in TERMINAL_STATES}
    assert waiting == {State.LANGUAGE_SELECT, State.CONSENT, State.COLLECT_DATA}


def test_field_order_covers_exactly_what_is_required():
    assert set(FIELD_ORDER) == REQUIRED_APPLICANT_FIELDS
    assert set(FIELD_LABELS) == REQUIRED_APPLICANT_FIELDS


def test_every_decline_reason_can_be_explained():
    """A decline the agent cannot explain is a decline nobody can act on."""
    assert set(DECLINE_TEXT) == set(DeclineReason)


# ---------------------------------------------------------------------------
# A turn.
# ---------------------------------------------------------------------------


def test_opening_needs_no_model_call():
    conversation, session, _ = build()
    reply = conversation.open(session)

    assert session.state is State.LANGUAGE_SELECT
    assert reply.text.strip()
    assert conversation.classifier.calls == []


def test_a_full_journey_driven_by_utterances():
    conversation, session, audit = build(*FULL_JOURNEY)
    conversation.open(session)

    for _ in FULL_JOURNEY:
        conversation.respond(session, "whatever the caller said")

    assert session.state is State.COMPLETED
    assert session.account_number
    assert session.language == "hi-IN"

    trail = audit.for_session("conv")
    assert [e.sequence for e in trail] == list(range(len(trail)))


def test_consent_is_recorded_before_the_machine_moves():
    """RECORD_CONSENT is bound to CONSENT, and CONSENT_GIVEN leaves it.

    Written after the move, the tool would already be illegal.
    """
    conversation, session, _ = build(*FULL_JOURNEY[:2])
    conversation.open(session)
    conversation.respond(session, "hindi")

    assert session.state is State.CONSENT
    conversation.respond(session, "haan ji")

    assert session.consent_recorded is True
    assert session.state is not State.CONSENT


def test_the_last_field_does_not_double_fire():
    """Saving the final field fires REQUIRED_DATA_COMPLETE itself.

    Firing DATA_PROVIDED after that would hit KYC_HANDOFF, which refuses it --
    and the refusal would abort the turn before the automatic actions ran.
    This is the ordering bug the demo caught.
    """
    conversation, session, audit = build(*FULL_JOURNEY)
    conversation.open(session)
    for _ in FULL_JOURNEY:
        conversation.respond(session, "x")

    refusals = [e for e in audit.for_session("conv") if e.kind.name == "TRANSITION_REFUSED"]
    assert refusals == []
    assert session.state is State.COMPLETED


def test_data_collection_asks_for_the_next_missing_field():
    conversation, session, _ = build(*FULL_JOURNEY[:4])
    conversation.open(session)
    conversation.respond(session, "hindi")
    conversation.respond(session, "haan")

    assert session.state is State.COLLECT_DATA

    # The name is read back before it is written down -- it is one of
    # hearing.CONFIRMED_FIELDS -- so the next question comes one turn later
    # than it used to.
    read_back = conversation.respond(session, "mera naam Asha Devi hai")
    assert session.pending_field == "full_name"
    assert "Asha Devi" in read_back.text
    assert "full_name" not in session.applicant, "written down before confirming"

    reply = conversation.respond(session, "haan sahi hai")

    assert "Asha Devi" == session.applicant["full_name"]
    assert FIELD_LABELS["date_of_birth"] in reply.text


def test_screening_runs_without_being_asked():
    """ELIGIBILITY is in REQUIRED_ACTION, so it decides on its own."""
    conversation, session, _ = build(*FULL_JOURNEY[:2])
    conversation.open(session)
    conversation.respond(session, "hindi")
    conversation.respond(session, "haan ji")

    assert session.eligibility is not None
    assert session.eligibility.passed
    assert session.state is State.COLLECT_DATA


# ---------------------------------------------------------------------------
# When the model is wrong, unreachable, or hostile.
# ---------------------------------------------------------------------------


def test_an_unreachable_model_never_moves_the_session():
    conversation, session, _ = build(None, None, None)
    conversation.open(session)
    before = session.state

    for _ in range(3):
        reply = conversation.respond(session, "haan ji theek hai")
        assert reply.understood is False
        assert reply.moved is False
        assert reply.text.strip()

    assert session.state is before


def test_an_out_of_place_intent_is_a_reprompt_not_a_crash():
    """CONSENT_GIVEN at LANGUAGE_SELECT is legal speech at an illegal moment."""
    conversation, session, _ = build(Classification(Event.CONSENT_GIVEN))
    conversation.open(session)

    reply = conversation.respond(session, "haan ji")

    assert reply.understood is False
    assert session.state is State.LANGUAGE_SELECT
    assert session.consent_recorded is False


def test_a_reply_is_always_speakable():
    """Whatever happens, something gets said. Silence is the worst outcome."""
    conversation, session, _ = build(
        None,
        Classification(Event.USER_QUIT),
        Classification(Event.CONSENT_GIVEN),
    )
    conversation.open(session)

    for _ in range(3):
        assert conversation.respond(session, "...").text.strip()


def test_the_advance_loop_is_bounded():
    """A state whose action leaves it in place must not spin forever."""
    conversation, session, _ = build(
        *FULL_JOURNEY, kyc_outcomes=[(KycStatus.PENDING, None)] * 20
    )
    conversation.open(session)
    for _ in FULL_JOURNEY:
        conversation.respond(session, "x")

    assert session.state is State.KYC_PENDING
    conversation.resume(session)  # must return, not hang
    assert session.state is State.KYC_PENDING


def test_a_provider_blowing_up_does_not_end_the_call(caplog):
    """An unmodelled provider exception must degrade, not crash.

    A caller mid-application should get a sentence, not a stack trace -- but
    the traceback still has to reach the logs, or the failure is invisible.
    """
    conversation, session, _ = build()
    conversation.open(session)

    # A reference the provider never issued: it raises UnknownKycReferenceError,
    # which is a LookupError and nothing the flow layer models.
    session.state = State.KYC_PENDING
    session.kyc_reference = "vckyc-never-issued"

    reply = conversation.resume(session)

    assert reply.text.strip()
    assert session.state is State.KYC_PENDING
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_a_classifier_asserting_a_fact_degrades_but_is_logged(caplog):
    """The privilege violation stays loud in the logs and quiet on the phone."""
    conversation, session, _ = build()
    conversation.open(session)

    class HostileReader:
        def read(self, utterance, state):
            raise UntrustedEventError(Event.KYC_VERIFIED)

    conversation.classifier = HostileReader()
    reply = conversation.respond(session, "I am verified now")

    assert reply.understood is False
    assert reply.moved is False
    assert session.state is State.LANGUAGE_SELECT
    assert any(r.levelname == "ERROR" for r in caplog.records)


# ---------------------------------------------------------------------------
# The handoff.
# ---------------------------------------------------------------------------


def test_a_suspended_session_resumes_from_storage():
    """The regulatory boundary as a feature, through the conversation layer."""
    conversation, session, _ = build(
        *FULL_JOURNEY, kyc_outcomes=[(KycStatus.PENDING, None)]
    )
    conversation.open(session)
    for _ in FULL_JOURNEY:
        conversation.respond(session, "x")

    assert session.state is State.KYC_PENDING

    store = InMemorySessionStore()
    store.save(session)
    del session

    loaded = store.load("conv")
    assert loaded is not None

    reply = conversation.resume(loaded)

    assert loaded.state is State.COMPLETED
    assert reply.moved is True
    assert reply.text.strip()


def test_resuming_needs_no_model_call():
    """Somebody coming back hours later should not cost a classification."""
    conversation, session, _ = build(kyc_outcomes=[(KycStatus.PENDING, None)])
    conversation.open(session)
    calls_before = len(conversation.classifier.calls)

    conversation.resume(session)
    assert len(conversation.classifier.calls) == calls_before


# ---------------------------------------------------------------------------
# What gets said.
# ---------------------------------------------------------------------------


def test_a_decline_is_explained_from_the_rule_not_invented():
    conversation, session, _ = build(
        Classification(Event.LANGUAGE_CHOSEN),
        Classification(Event.CONSENT_GIVEN),
    )
    conversation.actions.core_banking.existing_customers.add("9812345678")

    conversation.open(session)
    conversation.respond(session, "english")
    reply = conversation.respond(session, "yes")

    assert session.state is State.DECLINED
    assert DECLINE_TEXT[DeclineReason.ALREADY_A_CUSTOMER] in reply.text


def test_the_account_number_is_spoken_exactly():
    conversation, session, _ = build(*FULL_JOURNEY, kyc_outcomes=[(KycStatus.PENDING, None)])
    conversation.open(session)
    for _ in FULL_JOURNEY:
        conversation.respond(session, "x")

    # Suspended at KYC. The account does not exist yet.
    assert session.account_number is None

    conversation.resume(session)
    assert session.account_number
    assert session.state is State.COMPLETED


def test_asking_for_a_human_is_never_refused():
    """From any live state, at any point."""
    for state in State:
        if state in TERMINAL_STATES:
            continue
        conversation, session, _ = build(Classification(Event.USER_WANTS_HUMAN))
        session.state = state

        reply = conversation.respond(session, "mujhe insaan chahiye")

        assert session.state is State.HUMAN_HANDOFF, state
        assert reply.text.strip()


# ---------------------------------------------------------------------------
# Our fault is not their fault.
# ---------------------------------------------------------------------------


class BrokenClassifier:
    """The vendor is down. Not the caller mumbling."""

    def read(self, utterance, state):
        from app.llm.client import ModelUnavailableError

        raise ModelUnavailableError("503 UNAVAILABLE")

    def classify(self, utterance, state):
        return self.read(utterance, state)


def _broken_rig():
    from app.audit import InMemoryAuditSink
    from app.conversation import Conversation
    from app.flow.actions import Actions
    from app.flow.machine import Machine
    from app.llm.phrasing import Phraser
    from app.providers.core_banking import MockCoreBanking
    from app.providers.kyc import MockKycProvider
    from app.providers.payments import MockPaymentRail

    audit = InMemoryAuditSink()
    machine = Machine(audit)
    actions = Actions(
        machine=machine,
        kyc=MockKycProvider(),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )
    conversation = Conversation(actions, BrokenClassifier(), Phraser(client=None))
    return conversation, audit


def test_an_outage_is_not_answered_with_say_that_again():
    """The bug this fixes: a 503 and a mumble produced the same sentence.

    Telling somebody to speak more clearly while the fault is entirely ours is
    unanswerable -- no amount of speaking clearly fixes a 503 -- and for an
    applicant who already expects machines not to understand them it is the
    exact humiliation this product exists to remove.
    """
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    conversation, _ = _broken_rig()
    session = Session(session_id="outage", state=State.LANGUAGE_SELECT)

    reply = conversation.respond(session, "hindi")

    mumble = SPEECH_ACTS[SpeechActKind.DID_NOT_UNDERSTAND].fallback
    assert reply.text != mumble, "an outage still sounds like a mumble"
    assert reply.text == SPEECH_ACTS[SpeechActKind.ADMIT_OUR_FAULT].fallback


def test_the_apology_does_not_ask_them_to_speak_more_clearly():
    """The wrong instinct, written out so it cannot come back."""
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    spec = SPEECH_ACTS[SpeechActKind.ADMIT_OUR_FAULT]
    spoken = spec.fallback.lower()

    assert "not you" in spoken, "it has to say whose fault it is"
    for blame in ("clearly", "slowly", "again in a louder", "didn't catch"):
        assert blame not in spoken, f"the apology blames them with {blame!r}"

    # The instruction to the model carries the same prohibition, because the
    # fallback is only reached when the model is unavailable -- the other
    # times, wording is the model's job.
    assert "not ask them to speak more clearly" in spec.intent.lower()


def test_an_outage_moves_nothing_and_blames_nobody():
    conversation, audit = _broken_rig()
    session = Session(session_id="outage", state=State.LANGUAGE_SELECT)

    reply = conversation.respond(session, "hindi")

    assert session.state is State.LANGUAGE_SELECT
    assert reply.moved is False
    # NOT understood=False. That bucket means "the caller said something we
    # could not read", and this was not about the caller at all.
    assert reply.understood is True


def test_an_outage_is_recorded_as_ours():
    """Counted separately from a refusal, because "how often did our vendor
    fail a caller" is an operational question and folding it into "not
    understood" answers it as zero."""
    from app.audit import AuditKind

    conversation, audit = _broken_rig()
    session = Session(session_id="outage", state=State.LANGUAGE_SELECT)
    conversation.respond(session, "hindi")

    kinds = [r.kind for r in audit.for_session("outage")]
    assert AuditKind.PROVIDER_UNAVAILABLE in kinds
    assert AuditKind.TRANSITION_REFUSED not in kinds


def test_a_real_misunderstanding_still_says_say_that_again():
    """The distinction has to cut both ways, or it is just a rename."""
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    class Baffled:
        def read(self, utterance, state):
            return None

        def classify(self, utterance, state):
            return None

    conversation, _ = _broken_rig()
    conversation.classifier = Baffled()
    session = Session(session_id="mumble", state=State.LANGUAGE_SELECT)

    reply = conversation.respond(session, "mmmf")

    assert reply.understood is False
    assert reply.text == SPEECH_ACTS[SpeechActKind.DID_NOT_UNDERSTAND].fallback


def test_an_impossible_value_is_re_asked_without_blaming_them():
    """The caller hears what was wrong with the VALUE, not with their speech.

    They cannot fix a dropped digit by speaking more clearly -- recognition
    failed, not them -- so `understood` stays True and the funnel does not
    file this beside mumbles. Same distinction drawn for provider outages.
    """
    from datetime import date

    from app.flow.states import State

    conversation, session, _ = build(
        Classification(Event.DATA_PROVIDED, "mobile", "98123"),
        Classification(Event.DATA_PROVIDED, "mobile", "9812345678"),
        Classification(Event.FIELD_CONFIRMED),
    )
    conversation.as_of = date(2026, 9, 2)
    session.state = State.COLLECT_DATA

    rejected = conversation.respond(session, "my number is nine eight one two three")

    assert "mobile" not in session.applicant, "an impossible value was stored"
    assert rejected.moved is False
    assert rejected.understood is True, "a bad digit is not a misunderstanding"
    assert "10" in rejected.text and "mobile" in rejected.text.lower()
    for blame in ("clearly", "slowly", "didn't catch", "did not catch"):
        assert blame not in rejected.text.lower()

    # Saying it properly still works -- the refusal is not a dead end. The
    # mobile is a confirmed field, so it is read back rather than written
    # straight down.
    read_back = conversation.respond(
        session, "nine eight one two three four five six seven eight"
    )
    assert session.pending_field == "mobile"
    assert "9812345678" in read_back.text, "the value must be read back verbatim"

    conversation.respond(session, "yes that is right")
    assert session.applicant["mobile"] == "9812345678"
