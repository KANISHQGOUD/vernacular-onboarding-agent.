"""The wiring layer: tools -> providers -> facts -> transitions.

The headline test here is `test_actions_never_fires_a_user_intent_event`. It
reads the source of actions.py and asserts that every event it fires is a fact.
Together with the classifier tests in test_intents.py, that closes the loop:

    a model may emit only intents          (test_intents.py)
    this layer may emit only facts         (here)

so no fact in any session can originate anywhere but a provider response.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date

import pytest

from app.audit import AuditKind, InMemoryAuditSink
from app.flow import actions as actions_module
from app.flow.actions import ActionRefused, Actions
from app.flow.eligibility import DeclineReason, Residency
from app.flow.intents import FACT_EVENTS, USER_INTENT_EVENTS
from app.flow.machine import Machine, ToolNotBoundError
from app.flow.states import MAX_KYC_ATTEMPTS, Event, State
from app.providers.core_banking import AccountOpeningFailure, MockCoreBanking
from app.providers.kyc import KycFailureReason, KycStatus, MockKycProvider
from app.providers.payments import (
    FIRST_TRANSACTION_CEILING_PAISE,
    MockPaymentRail,
    PaymentFailure,
)
from app.session import Session
from tests.conftest import AS_OF

GOOD_APPLICANT = dict(
    date_of_birth=date(1990, 1, 1),
    residency=Residency.RESIDENT,
    has_pan=True,
    has_form_60=False,
    mobile="9000000000",
    mobile_linked_to_aadhaar=True,
    as_of=AS_OF,
)


@pytest.fixture
def rig():
    """A full stack: machine, audit, and all three mocked providers."""
    audit = InMemoryAuditSink()
    machine = Machine(audit)
    kyc = MockKycProvider()
    bank = MockCoreBanking()
    rail = MockPaymentRail()
    return Actions(machine=machine, kyc=kyc, core_banking=bank, payments=rail), audit


def drive_to_kyc_pending(actions: Actions, session: Session) -> Session:
    """Walk to the handoff using real provider calls, not hand-fired events."""
    actions.machine.fire(session, Event.SESSION_STARTED)
    actions.set_language(session, "hi-IN")
    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.record_consent(session)
    actions.machine.fire(session, Event.CONSENT_GIVEN)

    actions.check_eligibility(session, **GOOD_APPLICANT)

    # Only what is still missing. Screening already captured the mobile number,
    # and asking a second time would be both rude and -- since the required set
    # would already be complete -- refused by the machine.
    for name in sorted(session.missing_fields):
        actions.save_applicant_field(session, name, "provided")

    actions.start_video_kyc(session)
    return session


# ---------------------------------------------------------------------------
# The invariant.
# ---------------------------------------------------------------------------


def test_actions_never_fires_a_user_intent_event():
    """Read the source and check every `Event.X` handed to `fire`.

    Structural rather than behavioural: it catches a violation the moment it is
    written, including on a branch no test happens to exercise.
    """
    tree = ast.parse(inspect.getsource(actions_module))

    fired: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not (isinstance(target, ast.Attribute) and target.attr == "fire"):
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Attribute)
                and isinstance(argument.value, ast.Name)
                and argument.value.id == "Event"
            ):
                fired.add(argument.attr)

    assert fired, "found no fire(Event.X) calls -- the parser is broken, not the code"

    fact_names = {e.name for e in FACT_EVENTS}
    intent_names = {e.name for e in USER_INTENT_EVENTS}

    assert fired <= fact_names, f"actions.py fires non-fact events: {fired - fact_names}"
    assert not (fired & intent_names)


def test_recording_tools_do_not_move_the_machine(rig):
    """Consent and language are intents; recording one is not a transition."""
    actions, _ = rig
    session = Session(session_id="record")
    actions.machine.fire(session, Event.SESSION_STARTED)

    actions.set_language(session, "ta-IN")
    assert session.state is State.LANGUAGE_SELECT
    assert session.language == "ta-IN"

    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.record_consent(session)
    assert session.state is State.CONSENT
    assert session.consent_recorded


# ---------------------------------------------------------------------------
# Binding is checked before the provider is called, not after.
# ---------------------------------------------------------------------------


def test_an_out_of_state_tool_never_reaches_the_provider(rig):
    """The order matters: a late check means the money already moved."""
    actions, _ = rig
    session = Session(session_id="early", state=State.ELIGIBILITY)
    session.account_number = "00000000001"

    with pytest.raises(ToolNotBoundError):
        actions.initiate_deposit(session, 50_000)

    assert actions.payments.balance_paise("00000000001") == 0, "provider was called anyway"


def test_opening_an_account_from_the_wrong_state_is_refused(rig):
    actions, _ = rig
    session = Session(session_id="wrong", state=State.COLLECT_DATA)
    session.kyc_reference = "vckyc-anything"

    with pytest.raises(ToolNotBoundError):
        actions.open_account(session)

    assert session.account_number is None


# ---------------------------------------------------------------------------
# Eligibility now runs off a real lookup.
# ---------------------------------------------------------------------------


def test_eligibility_reads_the_existing_account_flag_from_core_banking(rig):
    actions, _ = rig
    actions.core_banking.existing_customers.add("9876543210")

    session = Session(session_id="existing")
    actions.machine.fire(session, Event.SESSION_STARTED)
    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.machine.fire(session, Event.CONSENT_GIVEN)

    actions.check_eligibility(session, **{**GOOD_APPLICANT, "mobile": "9876543210"})

    assert session.state is State.DECLINED
    assert session.eligibility.decline_reasons == (DeclineReason.ALREADY_A_CUSTOMER,)


def test_the_screening_input_is_stored_for_replay(rig):
    """Storing only the verdict makes 'why in March' unanswerable later."""
    actions, _ = rig
    session = Session(session_id="replay")
    actions.machine.fire(session, Event.SESSION_STARTED)
    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.machine.fire(session, Event.CONSENT_GIVEN)
    actions.check_eligibility(session, **GOOD_APPLICANT)

    assert session.eligibility_input is not None
    assert session.eligibility_input.as_of == AS_OF

    # The stored input reproduces the stored decision exactly.
    from app.flow.eligibility import evaluate

    assert evaluate(session.eligibility_input) == session.eligibility


def test_screening_carries_the_mobile_forward_so_it_is_not_asked_twice(rig):
    """A small thing that matters on a voice call.

    The mobile number is captured during screening, so COLLECT_DATA starts with
    it already filled and never asks again. Re-asking would cost a turn, and on
    a voice call with a nervous first-time user, turns are the expensive thing.
    """
    actions, _ = rig
    session = Session(session_id="no-repeat")
    actions.machine.fire(session, Event.SESSION_STARTED)
    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.machine.fire(session, Event.CONSENT_GIVEN)
    actions.check_eligibility(session, **GOOD_APPLICANT)

    assert session.state is State.COLLECT_DATA
    assert session.applicant["mobile"] == "9000000000"
    assert "mobile" not in session.missing_fields

    # And the machine advances on the last genuinely missing field, not on a
    # fixed count of tool calls.
    for name in sorted(session.missing_fields):
        actions.save_applicant_field(session, name, "provided")

    assert session.state is State.KYC_HANDOFF


def test_the_screening_input_survives_storage(rig, store):
    actions, _ = rig
    session = Session(session_id="round-trip")
    actions.machine.fire(session, Event.SESSION_STARTED)
    actions.machine.fire(session, Event.LANGUAGE_CHOSEN)
    actions.machine.fire(session, Event.CONSENT_GIVEN)
    actions.check_eligibility(session, **GOOD_APPLICANT)

    store.save(session)
    loaded = store.load("round-trip")

    assert loaded.eligibility_input == session.eligibility_input
    assert loaded.eligibility == session.eligibility


# ---------------------------------------------------------------------------
# KYC through the real provider.
# ---------------------------------------------------------------------------


def test_still_pending_moves_nothing(rig):
    """The applicant may not have joined yet. That is not a failure."""
    actions, _ = rig
    actions.kyc.default_outcome = (KycStatus.PENDING, None)

    session = drive_to_kyc_pending(actions, Session(session_id="waiting"))
    assert session.state is State.KYC_PENDING

    actions.check_kyc_status(session)
    actions.check_kyc_status(session)

    assert session.state is State.KYC_PENDING


def test_a_dropped_call_costs_an_attempt_not_the_journey(rig):
    actions, _ = rig
    actions.kyc.outcomes = [(KycStatus.FAILED, KycFailureReason.NETWORK_DROPPED)]

    session = drive_to_kyc_pending(actions, Session(session_id="retry"))
    assert session.kyc_attempts == 1

    actions.check_kyc_status(session)
    assert session.state is State.KYC_HANDOFF

    actions.start_video_kyc(session)
    assert session.kyc_attempts == 2
    assert session.state is State.KYC_PENDING


def test_attempts_are_counted_at_dispatch_so_abandoning_cannot_loop_forever(rig):
    """Counting only failures would let someone hang up each time, forever."""
    actions, _ = rig
    actions.kyc.default_outcome = (KycStatus.FAILED, KycFailureReason.NETWORK_DROPPED)

    session = drive_to_kyc_pending(actions, Session(session_id="budget"))

    for _ in range(MAX_KYC_ATTEMPTS - 1):
        actions.check_kyc_status(session)
        assert session.state is State.KYC_HANDOFF
        actions.start_video_kyc(session)

    assert session.kyc_attempts == MAX_KYC_ATTEMPTS
    actions.check_kyc_status(session)
    assert session.state is State.HUMAN_HANDOFF


def test_an_expired_link_is_treated_like_a_failure(rig):
    actions, _ = rig
    actions.kyc.outcomes = [(KycStatus.EXPIRED, KycFailureReason.LINK_EXPIRED)]

    session = drive_to_kyc_pending(actions, Session(session_id="expired"))
    actions.check_kyc_status(session)

    assert session.state is State.KYC_HANDOFF


def test_polling_without_a_reference_is_refused(rig):
    actions, _ = rig
    session = Session(session_id="no-ref", state=State.KYC_PENDING)

    with pytest.raises(ActionRefused):
        actions.check_kyc_status(session)


# ---------------------------------------------------------------------------
# Account opening carries the real reference through.
# ---------------------------------------------------------------------------


def test_the_reference_from_kyc_is_the_one_core_banking_receives(rig):
    """End to end, not a placeholder: what the vendor issued is what is used."""
    actions, _ = rig
    session = drive_to_kyc_pending(actions, Session(session_id="carry"))
    issued = session.kyc_reference

    actions.core_banking.valid_kyc_references = {issued}

    actions.check_kyc_status(session)
    assert session.state is State.ACCOUNT_CREATE

    actions.open_account(session)
    assert session.state is State.FUNDING
    assert session.account_number


def test_core_banking_failure_routes_to_a_human(rig):
    actions, _ = rig
    session = drive_to_kyc_pending(actions, Session(session_id="cbs-down"))
    actions.check_kyc_status(session)

    actions.core_banking.fail_next_opening_with = AccountOpeningFailure.BRANCH_SYSTEM_DOWN
    actions.open_account(session)

    assert session.state is State.HUMAN_HANDOFF
    assert session.account_number is None


# ---------------------------------------------------------------------------
# Money.
# ---------------------------------------------------------------------------


def test_a_failed_deposit_leaves_you_in_funding_to_try_again(rig):
    """Failing to add money is not a change of state."""
    actions, _ = rig
    session = drive_to_kyc_pending(actions, Session(session_id="dep-fail"))
    actions.check_kyc_status(session)
    actions.open_account(session)
    assert session.state is State.FUNDING

    actions.payments.fail_next_with = PaymentFailure.RAIL_UNAVAILABLE
    actions.initiate_deposit(session, 50_000)

    assert session.state is State.FUNDING
    assert session.deposit_reference is None


def test_the_first_payment_ceiling_is_our_policy_not_the_rails(rig):
    actions, _ = rig
    session = drive_to_kyc_pending(actions, Session(session_id="ceiling"))
    actions.check_kyc_status(session)
    actions.open_account(session)
    actions.initiate_deposit(session, 10_000_000)
    assert session.state is State.FIRST_TRANSACTION

    with pytest.raises(ActionRefused, match="capped"):
        actions.make_first_payment(session, "shop@upi", FIRST_TRANSACTION_CEILING_PAISE + 1)

    assert session.state is State.FIRST_TRANSACTION


# ---------------------------------------------------------------------------
# The whole journey, driven only by provider calls.
# ---------------------------------------------------------------------------


def test_a_complete_journey_runs_on_real_provider_calls(rig):
    actions, audit = rig
    session = drive_to_kyc_pending(actions, Session(session_id="whole"))

    actions.check_kyc_status(session)
    actions.open_account(session)
    actions.initiate_deposit(session, 50_000)          # Rs 500 in
    actions.make_first_payment(session, "kirana@upi", 12_550)  # Rs 125.50 out

    assert session.state is State.COMPLETED
    assert session.first_payment_reference
    assert actions.payments.balance_paise(session.account_number) == 37_450

    trail = audit.for_session("whole")
    assert [e.sequence for e in trail] == list(range(len(trail)))
    assert trail[-1].to_state == State.COMPLETED
    assert any(e.kind is AuditKind.TOOL_ALLOWED for e in trail)
    assert any(e.kind is AuditKind.TRANSITION_ACCEPTED for e in trail)
