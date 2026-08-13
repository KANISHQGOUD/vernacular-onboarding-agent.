"""End-to-end journeys: the happy path, KYC retry, and the suspend/resume handoff."""

from __future__ import annotations

import pytest

from app.audit import AuditKind, InMemoryAuditSink
from app.flow.eligibility import evaluate
from app.flow.machine import IllegalTransitionError, Machine
from app.flow.states import MAX_KYC_ATTEMPTS, Event, State
from app.session import InMemorySessionStore, Session
from tests.conftest import complete_applicant_data, eligible_applicant


def _walk_to_kyc_pending(machine: Machine, session: Session) -> Session:
    """Drive a qualifying applicant as far as the regulatory boundary."""
    machine.fire(session, Event.SESSION_STARTED)

    session.language = "hi-IN"
    machine.fire(session, Event.LANGUAGE_CHOSEN)

    session.consent_recorded = True
    machine.fire(session, Event.CONSENT_GIVEN)

    session.eligibility = evaluate(eligible_applicant())
    machine.fire(session, Event.ELIGIBILITY_CHECKED)

    session.applicant = complete_applicant_data()
    machine.fire(session, Event.REQUIRED_DATA_COMPLETE)

    session.kyc_reference = "vckyc-ref-001"
    machine.fire(session, Event.KYC_DISPATCHED)
    return session


def test_happy_path_reaches_completed(machine: Machine, session: Session):
    _walk_to_kyc_pending(machine, session)
    assert session.state is State.KYC_PENDING

    machine.fire(session, Event.KYC_VERIFIED)
    session.account_number = "0000123456789"
    machine.fire(session, Event.ACCOUNT_OPENED)

    session.deposit_reference = "dep-001"
    machine.fire(session, Event.DEPOSIT_CONFIRMED)

    session.first_payment_reference = "upi-001"
    machine.fire(session, Event.PAYMENT_CONFIRMED)

    assert session.state is State.COMPLETED


def test_opening_the_account_is_not_the_finish_line(machine: Machine, session: Session):
    """The metric is first-transaction completion, so the graph says so.

    An applicant who reaches ACCOUNT_OPENED and stops has produced exactly the
    dormant account this product exists to prevent, and the machine records
    that as ABANDONED rather than as any kind of success.
    """
    _walk_to_kyc_pending(machine, session)
    machine.fire(session, Event.KYC_VERIFIED)
    machine.fire(session, Event.ACCOUNT_OPENED)

    assert session.state is State.FUNDING
    assert session.state is not State.COMPLETED

    machine.fire(session, Event.USER_QUIT)
    assert session.state is State.ABANDONED


def test_ineligible_applicant_is_declined_with_a_reason(machine: Machine, session: Session):
    machine.fire(session, Event.SESSION_STARTED)
    machine.fire(session, Event.LANGUAGE_CHOSEN)
    machine.fire(session, Event.CONSENT_GIVEN)

    session.eligibility = evaluate(eligible_applicant(has_existing_active_account=True))
    machine.fire(session, Event.ELIGIBILITY_CHECKED)

    assert session.state is State.DECLINED
    assert session.eligibility.decline_reasons  # the reason is stored, not improvised


def test_data_collection_loops_until_every_field_is_present(machine: Machine, session: Session):
    machine.fire(session, Event.SESSION_STARTED)
    machine.fire(session, Event.LANGUAGE_CHOSEN)
    machine.fire(session, Event.CONSENT_GIVEN)
    session.eligibility = evaluate(eligible_applicant())
    machine.fire(session, Event.ELIGIBILITY_CHECKED)
    assert session.state is State.COLLECT_DATA

    for field_name in ("full_name", "date_of_birth", "address"):
        session.applicant[field_name] = "provided"
        machine.fire(session, Event.DATA_PROVIDED)
        assert session.state is State.COLLECT_DATA

    assert not session.required_data_complete
    assert session.missing_fields == {"occupation", "mobile"}


# ---------------------------------------------------------------------------
# KYC retry and escalation.
# ---------------------------------------------------------------------------


def test_a_single_kyc_failure_is_not_a_rejection(machine: Machine, session: Session):
    """Bad light and dropped 4G must not end the journey."""
    _walk_to_kyc_pending(machine, session)

    session.kyc_attempts += 1
    machine.fire(session, Event.KYC_FAILED)

    assert session.state is State.KYC_HANDOFF


def test_kyc_escalates_to_a_human_once_the_budget_is_spent(machine: Machine, session: Session):
    _walk_to_kyc_pending(machine, session)

    for attempt in range(1, MAX_KYC_ATTEMPTS):
        session.kyc_attempts = attempt
        machine.fire(session, Event.KYC_FAILED)
        assert session.state is State.KYC_HANDOFF
        machine.fire(session, Event.KYC_DISPATCHED)

    session.kyc_attempts = MAX_KYC_ATTEMPTS
    machine.fire(session, Event.KYC_FAILED)

    assert session.state is State.HUMAN_HANDOFF


def test_a_human_can_be_requested_from_any_live_state(machine: Machine):
    for state in State:
        if machine.is_terminal(state):
            continue
        session = Session(session_id=f"escape-{state}", state=state)
        machine.fire(session, Event.USER_WANTS_HUMAN)
        assert session.state is State.HUMAN_HANDOFF


def test_language_can_be_changed_without_losing_your_place(machine: Machine):
    """The user most likely to ask is the one already struggling, deep in the flow."""
    for state in State:
        if machine.is_terminal(state) or state is State.LANGUAGE_SELECT:
            continue
        session = Session(session_id=f"lang-{state}", state=state)
        machine.fire(session, Event.LANGUAGE_CHANGE_REQUESTED)
        assert session.state is state


# ---------------------------------------------------------------------------
# The handoff: a session that survives the process that created it.
# ---------------------------------------------------------------------------


def test_a_suspended_session_resumes_hours_later_in_another_process(
    machine: Machine, store: InMemorySessionStore
):
    """The regulatory boundary as a feature.

    Video KYC happens outside our process. The user may finish it that evening,
    at a branch, on another device. The store deliberately keeps only the
    serialised row, so `loaded` is a genuinely different object rebuilt from
    storage -- not the one we handed it.
    """
    original = Session(session_id="resume-me")
    _walk_to_kyc_pending(machine, original)
    store.save(original)

    del original  # the process ends here

    loaded = store.load("resume-me")
    assert loaded is not None
    assert loaded.state is State.KYC_PENDING
    assert loaded.language == "hi-IN"
    assert loaded.kyc_reference == "vckyc-ref-001"
    assert loaded.eligibility is not None and loaded.eligibility.passed
    assert loaded.required_data_complete

    machine.fire(loaded, Event.KYC_VERIFIED)
    machine.fire(loaded, Event.ACCOUNT_OPENED)
    machine.fire(loaded, Event.DEPOSIT_CONFIRMED)
    machine.fire(loaded, Event.PAYMENT_CONFIRMED)

    assert loaded.state is State.COMPLETED


def test_an_unknown_session_loads_as_none(store: InMemorySessionStore):
    assert store.load("never-existed") is None


# ---------------------------------------------------------------------------
# The audit trail.
# ---------------------------------------------------------------------------


def test_the_happy_path_leaves_a_complete_ordered_trail(audit: InMemoryAuditSink):
    machine = Machine(audit)
    session = Session(session_id="audited-journey")

    _walk_to_kyc_pending(machine, session)
    machine.fire(session, Event.KYC_VERIFIED)
    machine.fire(session, Event.ACCOUNT_OPENED)
    machine.fire(session, Event.DEPOSIT_CONFIRMED)
    machine.fire(session, Event.PAYMENT_CONFIRMED)

    trail = audit.for_session("audited-journey")

    assert [e.sequence for e in trail] == list(range(len(trail))), "sequence has gaps"
    assert all(e.kind is AuditKind.TRANSITION_ACCEPTED for e in trail)
    assert trail[0].from_state == State.START
    assert trail[-1].to_state == State.COMPLETED

    # Each row's destination is the next row's origin: an unbroken chain.
    for earlier, later in zip(trail, trail[1:]):
        assert earlier.to_state == later.from_state


def test_refusals_are_recorded_too(audit: InMemoryAuditSink):
    """A log of what worked proves nothing about what didn't."""
    machine = Machine(audit)
    session = Session(session_id="refused", state=State.COLLECT_DATA)

    with pytest.raises(IllegalTransitionError):
        machine.fire(session, Event.ACCOUNT_OPENED)

    trail = audit.for_session("refused")
    assert len(trail) == 1
    assert trail[0].kind is AuditKind.TRANSITION_REFUSED
    assert trail[0].from_state == State.COLLECT_DATA
    assert trail[0].to_state is None
    assert trail[0].event == Event.ACCOUNT_OPENED


def test_the_trail_cannot_be_edited_through_the_reader(audit: InMemoryAuditSink):
    machine = Machine(audit)
    session = Session(session_id="immutable")
    machine.fire(session, Event.SESSION_STARTED)

    trail = audit.for_session("immutable")
    assert isinstance(trail, tuple)
    with pytest.raises((AttributeError, TypeError)):
        trail[0].detail = "tampered"  # type: ignore[misc]
