"""End-to-end journeys: the happy path, KYC retry, and the suspend/resume handoff."""

from __future__ import annotations

import pytest

from app.audit import AuditKind, InMemoryAuditSink
from app.flow.eligibility import evaluate
from app.flow.machine import IllegalTransitionError, Machine
from app.flow.states import MAX_KYC_ATTEMPTS, TERMINAL_STATES, Event, State
from app.flow.actions import Actions
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import MockKycProvider
from app.providers.payments import MockPaymentRail
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


# ---------------------------------------------------------------------------
# Saying no is an answer. Not answering is not.
# ---------------------------------------------------------------------------


def _at_consent(machine: Machine, session: Session) -> Session:
    machine.fire(session, Event.SESSION_STARTED)
    session.language = "hi-IN"
    machine.fire(session, Event.LANGUAGE_CHOSEN)
    assert session.state is State.CONSENT
    return session


def test_refusing_consent_is_kept_apart_from_giving_up(machine: Machine):
    """Three ways to leave the consent gate, and they are three events.

    Folded together, "the notice is badly worded" and "the form is too long"
    and "the network here is bad" all look like one number going up, and that
    number cannot tell you which to fix -- rewrite the notice, shorten the
    form, or handle silence. The success metric is first-transaction
    completion, so every exit has to be attributable.

    They are also not equally recoverable, which is the part that matters for
    what happens next: somebody who refused said no, somebody who quit chose
    to stop, and somebody who went quiet may be trying to get back in.
    """
    refused = _at_consent(machine, Session(session_id="refused"))
    machine.fire(refused, Event.CONSENT_REFUSED)

    quit_out = _at_consent(machine, Session(session_id="quit"))
    machine.fire(quit_out, Event.USER_QUIT)

    assert refused.state is State.CONSENT_REFUSED, "a refusal is an answer"
    assert quit_out.state is State.ABANDONED, "quitting is a decision to stop"

    # No two ways of finishing may share a state. Written as a property so
    # that adding a third and quietly pointing it at an existing one fails.
    assert len({refused.state, quit_out.state}) == 2

    # And the third way of not finishing is not an ending at all: going quiet
    # leaves the session exactly where it was, waiting. See the lapse tests.
    lapsed = _at_consent(machine, Session(session_id="quiet"))
    _actions(machine).record_lapse(lapsed)
    assert lapsed.state is State.CONSENT, "silence moved the session"


def test_a_refusal_is_answered_without_escalating_to_a_human():
    """The bug this split actually fixes.

    While a refusal landed in ABANDONED it inherited HANDING_TO_HUMAN, so
    somebody who said "do not process my data" was answered with "let me get
    someone for you" -- an escalation triggered by the word no. Nothing in the
    state machine was wrong; the phrasing map was reading a state that meant
    three different things.
    """
    from app.conversation import STATE_SPEECH
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    kind = STATE_SPEECH[State.CONSENT_REFUSED]

    assert kind is not SpeechActKind.HANDING_TO_HUMAN
    assert kind is SpeechActKind.ACKNOWLEDGE_NO_CONSENT

    # The wording is the model's job, but the fallback ships with the code and
    # is what a caller hears when the API is unreachable -- so it is worth
    # pinning that it accepts the answer rather than reopening the question.
    spoken = SPEECH_ACTS[kind].fallback.lower()
    assert "start again" in spoken, "leave the door open"
    for pressure in ("why", "are you sure", "instead"):
        assert pressure not in spoken, f"the fallback pushes back with {pressure!r}"


def test_refusing_consent_is_terminal(machine: Machine):
    """No consent, no processing -- including no way back in through an event."""
    session = _at_consent(machine, Session(session_id="terminal"))
    machine.fire(session, Event.CONSENT_REFUSED)

    assert session.state in TERMINAL_STATES
    for event in Event:
        with pytest.raises(IllegalTransitionError):
            machine.fire(session, event)
    assert session.state is State.CONSENT_REFUSED


def test_nothing_personal_was_collected_before_the_refusal(machine: Machine):
    """Why this edge needs no erasure step.

    Consent gates ELIGIBILITY and COLLECT_DATA, so a refusal happens before
    any question that collects personal data. If that ordering ever changed,
    the transition would start stranding data belonging to someone who
    explicitly refused -- so the ordering is asserted, not assumed.
    """
    session = _at_consent(machine, Session(session_id="clean"))
    machine.fire(session, Event.CONSENT_REFUSED)

    assert session.applicant == {}
    assert session.eligibility_input is None
    assert session.eligibility is None
    assert session.account_number is None


def test_the_refusal_is_written_down(machine: Machine, audit: InMemoryAuditSink):
    """A lawful stop still has to be provable afterwards."""
    session = _at_consent(machine, Session(session_id="recorded"))
    machine.fire(session, Event.CONSENT_REFUSED)

    (last,) = [
        record
        for record in audit.for_session("recorded")
        if record.kind is AuditKind.TRANSITION_ACCEPTED
        and record.to_state == str(State.CONSENT_REFUSED)
    ]
    assert last.from_state == str(State.CONSENT)
    assert last.event == str(Event.CONSENT_REFUSED)


def test_asking_to_stop_is_not_answered_with_an_escalation():
    """The second instance of the bug CONSENT_REFUSED had.

    ABANDONED was phrased as HANDING_TO_HUMAN, so asking to stop was answered
    by escalating to a person and promising they already had your details.
    """
    from app.conversation import STATE_SPEECH
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    stopped = STATE_SPEECH[State.ABANDONED]
    assert stopped is not SpeechActKind.HANDING_TO_HUMAN

    # Quitting is legal from FUNDING and FIRST_TRANSACTION, where the account
    # already exists -- so the fallback may not claim nothing was submitted.
    spoken = SPEECH_ACTS[stopped].fallback.lower()
    assert "saved" in spoken
    assert "nothing has been" not in spoken

    # And there is no act for going quiet, because there is no moment to say
    # it in: nobody is listening when it happens, and by the time they are
    # back the answer is whatever the state they are still in asks.
    assert not hasattr(SpeechActKind, "ANNOUNCE_TIMED_OUT")


def test_waiting_at_the_handoff_costs_nothing(machine: Machine, session: Session):
    """The deliberate omission that became the general rule.

    A video KYC handoff can take hours. Ending it for silence would punish
    people who did nothing but wait -- and once lapsing stopped being a
    transition, that became true of every state, not just this one.
    """
    _walk_to_kyc_pending(machine, session)

    for _ in range(3):
        _actions(machine).record_lapse(session)

    assert session.state is State.KYC_PENDING, "waiting cost them their place"
    assert session.lapses == 3

    # But a person who says they are done still gets to be done.
    machine.fire(session, Event.USER_QUIT)
    assert session.state is State.ABANDONED


def test_every_way_of_not_finishing_has_its_own_state():
    """A funnel is only readable if the exits are distinguishable.

    COMPLETED is the one success. Everything else is a different reason, and
    each needs its own bucket or the breakdown collapses back into "didn't
    finish", which is the number we already had and could not act on.
    """
    exits = {
        State.COMPLETED,
        State.DECLINED,
        State.CONSENT_REFUSED,
        State.ABANDONED,
        State.HUMAN_HANDOFF,
    }
    assert exits == set(TERMINAL_STATES)

    # Going quiet is deliberately absent: it is not a way of finishing, it is
    # a pause. Counting it as an outcome was the bug.
    for state in TERMINAL_STATES:
        assert "TIMED" not in str(state).upper()


# ---------------------------------------------------------------------------
# Lapsing. Not an ending, so not a transition.
# ---------------------------------------------------------------------------


def _actions(machine: Machine) -> Actions:
    return Actions(
        machine=machine,
        kyc=MockKycProvider(),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )


def test_going_quiet_leaves_the_session_exactly_where_it_was(machine: Machine):
    """The whole design, in one assertion.

    They stopped answering halfway through the form. They are still halfway
    through the form. Resuming needs no edge back into COLLECT_DATA, which is
    what keeps every entrance in the map singular.
    """
    session = _at_consent(machine, Session(session_id="quiet"))
    machine.fire(session, Event.CONSENT_GIVEN)
    session.eligibility = evaluate(eligible_applicant())
    machine.fire(session, Event.ELIGIBILITY_CHECKED)
    assert session.state is State.COLLECT_DATA

    _actions(machine).record_lapse(session)

    assert session.state is State.COLLECT_DATA
    assert session.lapses == 1
    assert session.has_lapsed

    # And they can carry straight on, because nothing was undone.
    session.applicant = complete_applicant_data()
    machine.fire(session, Event.REQUIRED_DATA_COMPLETE)
    assert session.state is State.KYC_HANDOFF


def test_a_lapse_is_written_down_without_claiming_a_transition(
    machine: Machine, audit: InMemoryAuditSink
):
    """It is a real event and belongs in the trail -- as itself.

    Recording it as TRANSITION_ACCEPTED would put a move in the audit log that
    never happened, which is exactly the kind of thing the trail exists to be
    trusted about.
    """
    session = _at_consent(machine, Session(session_id="recorded"))
    actions = _actions(machine)
    actions.record_lapse(session)
    actions.record_return(session)

    trail = audit.for_session("recorded")
    kinds = [record.kind for record in trail]

    assert AuditKind.SESSION_LAPSED in kinds
    assert AuditKind.SESSION_RETURNED in kinds

    for record in trail:
        if record.kind in (AuditKind.SESSION_LAPSED, AuditKind.SESSION_RETURNED):
            assert record.from_state == record.to_state, "a lapse claimed a move"


def test_the_machine_refuses_to_forge_a_transition_record(machine: Machine):
    """`record` is the only way to write a non-transition fact, and it is
    narrow on purpose: if anything could write TRANSITION_ACCEPTED, the trail
    would record what callers claimed rather than what the machine did."""
    session = Session(session_id="forge", state=State.CONSENT)

    for kind in (AuditKind.TRANSITION_ACCEPTED, AuditKind.TRANSITION_REFUSED):
        with pytest.raises(ValueError, match="fire"):
            machine.record(session, kind, detail="did not happen")


def test_lapses_survive_erasure(machine: Machine):
    """Operational metadata, not personal data -- which is why it lives in
    `notes`. A bank still has to be able to answer "how often did this
    application get interrupted" after it can no longer answer "whose"."""
    session = _at_consent(machine, Session(session_id="erased"))
    session.applicant = {"full_name": "Asha Devi"}
    _actions(machine).record_lapse(session)

    session.erase_personal_data()

    assert session.applicant == {}
    assert session.lapses == 1
    assert session.notes["lapsed_in"] == str(State.CONSENT)
