"""Tests that try to break the machine and assert it refuses.

The centrepiece is `test_illegal_pair_is_refused`. It is not a list of illegal
moves someone thought of -- it is the arithmetic complement of the map. Every
(state, event) pair that is not in TRANSITIONS must raise, and the parameter
list is computed, not written. Add a state to states.py tomorrow and this file
generates the new refusal cases for it without being edited.

That is the property an if-statement implementation cannot give you: you can
only test the branches you remembered, which are the same branches you
remembered to write.
"""

from __future__ import annotations

import pytest

from app.flow.machine import IllegalTransitionError, Machine
from app.flow.states import (
    STATE_TOOLS,
    TERMINAL_STATES,
    TRANSITIONS,
    Event,
    State,
)
from app.session import Session

ALL_PAIRS = [(state, event) for state in State for event in Event]
LEGAL_PAIRS = sorted(TRANSITIONS.keys())
ILLEGAL_PAIRS = [pair for pair in ALL_PAIRS if pair not in TRANSITIONS]


def test_the_map_is_mostly_refusal():
    """Sanity check on the arithmetic, so a broken import can't silently pass."""
    assert len(ALL_PAIRS) == len(State) * len(Event)
    assert len(LEGAL_PAIRS) + len(ILLEGAL_PAIRS) == len(ALL_PAIRS)
    assert len(ILLEGAL_PAIRS) > len(LEGAL_PAIRS)


@pytest.mark.parametrize("state,event", ILLEGAL_PAIRS, ids=lambda v: str(v))
def test_illegal_pair_is_refused(machine: Machine, state: State, event: Event):
    """Every pair absent from the map must raise. All of them, generated."""
    session = Session(session_id="illegal", state=state)

    with pytest.raises(IllegalTransitionError) as caught:
        machine.fire(session, event)

    # The session must not have moved.
    assert session.state is state
    assert caught.value.state is state
    assert caught.value.event is event


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES), ids=lambda v: str(v))
@pytest.mark.parametrize("event", sorted(Event), ids=lambda v: str(v))
def test_terminal_states_accept_nothing(machine: Machine, state: State, event: Event):
    """A finished session is finished. No event of any kind restarts it."""
    session = Session(session_id="terminal", state=state)

    with pytest.raises(IllegalTransitionError):
        machine.fire(session, event)

    assert session.state is state


# ---------------------------------------------------------------------------
# The specific illegal moves worth naming, because they are the ones that
# would be regulatory incidents rather than bugs.
# ---------------------------------------------------------------------------


def entrances_to(target: State) -> set[tuple[State, Event]]:
    """Every (state, event) that ENTERS `target` from somewhere else.

    Self-loops are excluded deliberately. The language self-loops mean most
    states have an edge pointing at themselves, and staying put is not the same
    as getting in -- which is the only thing these guarantees are about.
    """
    return {
        (state, event)
        for (state, event), candidates in TRANSITIONS.items()
        for t in candidates
        if t.to is target and state is not target
    }


def test_cannot_open_an_account_without_passing_kyc(machine: Machine):
    """ACCOUNT_CREATE is enterable only from KYC_PENDING, only via KYC_VERIFIED."""
    assert entrances_to(State.ACCOUNT_CREATE) == {(State.KYC_PENDING, Event.KYC_VERIFIED)}


def test_cannot_skip_from_data_collection_straight_to_an_account(machine: Machine):
    session = Session(session_id="skip", state=State.COLLECT_DATA)

    with pytest.raises(IllegalTransitionError):
        machine.fire(session, Event.ACCOUNT_OPENED)

    assert session.state is State.COLLECT_DATA


def test_consent_cannot_be_bypassed():
    """ELIGIBILITY is enterable only through CONSENT_GIVEN.

    DPDP Act 2023: no processing of personal data without notice and consent,
    so the gate has to be structural rather than a step we remember to take.
    """
    assert entrances_to(State.ELIGIBILITY) == {(State.CONSENT, Event.CONSENT_GIVEN)}


def test_data_collection_is_enterable_only_after_passing_eligibility():
    assert entrances_to(State.COLLECT_DATA) == {(State.ELIGIBILITY, Event.ELIGIBILITY_CHECKED)}


def test_completed_is_enterable_only_by_a_confirmed_payment():
    """The success metric has exactly one door, and money has to move through it."""
    assert entrances_to(State.COMPLETED) == {(State.FIRST_TRANSACTION, Event.PAYMENT_CONFIRMED)}


def test_kyc_pending_does_not_time_out(machine: Machine):
    """The suspension may last hours. Timing it out would defeat the handoff."""
    assert (State.KYC_PENDING, Event.SESSION_TIMEOUT) not in TRANSITIONS

    session = Session(session_id="patient", state=State.KYC_PENDING)
    with pytest.raises(IllegalTransitionError):
        machine.fire(session, Event.SESSION_TIMEOUT)

    # Every other live state does time out -- this exclusion is specific.
    for state in State:
        if state in TERMINAL_STATES or state is State.KYC_PENDING:
            continue
        assert (state, Event.SESSION_TIMEOUT) in TRANSITIONS


# ---------------------------------------------------------------------------
# Graph integrity. These stop the map rotting as states get added.
# ---------------------------------------------------------------------------


def test_every_state_has_a_tool_binding():
    assert set(STATE_TOOLS) == set(State)


def test_no_dead_ends():
    """Every non-terminal state has at least one way out."""
    for state in State:
        if state in TERMINAL_STATES:
            continue
        assert any(s is state for (s, _) in TRANSITIONS), f"{state} traps the user"


def test_terminal_states_have_no_outgoing_edges():
    for state in TERMINAL_STATES:
        assert not [1 for (s, _) in TRANSITIONS if s is state]


def test_every_state_is_reachable_from_start():
    seen = {State.START}
    frontier = [State.START]
    while frontier:
        current = frontier.pop()
        for (state, _), candidates in TRANSITIONS.items():
            if state is not current:
                continue
            for transition in candidates:
                if transition.to not in seen:
                    seen.add(transition.to)
                    frontier.append(transition.to)

    assert seen == set(State), f"unreachable: {set(State) - seen}"


def test_every_row_ends_in_an_unconditional_fallback():
    """Totality: guards may narrow, but something must always match.

    This is what makes NoGuardMatchedError structurally unreachable today. It
    is kept as a guard against a future edit that adds a guarded candidate
    without a fallback, which would strand a session mid-flow.
    """
    for key, candidates in TRANSITIONS.items():
        assert candidates, f"{key} has no candidates"
        assert candidates[-1].guard is None, f"{key} has no unconditional fallback"


def test_no_candidate_is_shadowed_by_an_earlier_fallback():
    """Anything after an unconditional candidate is dead code."""
    for key, candidates in TRANSITIONS.items():
        for index, transition in enumerate(candidates[:-1]):
            assert transition.guard is not None, f"{key} candidate {index} shadows the rest"


@pytest.mark.parametrize("key", LEGAL_PAIRS, ids=lambda v: f"{v[0]}-{v[1]}")
def test_guards_are_pure(key):
    """A guard asked twice must answer twice the same.

    A weak check -- it cannot prove purity -- but it catches the realistic
    mistake, which is a guard that reads a clock or mutates the session it was
    handed.
    """
    session = Session(session_id="purity", state=key[0])
    for transition in TRANSITIONS[key]:
        if transition.guard is None:
            continue
        before = session.to_dict()
        first = transition.guard(session)
        second = transition.guard(session)
        assert first == second, f"{key} -> {transition.to} guard is not deterministic"
        assert session.to_dict() == before, f"{key} -> {transition.to} guard mutated the session"
