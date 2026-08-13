"""Per-state tool binding: the machine enforces it, not the prompt.

Same generated-complement trick as the transition tests. For every state, every
tool NOT bound to it must be refused -- 14 x 10 = 140 pairs, computed rather
than listed.
"""

from __future__ import annotations

import pytest

from app.audit import AuditKind, InMemoryAuditSink
from app.flow.machine import Machine, ToolNotBoundError
from app.flow.states import STATE_TOOLS, TERMINAL_STATES, State, ToolName
from app.session import Session

BOUND_PAIRS = [(s, t) for s in State for t in ToolName if t in STATE_TOOLS[s]]
UNBOUND_PAIRS = [(s, t) for s in State for t in ToolName if t not in STATE_TOOLS[s]]


@pytest.mark.parametrize("state,tool", UNBOUND_PAIRS, ids=lambda v: str(v))
def test_unbound_tool_is_refused(machine: Machine, state: State, tool: ToolName):
    session = Session(session_id="binding", state=state)

    with pytest.raises(ToolNotBoundError) as caught:
        machine.assert_tool_allowed(session, tool)

    assert caught.value.state is state
    assert caught.value.tool is tool


@pytest.mark.parametrize("state,tool", BOUND_PAIRS, ids=lambda v: str(v))
def test_bound_tool_is_permitted(machine: Machine, state: State, tool: ToolName):
    session = Session(session_id="binding", state=state)
    machine.assert_tool_allowed(session, tool)  # must not raise


# ---------------------------------------------------------------------------
# The named cases from the brief.
# ---------------------------------------------------------------------------


def test_bank_details_cannot_be_collected_during_screening(machine: Machine):
    """SAVE_APPLICANT_FIELD is legal from COLLECT_DATA and impossible from ELIGIBILITY."""
    machine.assert_tool_allowed(
        Session(session_id="ok", state=State.COLLECT_DATA), ToolName.SAVE_APPLICANT_FIELD
    )

    with pytest.raises(ToolNotBoundError):
        machine.assert_tool_allowed(
            Session(session_id="no", state=State.ELIGIBILITY), ToolName.SAVE_APPLICANT_FIELD
        )


def test_open_account_exists_in_exactly_one_state():
    states_binding_it = {s for s in State if ToolName.OPEN_ACCOUNT in STATE_TOOLS[s]}
    assert states_binding_it == {State.ACCOUNT_CREATE}


def test_kyc_pending_can_only_poll(machine: Machine):
    """During the regulatory handoff the agent's hands are tied."""
    assert STATE_TOOLS[State.KYC_PENDING] == frozenset(
        {ToolName.CHECK_KYC_STATUS, ToolName.SET_LANGUAGE, ToolName.ESCALATE_TO_HUMAN}
    )


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES), ids=lambda v: str(v))
def test_terminal_states_bind_no_tools(machine: Machine, state: State):
    assert STATE_TOOLS[state] == frozenset()
    for tool in ToolName:
        with pytest.raises(ToolNotBoundError):
            machine.assert_tool_allowed(Session(session_id="done", state=state), tool)


def test_escape_hatches_are_available_everywhere_live():
    """'Get me a person' and 'change my language' must never be refused mid-flow."""
    for state in State:
        if state in TERMINAL_STATES:
            continue
        assert ToolName.ESCALATE_TO_HUMAN in STATE_TOOLS[state], state
        assert ToolName.SET_LANGUAGE in STATE_TOOLS[state], state


# ---------------------------------------------------------------------------
# The mis-selling boundary.
# ---------------------------------------------------------------------------


def test_no_cross_sell_tool_exists_anywhere():
    """The risk is closed by absence, not by instruction.

    A model cannot offer a loan it has no tool to offer. This asserts on the
    registry itself, so adding `offer_personal_loan` to ToolName fails here
    even before anyone binds it to a state.
    """
    forbidden_stems = ("offer", "upsell", "cross_sell", "recommend", "loan", "credit", "insurance")
    for tool in ToolName:
        assert not any(stem in str(tool).lower() for stem in forbidden_stems), (
            f"{tool} looks like a sales tool; a state machine that can sell "
            f"is a mis-selling incident waiting to happen"
        )


def test_every_declared_tool_is_bound_somewhere():
    """No orphan tools. An unbound tool is either dead code or a forgotten binding."""
    bound = set().union(*STATE_TOOLS.values())
    assert bound == set(ToolName), f"never usable: {set(ToolName) - bound}"


# ---------------------------------------------------------------------------
# Refusals must be recorded, or they are not auditable.
# ---------------------------------------------------------------------------


def test_a_refused_tool_is_written_to_the_audit_trail(audit: InMemoryAuditSink):
    machine = Machine(audit)
    session = Session(session_id="audited", state=State.ELIGIBILITY)

    with pytest.raises(ToolNotBoundError):
        machine.assert_tool_allowed(session, ToolName.OPEN_ACCOUNT)

    trail = audit.for_session("audited")
    assert len(trail) == 1
    assert trail[0].kind is AuditKind.TOOL_REFUSED
    assert trail[0].tool == ToolName.OPEN_ACCOUNT
    assert trail[0].from_state == State.ELIGIBILITY
