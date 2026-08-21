"""The engine. Reads the map in states.py and does nothing clever.

Its smallness is the feature. Every question about what this system permits is
answered by `states.py`; this file only carries the declarations out. If you
find yourself adding policy here -- a special case, an exception for one state
-- it belongs in the map instead, where it can be read and audited.

Two enforcement duties:

  fire()               a transition happens only if the map allows it
  assert_tool_allowed()  a tool runs only if the current state binds it

The second one is the load-bearing part of the whole architecture, so it gets
the longer explanation, down at the method.
"""

from __future__ import annotations

from app.audit import AuditKind, AuditSink
from app.flow.states import (
    STATE_TOOLS,
    TERMINAL_STATES,
    TRANSITIONS,
    Event,
    State,
    ToolName,
    Transition,
)
from app.session import Session


class FlowError(Exception):
    """Base for every refusal. Callers can catch this to mean 'the machine said no'."""


class IllegalTransitionError(FlowError):
    """The (state, event) pair is not in the map at all."""

    def __init__(self, state: State, event: Event, legal: frozenset[Event]) -> None:
        self.state = state
        self.event = event
        self.legal = legal
        legal_list = ", ".join(sorted(legal)) or "nothing (terminal state)"
        super().__init__(
            f"{event} is not legal in {state}. Legal here: {legal_list}."
        )


class NoGuardMatchedError(FlowError):
    """The pair exists but every guard rejected the session.

    Distinct from IllegalTransitionError on purpose. This one means the map has
    a hole -- an event was expected in this state but no candidate covers the
    session's actual data -- and it should be loud rather than silently
    treated as 'not allowed'.
    """

    def __init__(self, state: State, event: Event) -> None:
        self.state = state
        self.event = event
        super().__init__(
            f"{event} is legal in {state}, but no guard matched this session. "
            f"The map is incomplete for this data."
        )


class ToolNotBoundError(FlowError):
    """A tool was attempted from a state that does not bind it."""

    def __init__(self, state: State, tool: ToolName, allowed: frozenset[ToolName]) -> None:
        self.state = state
        self.tool = tool
        self.allowed = allowed
        allowed_list = ", ".join(sorted(allowed)) or "no tools (terminal state)"
        super().__init__(
            f"{tool} is not available in {state}. Available here: {allowed_list}."
        )


class Machine:
    """Drives a Session across the map, writing every decision to the audit trail."""

    def __init__(self, audit: AuditSink) -> None:
        self._audit = audit

    # -- Read-only queries about the map ------------------------------------

    @staticmethod
    def legal_events(state: State) -> frozenset[Event]:
        """Every event this state accepts. Empty for terminal states."""
        return frozenset(event for (s, event) in TRANSITIONS if s is state)

    @staticmethod
    def tools_for(state: State) -> frozenset[ToolName]:
        """The state's tool binding.

        In Phase 2 this is what becomes the tool/function declarations sent to
        the model -- so this method decides its entire action space for the
        turn, whichever provider is behind it.
        """
        return STATE_TOOLS[state]

    @staticmethod
    def is_terminal(state: State) -> bool:
        return state in TERMINAL_STATES

    # -- Enforcement --------------------------------------------------------

    def assert_tool_allowed(self, session: Session, tool: ToolName) -> None:
        """Refuse any tool the current state does not bind.

        This is the second of two layers, and the one that actually guarantees
        anything.

        Layer one is `tools_for()`: only the current state's schemas are sent
        to the model, so an unbound tool is not forbidden to it -- it is absent
        from its world. That stops essentially all wandering, because a model
        cannot call a function it was never shown.

        Layer two is this method, which re-checks at execution time. It exists
        because layer one lives inside the prompt, and a prompt is not a trust
        boundary. A model can hallucinate a plausible function name it saw in
        training. A stale tool call can arrive from a turn taken before the
        state changed. A future retry or replay can resubmit an old call. None
        of those are exotic -- they are ordinary distributed-systems mistakes,
        and in a bank the ordinary ones are the ones that matter.

        So the rule is: the prompt narrows what the model will try, and this
        method decides what actually runs. If those two ever disagree, this one
        wins and the disagreement is written down -- which also makes wandering
        MEASURABLE. `TOOL_REFUSED` rows are a metric, not just an error. A
        model that starts drifting after a prompt change shows up as a count.
        """
        allowed = STATE_TOOLS[session.state]
        if tool not in allowed:
            self._audit.append(
                session.session_id,
                AuditKind.TOOL_REFUSED,
                from_state=str(session.state),
                tool=str(tool),
                detail=f"not bound in {session.state}",
            )
            raise ToolNotBoundError(session.state, tool, allowed)

        self._audit.append(
            session.session_id,
            AuditKind.TOOL_ALLOWED,
            from_state=str(session.state),
            tool=str(tool),
        )

    def fire(self, session: Session, event: Event) -> Session:
        """Apply an event. Moves the session or raises; never silently does nothing.

        Terminal states need no special case here -- they have no rows in the
        map, so every event raises IllegalTransitionError naturally.

        FAIL CLOSED: THE RECORD IS WRITTEN BEFORE THE MOVE
        --------------------------------------------------
        The accepted branch audits first and assigns `session.state` second,
        and the order is the point. If the sink raises, the session has not
        moved and the caller sees the failure -- rather than advancing into a
        state with no row explaining how it got there.

        This used to be the other way round, and no test could catch it:
        `InMemoryAuditSink` cannot fail, so the ordering never mattered until a
        Postgres sink made a failed append reachable. The invariant worth
        stating is that a session never occupies a state the trail cannot
        account for, and in a flow where one of those states is "the account
        exists now", losing that is worse than refusing to proceed.
        """
        origin = session.state
        candidates: tuple[Transition, ...] | None = TRANSITIONS.get((origin, event))

        if candidates is None:
            legal = self.legal_events(origin)
            self._audit.append(
                session.session_id,
                AuditKind.TRANSITION_REFUSED,
                from_state=str(origin),
                event=str(event),
                detail=f"no such edge; legal here: {','.join(sorted(legal)) or 'none'}",
            )
            raise IllegalTransitionError(origin, event, legal)

        for candidate in candidates:
            # guard=None is an unconditional fallback. states.py guarantees it
            # is last in the tuple, so reaching it means every guarded
            # alternative was already tried and rejected.
            if candidate.guard is None or candidate.guard(session):
                # Write the justification, then move. Not the reverse: see the
                # docstring. An append that raises here leaves the session
                # exactly where it was.
                self._audit.append(
                    session.session_id,
                    AuditKind.TRANSITION_ACCEPTED,
                    from_state=str(origin),
                    to_state=str(candidate.to),
                    event=str(event),
                    detail=candidate.why,
                )
                session.state = candidate.to
                return session

        self._audit.append(
            session.session_id,
            AuditKind.TRANSITION_REFUSED,
            from_state=str(origin),
            event=str(event),
            detail="edge exists but no guard matched",
        )
        raise NoGuardMatchedError(origin, event)
