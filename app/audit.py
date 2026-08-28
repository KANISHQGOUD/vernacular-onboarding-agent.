"""Append-only audit trail.

Two things make this an audit log rather than logging:

1. It records REFUSALS, not just successes. "The agent tried to open an account
   from COLLECT_DATA and was stopped" is the sentence a regulator wants to
   read, and it only exists if you write it down. A log of things that worked
   proves nothing about the things that didn't.

2. Nothing here can update or delete. The sink exposes `append` and reads --
   there is no `update`, no `delete`, no way to reach the underlying list. When
   this moves to Postgres the table gets no UPDATE/DELETE grant and the same
   property holds at the database level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol


class AuditKind(StrEnum):
    """What happened. Refusals are first-class, not an error log."""

    TRANSITION_ACCEPTED = "TRANSITION_ACCEPTED"
    TRANSITION_REFUSED = "TRANSITION_REFUSED"
    TOOL_ALLOWED = "TOOL_ALLOWED"
    TOOL_REFUSED = "TOOL_REFUSED"
    ELIGIBILITY_DECIDED = "ELIGIBILITY_DECIDED"
    # Somebody stopped answering, and later started again. Neither moves the
    # session, so neither is a TRANSITION_ACCEPTED -- but "this application
    # was interrupted four times" is exactly the kind of thing the trail
    # exists to be able to answer afterwards.
    SESSION_LAPSED = "SESSION_LAPSED"
    SESSION_RETURNED = "SESSION_RETURNED"
    # They started talking before the agent had finished. Always permitted --
    # a person must be able to interrupt a machine -- but at the consent gate
    # it means the notice may not have been heard in full, and DPDP requires
    # notice as well as consent. Recorded so the question is answerable later.
    CALLER_INTERRUPTED = "CALLER_INTERRUPTED"
    # A turn lost to us, not to them. Counted separately from a refusal
    # because "how often did our vendor fail a caller" is an operational
    # question, and folding it into "not understood" answers it as zero.
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One immutable line in the trail.

    `sequence` is per-session and gapless. A missing number means a record was
    dropped or removed, which is exactly the thing an append-only log is
    supposed to make detectable.
    """

    session_id: str
    sequence: int
    kind: AuditKind
    recorded_at: datetime
    from_state: str | None = None
    to_state: str | None = None
    event: str | None = None
    tool: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "kind": str(self.kind),
            "recorded_at": self.recorded_at.isoformat(),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event": self.event,
            "tool": self.tool,
            "detail": self.detail,
        }


class AuditSink(Protocol):
    """The seam Postgres drops into. Deliberately has no mutation methods."""

    def append(
        self,
        session_id: str,
        kind: AuditKind,
        *,
        from_state: str | None = None,
        to_state: str | None = None,
        event: str | None = None,
        tool: str | None = None,
        detail: str = "",
    ) -> AuditEvent: ...

    def for_session(self, session_id: str) -> tuple[AuditEvent, ...]: ...


@dataclass
class InMemoryAuditSink:
    """Phase 1 implementation. Same interface the Postgres one will satisfy."""

    _events: list[AuditEvent] = field(default_factory=list)
    _next_sequence: dict[str, int] = field(default_factory=dict)

    def append(
        self,
        session_id: str,
        kind: AuditKind,
        *,
        from_state: str | None = None,
        to_state: str | None = None,
        event: str | None = None,
        tool: str | None = None,
        detail: str = "",
    ) -> AuditEvent:
        sequence = self._next_sequence.get(session_id, 0)
        self._next_sequence[session_id] = sequence + 1
        record = AuditEvent(
            session_id=session_id,
            sequence=sequence,
            kind=kind,
            recorded_at=datetime.now(timezone.utc),
            from_state=from_state,
            to_state=to_state,
            event=event,
            tool=tool,
            detail=detail,
        )
        self._events.append(record)
        return record

    def for_session(self, session_id: str) -> tuple[AuditEvent, ...]:
        """A tuple, not the list. Callers cannot append or reorder through this."""
        return tuple(e for e in self._events if e.session_id == session_id)
