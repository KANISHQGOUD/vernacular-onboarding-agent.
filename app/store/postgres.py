"""Postgres implementations of the three storage seams.

`SessionStore`, `TokenStore` and `AuditSink` were protocols from the first
commit precisely so this file could exist without touching the state machine.
Nothing in `app/flow/` imports anything here.

WHAT CHANGES WHEN YOU SWITCH
----------------------------
Restarting the process stops losing everybody's application. That is the whole
point: the in-memory registry made a deploy indistinguishable from an outage
for anyone mid-form, and worse, indistinguishable from an expired session for
anyone parked at the KYC handoff -- which is a state designed to last hours.

WHAT DOES NOT CHANGE
--------------------
The audit trail's append-only guarantee gets stronger, not weaker. In Python it
rested on the sink having no update method, which stops accidents and stops
nothing else. In Postgres a trigger rejects UPDATE and DELETE for every role
including the owner, so the claim survives contact with a psql prompt.

CONCURRENCY, HONESTLY
---------------------
Audit sequences are per-session and gapless, which is a claim two writers can
break. The next number is computed and inserted in one statement, and the
composite primary key means a race ends in a unique violation rather than a
silent overwrite. The loser retries. That is why `append` has a retry loop and
the in-memory version does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Jsonb

from app.audit import AuditEvent, AuditKind
from app.session import Session

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# How many times to lose the race for a sequence number before giving up.
# Contention is per session, and a session has one conversation attached to it,
# so real contention is rare and short.
SEQUENCE_ATTEMPTS = 5


def apply_schema(conninfo: str) -> None:
    """Create the tables, indexes and triggers. Safe to run repeatedly."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(conninfo, autocommit=True) as connection:
        connection.execute(sql)
    log.info("schema applied")


@dataclass
class PostgresSessionStore:
    """Satisfies `SessionStore`. One JSONB column, not a column per field.

    `Session.to_dict()` is already the storage format and already round-trip
    tested. A column per field would be a second schema to keep in step with
    the dataclass, and the two would drift.
    """

    conninfo: str

    def save(self, session: Session) -> None:
        data = session.to_dict()
        with psycopg.connect(self.conninfo, autocommit=True) as connection:
            connection.execute(
                """
                INSERT INTO sessions (session_id, state, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET state = EXCLUDED.state,
                        data  = EXCLUDED.data
                """,
                (session.session_id, str(session.state), Jsonb(data)),
            )

    def load(self, session_id: str) -> Session | None:
        with psycopg.connect(self.conninfo) as connection:
            row = connection.execute(
                "SELECT data FROM sessions WHERE session_id = %s",
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return Session.from_dict(row[0])

    # -- Operational queries the product actually needs ---------------------

    def suspended_at_kyc(self) -> list[str]:
        """Sessions parked at the regulatory handoff: people awaiting a call.

        The one query worth an index. Everything else about a session is
        answered by loading it.
        """
        with psycopg.connect(self.conninfo) as connection:
            rows = connection.execute(
                "SELECT session_id FROM sessions WHERE state = %s ORDER BY updated_at",
                ("KYC_PENDING",),
            ).fetchall()
        return [row[0] for row in rows]


@dataclass
class PostgresTokenStore:
    """Satisfies `TokenStore`. Stores the hash the registry hands it.

    The registry hashes before calling this, so a plaintext token never
    reaches the database layer at all -- not even in a query parameter that
    might end up in a slow-query log.
    """

    conninfo: str

    def put(self, session_id: str, token_sha256: str) -> None:
        with psycopg.connect(self.conninfo, autocommit=True) as connection:
            connection.execute(
                """
                INSERT INTO session_tokens (session_id, token_sha256)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE
                    SET token_sha256 = EXCLUDED.token_sha256
                """,
                (session_id, token_sha256),
            )

    def get(self, session_id: str) -> str | None:
        with psycopg.connect(self.conninfo) as connection:
            row = connection.execute(
                "SELECT token_sha256 FROM session_tokens WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        return row[0] if row else None

    def forget(self, session_id: str) -> None:
        with psycopg.connect(self.conninfo, autocommit=True) as connection:
            connection.execute(
                "DELETE FROM session_tokens WHERE session_id = %s", (session_id,)
            )


@dataclass
class PostgresAuditSink:
    """Satisfies `AuditSink`. Append-only, enforced by the database."""

    conninfo: str

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
        recorded_at = datetime.now(timezone.utc)

        for attempt in range(SEQUENCE_ATTEMPTS):
            try:
                with psycopg.connect(self.conninfo, autocommit=True) as connection:
                    row = connection.execute(
                        """
                        INSERT INTO audit_events (
                            session_id, sequence, kind, recorded_at,
                            from_state, to_state, event, tool, detail
                        )
                        SELECT
                            %(session_id)s,
                            COALESCE(MAX(sequence) + 1, 0),
                            %(kind)s, %(recorded_at)s,
                            %(from_state)s, %(to_state)s,
                            %(event)s, %(tool)s, %(detail)s
                        FROM audit_events
                        WHERE session_id = %(session_id)s
                        RETURNING sequence
                        """,
                        {
                            "session_id": session_id,
                            "kind": str(kind),
                            "recorded_at": recorded_at,
                            "from_state": from_state,
                            "to_state": to_state,
                            "event": event,
                            "tool": tool,
                            "detail": detail,
                        },
                    ).fetchone()

                return AuditEvent(
                    session_id=session_id,
                    sequence=row[0],
                    kind=kind,
                    recorded_at=recorded_at,
                    from_state=from_state,
                    to_state=to_state,
                    event=event,
                    tool=tool,
                    detail=detail,
                )

            except pg_errors.UniqueViolation:
                # Another writer took this sequence number. The primary key
                # turned a silent overwrite into a loud retry, which is the
                # entire reason it is a composite key.
                log.debug("audit sequence race on %s, attempt %s", session_id, attempt + 1)
                continue

        raise RuntimeError(
            f"could not allocate an audit sequence for {session_id} "
            f"after {SEQUENCE_ATTEMPTS} attempts"
        )

    def for_session(self, session_id: str) -> tuple[AuditEvent, ...]:
        with psycopg.connect(self.conninfo) as connection:
            rows = connection.execute(
                """
                SELECT sequence, kind, recorded_at, from_state, to_state,
                       event, tool, detail
                FROM audit_events
                WHERE session_id = %s
                ORDER BY sequence
                """,
                (session_id,),
            ).fetchall()

        return tuple(
            AuditEvent(
                session_id=session_id,
                sequence=row[0],
                kind=AuditKind(row[1]),
                recorded_at=row[2],
                from_state=row[3],
                to_state=row[4],
                event=row[5],
                tool=row[6],
                detail=row[7] or "",
            )
            for row in rows
        )


def describe(conninfo: str) -> dict[str, Any]:
    """A cheap health check: is the schema there, and how much is in it."""
    with psycopg.connect(conninfo) as connection:
        counts = {}
        for table in ("sessions", "session_tokens", "audit_events"):
            counts[table] = connection.execute(
                f"SELECT count(*) FROM {table}"  # noqa: S608 - fixed literals
            ).fetchone()[0]

        trigger = connection.execute(
            """
            SELECT count(*) FROM pg_trigger
            WHERE tgname = 'audit_events_no_update' AND NOT tgisinternal
            """
        ).fetchone()[0]

    return {"counts": counts, "append_only_trigger": bool(trigger)}
