"""Who may start a session, and how many, once strangers can reach the server.

Every session spends from two budgets nobody can top up for free: Gemini's
free tier (20 requests a day, about three conversations) and a one-time Sarvam
credit. On a laptop the only caller is the developer. On a public URL,
`POST /api/session` with no gate is an open tap on both, and one bot drains
them before the person the link was sent to has opened it.

TWO CHECKS, IN THIS ORDER
-------------------------
1. An access code, shared out of band. Wrong or missing is refused before
   anything is counted, so somebody guessing codes cannot use up the day's
   allowance for the people who were actually given one.
2. A daily cap on sessions, counted per UTC day. Turned on deliberately, like
   SARVAM_TTS, because the right number depends on which Gemini tier the key
   is on -- a decision about money this module should not make for anyone.

The code is compared in constant time, for the same reason the session token
is: a plain == leaks how much of it was right.

The count lives in this process. That is correct for one worker, which is how
the server is deployed (see app/web/__main__.py); several workers would each
keep their own count and the real cap would be N times the configured one.
Moving it into Postgres is the fix if that ever changes.

The day is read per request, never at startup. A cap that read the date once
would stop resetting the first night -- the same bug `Conversation.as_of` once
had, where a server up for a week aged every applicant against the day it
booted.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Callable

# Long enough that guessing it is not a practical way in. It is shared in a
# link rather than typed from memory, so length costs nobody anything.
MIN_ACCESS_CODE_LENGTH = 12


def utc_today() -> date:
    return datetime.now(UTC).date()


class Refused(Exception):
    """Why a session was not created, with the HTTP status that says so."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class Gate:
    """Admits or refuses session creation. Empty code and no cap admit all."""

    access_code: str = ""
    daily_cap: int | None = None
    today: Callable[[], date] = utc_today
    _day: date | None = field(default=None, repr=False)
    _admitted: int = field(default=0, repr=False)

    @classmethod
    def from_env(cls) -> Gate:
        raw_cap = os.getenv("MAX_SESSIONS_PER_DAY", "").strip()
        return cls(
            access_code=os.getenv("ACCESS_CODE", "").strip(),
            daily_cap=int(raw_cap) if raw_cap else None,
        )

    def admit(self, presented: str) -> None:
        """Count one session, or raise Refused. Never counts a refusal."""
        if self.access_code and not secrets.compare_digest(
            presented.encode("utf-8"), self.access_code.encode("utf-8")
        ):
            raise Refused(401, "access code required")

        if self.daily_cap is not None:
            now = self.today()
            if now != self._day:
                self._day, self._admitted = now, 0
            if self._admitted >= self.daily_cap:
                raise Refused(429, "the daily session limit has been reached")
            self._admitted += 1


def public_deployment_problems(env: dict[str, str] | None = None) -> list[str]:
    """Everything that must be true before a public URL may serve this.

    Returns every problem rather than the first, so one failed start lists the
    whole fix -- the same reason eligibility evaluates without short-circuit.
    Reachability of the database is checked separately, in build_storage,
    because it needs a connection rather than an environment variable.
    """
    env = dict(os.environ) if env is None else env

    def value(name: str) -> str:
        return env.get(name, "").strip()

    problems: list[str] = []

    code = value("ACCESS_CODE")
    if not code:
        problems.append("ACCESS_CODE is empty -- anyone could start sessions")
    elif len(code) < MIN_ACCESS_CODE_LENGTH:
        problems.append(
            f"ACCESS_CODE is shorter than {MIN_ACCESS_CODE_LENGTH} characters"
        )

    cap = value("MAX_SESSIONS_PER_DAY")
    if not cap:
        problems.append(
            "MAX_SESSIONS_PER_DAY is empty -- choose it from your Gemini tier "
            "(the free tier's 20 requests/day is about 3 conversations)"
        )
    elif not cap.isdigit() or int(cap) < 1:
        problems.append("MAX_SESSIONS_PER_DAY must be a whole number of at least 1")

    if not value("DATABASE_URL"):
        problems.append(
            "DATABASE_URL is empty -- sessions and the audit trail would live "
            "in memory and vanish on every restart"
        )

    if not value("GOOGLE_API_KEY"):
        problems.append("GOOGLE_API_KEY is empty -- the agent would understand nothing")

    return problems


def is_public_deployment() -> bool:
    return os.getenv("PUBLIC_DEPLOYMENT", "").strip() == "1"
