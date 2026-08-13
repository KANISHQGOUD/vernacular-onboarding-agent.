"""The durable session -- everything the machine knows about one applicant.

This has to survive a process restart, because of where the flow goes. The
agent hands off to video KYC and the user may finish it hours later, at a
branch, on a different device. If the session lives only in memory, that
handoff is a dead end. So `Session` is designed as a database row from the
start: flat, JSON-serialisable, with no live objects hanging off it.

`to_dict` / `from_dict` are what make the Postgres swap boring later. They are
also what the suspend/resume test drives, so serialisation is exercised on
every run rather than discovered to be broken on the day it matters.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.flow.eligibility import EligibilityDecision, EligibilityInput
from app.flow.states import State

# What COLLECT_DATA has to gather before KYC can start. The machine checks
# this set rather than asking the model whether it thinks it has enough.
REQUIRED_APPLICANT_FIELDS: frozenset[str] = frozenset(
    {"full_name", "date_of_birth", "address", "occupation", "mobile"}
)


@dataclass
class Session:
    """One applicant's journey. Exactly one live state, always."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: State = State.START

    language: str | None = None
    consent_recorded: bool = False
    applicant: dict[str, str] = field(default_factory=dict)

    # Both halves of the screening are kept: the answers that went in, and the
    # verdict that came out. eligibility.py promises a decision can be replayed
    # from its stored input years later -- that promise is only real if the
    # input is actually stored, which is what this field is for.
    eligibility_input: EligibilityInput | None = None
    eligibility: EligibilityDecision | None = None

    kyc_attempts: int = 0
    kyc_reference: str | None = None

    account_number: str | None = None
    deposit_reference: str | None = None
    first_payment_reference: str | None = None

    # -- Properties the transition guards read ------------------------------
    # The guards in states.py are written as `lambda s: s.eligibility_passed`
    # rather than `s.eligibility is not None and s.eligibility.passed`. The
    # table stays readable, and the None-handling lives here where it belongs.

    @property
    def eligibility_passed(self) -> bool:
        return self.eligibility is not None and self.eligibility.passed

    @property
    def required_data_complete(self) -> bool:
        return REQUIRED_APPLICANT_FIELDS.issubset(self.applicant.keys())

    @property
    def missing_fields(self) -> frozenset[str]:
        return REQUIRED_APPLICANT_FIELDS - self.applicant.keys()

    # -- Persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": str(self.state),
            "language": self.language,
            "consent_recorded": self.consent_recorded,
            "applicant": dict(self.applicant),
            "eligibility_input": (
                self.eligibility_input.to_dict() if self.eligibility_input else None
            ),
            "eligibility": self.eligibility.to_dict() if self.eligibility else None,
            "kyc_attempts": self.kyc_attempts,
            "kyc_reference": self.kyc_reference,
            "account_number": self.account_number,
            "deposit_reference": self.deposit_reference,
            "first_payment_reference": self.first_payment_reference,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Session:
        return cls(
            session_id=raw["session_id"],
            state=State(raw["state"]),
            language=raw["language"],
            consent_recorded=raw["consent_recorded"],
            applicant=dict(raw["applicant"]),
            eligibility_input=(
                EligibilityInput.from_dict(raw["eligibility_input"])
                if raw["eligibility_input"]
                else None
            ),
            eligibility=(
                EligibilityDecision.from_dict(raw["eligibility"]) if raw["eligibility"] else None
            ),
            kyc_attempts=raw["kyc_attempts"],
            kyc_reference=raw["kyc_reference"],
            account_number=raw["account_number"],
            deposit_reference=raw["deposit_reference"],
            first_payment_reference=raw["first_payment_reference"],
        )


class SessionStore(Protocol):
    """The seam Postgres drops into."""

    def save(self, session: Session) -> None: ...

    def load(self, session_id: str) -> Session | None: ...


@dataclass
class InMemorySessionStore:
    """Phase 1 implementation.

    It stores the SERIALISED form and rebuilds on load, rather than keeping a
    reference to the object it was handed. That costs nothing and buys a lot:
    a store that handed the same object back would let a test 'resume' a
    session that was never actually written down, and serialisation bugs would
    stay hidden until the day Postgres arrives.
    """

    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, session: Session) -> None:
        self._rows[session.session_id] = session.to_dict()

    def load(self, session_id: str) -> Session | None:
        raw = self._rows.get(session_id)
        return Session.from_dict(raw) if raw else None
