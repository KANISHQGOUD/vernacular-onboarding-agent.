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

    # REGULATED PERSONAL DATA. Name, date of birth, address, occupation,
    # mobile. Under the DPDP Act this carries retention limits, correction
    # rights, and erasure obligations.
    applicant: dict[str, str] = field(default_factory=dict)

    # OPERATIONAL METADATA. Why a KYC attempt failed, why someone escalated.
    # Deliberately a separate dict rather than underscore-prefixed keys inside
    # `applicant`: an erasure request has to be able to delete the personal
    # data while the operational trail survives, and that is impossible when
    # both live in one dict. Never put personal data in here.
    notes: dict[str, str] = field(default_factory=dict)

    # What the recogniser actually returned for each field, before any
    # normalising. `applicant` holds what was recorded; this holds what was
    # heard, and when somebody disputes a field afterwards that difference is
    # the whole question.
    #
    # PERSONAL DATA, so it is erased with the rest and never reaches the audit
    # trail -- every `detail` there is structural, and that is what lets an
    # append-only log coexist with a deletion request.
    heard: dict[str, str] = field(default_factory=dict)

    # A value read back and awaiting a yes or no. Held here rather than
    # written to `applicant`, because a field that has not been confirmed is
    # not yet an answer -- and writing it early would mean a "no" had to
    # un-write it, which is a worse shape than never having written it.
    #
    # `pending_value` is personal data like the rest; `pending_field` is just
    # a field name, but it goes with it.
    pending_field: str | None = None
    pending_value: str | None = None

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

    def corrections(self, field_name: str) -> int:
        """How many times a read-back of this field has been denied.

        Operational, like `field_attempts` -- it describes how badly the
        conversation went, not who the person is -- so it survives erasure.
        """
        try:
            return int(self.notes.get(f"corrections:{field_name}", "0"))
        except ValueError:
            return 0

    def field_attempts(self, field_name: str) -> int:
        """How many times this field has been rejected already.

        In `notes` rather than a field of its own, for the same reason
        `lapses` is: it describes the conversation, not the person, so it
        survives `erase_personal_data()` and stays answerable afterwards.
        """
        try:
            return int(self.notes.get(f"attempts:{field_name}", "0"))
        except ValueError:
            return 0

    @property
    def lapses(self) -> int:
        """How many times this session went quiet and came back.

        Lives in `notes`, not as a field of its own, because it is operational
        metadata about the conversation rather than anything about the person
        -- so it survives `erase_personal_data()`, which is the whole reason
        that split exists.
        """
        try:
            return int(self.notes.get("lapses", "0"))
        except ValueError:
            return 0

    @property
    def has_lapsed(self) -> bool:
        return self.lapses > 0

    @property
    def required_data_complete(self) -> bool:
        return REQUIRED_APPLICANT_FIELDS.issubset(self.applicant.keys())

    @property
    def missing_fields(self) -> frozenset[str]:
        return REQUIRED_APPLICANT_FIELDS - self.applicant.keys()

    # -- Erasure ------------------------------------------------------------

    # What went in, and what went out. Both kept beside their references
    # and treated the same way.
    deposit_paise: int | None = None
    # What the first payment actually was. Kept beside the reference and
    # treated the same way -- transaction metadata rather than the person --
    # so it survives erasure exactly as `first_payment_reference` does.
    # Needed because the amount is now the caller's, so "that payment of X
    # went through" cannot be read off a constant any more.
    first_payment_paise: int | None = None

    def erase_personal_data(self) -> None:
        """Honour a deletion request without destroying the audit trail.

        This is the method the `notes` split exists for. It clears the
        regulated personal data and the screening inputs derived from it,
        while leaving the state, the decision, the operational notes, and the
        audit log intact -- so the bank can still answer "was this handled
        correctly" after it can no longer answer "who was it".

        `eligibility` (the verdict) is kept on purpose; `eligibility_input`
        (the person's date of birth and residency) is not.
        """
        self.applicant.clear()
        self.heard.clear()
        self.pending_field = None
        self.pending_value = None
        self.eligibility_input = None

    # -- Persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": str(self.state),
            "language": self.language,
            "consent_recorded": self.consent_recorded,
            "applicant": dict(self.applicant),
            "heard": dict(self.heard),
            "pending_field": self.pending_field,
            "pending_value": self.pending_value,
            "notes": dict(self.notes),
            "eligibility_input": (
                self.eligibility_input.to_dict() if self.eligibility_input else None
            ),
            "eligibility": self.eligibility.to_dict() if self.eligibility else None,
            "deposit_paise": self.deposit_paise,
            "first_payment_paise": self.first_payment_paise,
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
            heard=dict(raw.get("heard", {})),
            pending_field=raw.get("pending_field"),
            pending_value=raw.get("pending_value"),
            notes=dict(raw.get("notes", {})),
            eligibility_input=(
                EligibilityInput.from_dict(raw["eligibility_input"])
                if raw["eligibility_input"]
                else None
            ),
            deposit_paise=raw.get("deposit_paise"),
            first_payment_paise=raw.get("first_payment_paise"),
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
