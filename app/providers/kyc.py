"""Video KYC provider -- the one step this agent legally may not perform.

RBI requires video KYC (VCIP) to simulate real-time face-to-face contact with a
trained officer, and since August 2025 those sessions must additionally detect
AI-generated faces. An AI agent cannot be the verifier. It can only prepare the
applicant, hand off, and pick the thread up afterwards.

Look at what this interface offers:

    start_session()   ask the provider to open a session
    get_status()      ask the provider what it decided

That is the whole thing. There is no `verify()`, no `approve()`, no
`set_status()`, no `submit_decision()`. The boundary is not enforced by a
comment asking us to behave -- it is enforced by there being nothing to call.
You cannot write the bypass, because the method does not exist.

This is worth contrasting with how the same constraint appears elsewhere in the
codebase. The state machine says "you may only reach ACCOUNT_CREATE via
KYC_VERIFIED". The intent classifier says "a model may not emit KYC_VERIFIED".
And this file says "nothing in the system can produce a verification at all,
except the provider". Three independent walls around one requirement, because
it is the one that would end the project if it failed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class KycStatus(StrEnum):
    """What the provider says. Only the provider may say it."""

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class KycFailureReason(StrEnum):
    """Why a session did not verify.

    Note how ordinary most of these are. A real VCIP session fails far more
    often for bandwidth than for fraud, which is exactly why the flow budgets
    retries instead of treating one failure as a rejection.
    """

    POOR_VIDEO_QUALITY = "POOR_VIDEO_QUALITY"
    NETWORK_DROPPED = "NETWORK_DROPPED"
    OFFICER_UNAVAILABLE = "OFFICER_UNAVAILABLE"
    DOCUMENT_UNREADABLE = "DOCUMENT_UNREADABLE"
    FACE_MISMATCH = "FACE_MISMATCH"
    SYNTHETIC_FACE_SUSPECTED = "SYNTHETIC_FACE_SUSPECTED"
    LINK_EXPIRED = "LINK_EXPIRED"


@dataclass(frozen=True, slots=True)
class KycSession:
    """A handle on something happening outside our process.

    `join_url` is where the applicant goes. We never see the video, never see a
    face, and never receive a frame -- we hold a reference and ask about it.
    """

    reference: str
    status: KycStatus
    join_url: str
    failure_reason: KycFailureReason | None = None

    @property
    def is_settled(self) -> bool:
        """True once the provider has reached a verdict, either way."""
        return self.status is not KycStatus.PENDING


@runtime_checkable
class KycProvider(Protocol):
    """The seam a real VCIP vendor drops into.

    Deliberately two methods. Adding a third that accepts a verdict from our
    side would hand this system a capability the regulation says it must not
    have, so the narrowness is the point rather than an unfinished sketch.
    """

    def start_session(self, applicant_reference: str, language: str) -> KycSession: ...

    def get_status(self, reference: str) -> KycSession: ...


class UnknownKycReferenceError(LookupError):
    """Asked about a session this provider never issued."""


@dataclass
class MockKycProvider:
    """Stand-in for a vendor we cannot buy as an individual.

    Outcomes are scripted rather than random. A mock that failed one time in
    five would make the retry test flaky, and a flaky test on the regulatory
    path is worse than no test -- people learn to re-run it until it passes.

    Each entry in `outcomes` is consumed by one `get_status` call, so a script
    of [FAILED, VERIFIED] reproduces the real-world common case: the first
    attempt drops, the second works.
    """

    outcomes: list[tuple[KycStatus, KycFailureReason | None]] = field(default_factory=list)
    default_outcome: tuple[KycStatus, KycFailureReason | None] = (KycStatus.VERIFIED, None)

    _issued: dict[str, KycSession] = field(default_factory=dict)
    _outcome_index: int = 0

    def start_session(self, applicant_reference: str, language: str) -> KycSession:
        # Deterministic reference, so tests can assert on it and a replayed run
        # produces the same audit trail.
        digest = hashlib.sha256(
            f"{applicant_reference}:{len(self._issued)}".encode()
        ).hexdigest()[:12]
        session = KycSession(
            reference=f"vckyc-{digest}",
            status=KycStatus.PENDING,
            join_url=f"https://vckyc.example/{digest}?lang={language}",
        )
        self._issued[session.reference] = session
        return session

    def get_status(self, reference: str) -> KycSession:
        if reference not in self._issued:
            raise UnknownKycReferenceError(reference)

        status, reason = self._next_outcome()
        settled = KycSession(
            reference=reference,
            status=status,
            join_url=self._issued[reference].join_url,
            failure_reason=reason,
        )
        self._issued[reference] = settled
        return settled

    def _next_outcome(self) -> tuple[KycStatus, KycFailureReason | None]:
        if self._outcome_index < len(self.outcomes):
            outcome = self.outcomes[self._outcome_index]
            self._outcome_index += 1
            return outcome
        return self.default_outcome
