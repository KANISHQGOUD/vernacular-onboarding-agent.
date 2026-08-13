"""Shared fixtures and helpers."""

from __future__ import annotations

import sys
from datetime import date

import pytest

from app.audit import InMemoryAuditSink
from app.flow.eligibility import EligibilityInput, Residency
from app.flow.machine import Machine
from app.session import REQUIRED_APPLICANT_FIELDS, InMemorySessionStore, Session

# The Windows console mangles non-ASCII, and decline reasons will eventually be
# rendered in Devanagari and other Indic scripts during debugging.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# A fixed reference date so the age rule cannot drift as real time passes.
# Every eligibility test pins to this; none of them read a clock.
AS_OF = date(2026, 8, 13)


@pytest.fixture
def audit() -> InMemoryAuditSink:
    return InMemoryAuditSink()


@pytest.fixture
def machine(audit: InMemoryAuditSink) -> Machine:
    return Machine(audit)


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def session() -> Session:
    return Session(session_id="test-session")


def eligible_applicant(**overrides) -> EligibilityInput:
    """A person who passes every rule. Override one field to fail exactly one."""
    defaults = dict(
        date_of_birth=date(1990, 1, 1),
        residency=Residency.RESIDENT,
        has_pan=True,
        has_form_60=False,
        has_existing_active_account=False,
        mobile_linked_to_aadhaar=True,
        as_of=AS_OF,
    )
    defaults.update(overrides)
    return EligibilityInput(**defaults)


def complete_applicant_data() -> dict[str, str]:
    """Every field COLLECT_DATA requires, so the flow can leave that state."""
    return {name: "provided" for name in REQUIRED_APPLICANT_FIELDS}
