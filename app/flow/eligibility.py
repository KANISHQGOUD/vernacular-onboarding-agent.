"""Deterministic, versioned eligibility screening.

Three properties this file exists to guarantee:

VERSIONED   Every decision is stamped with the ruleset that produced it. When
            compliance amends a rule you add a new version rather than editing
            the old one, so a decision made last March can still be explained
            by the rules that were actually in force last March.

COMPLETE    Every rule is evaluated. No short-circuit on first failure. If
            someone fails on three counts, the record says three, because
            "why was this person declined" should be answerable from storage
            without ever re-running anything.

PURE        A rule is a function of its input snapshot and nothing else. No
            clock, no network, no database. Age is the trap here: reading
            today() inside a rule would mean the same applicant silently
            becomes eligible on their birthday and the golden tests rot. So
            `as_of` is part of the input, passed in by the caller.

Together those make a decision REPLAYABLE: feed the stored input back through
the stored ruleset version and you must get a bit-identical result, years
later. That is the difference between an audit trail and a log file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Callable

# Bump this when a rule is added, removed, or its meaning changes. Never edit
# a published ruleset in place -- decisions already stamped with it must stay
# reproducible.
ELIGIBILITY_RULESET_VERSION = "2026-08-13.v1"

MINIMUM_AGE_YEARS = 18


class Residency(StrEnum):
    RESIDENT = "RESIDENT"
    NRI = "NRI"
    FOREIGN_NATIONAL = "FOREIGN_NATIONAL"


class DeclineReason(StrEnum):
    """What the user gets told, derived from a rule id.

    The spoken sentence is a translation keyed off one of these constants. The
    language model phrases it; it never decides it. A model that could compose
    its own decline reason could invent a policy the bank does not have.
    """

    MINOR_NEEDS_GUARDIAN = "MINOR_NEEDS_GUARDIAN"
    NRI_DIFFERENT_PRODUCT = "NRI_DIFFERENT_PRODUCT"
    TAX_ID_MISSING = "TAX_ID_MISSING"
    ALREADY_A_CUSTOMER = "ALREADY_A_CUSTOMER"
    MOBILE_NOT_LINKED = "MOBILE_NOT_LINKED"


@dataclass(frozen=True, slots=True)
class EligibilityInput:
    """The frozen snapshot a decision is computed from.

    Everything any rule reads must appear here. That is what makes replay
    possible: store this, store the version, and the decision is recoverable.
    """

    date_of_birth: date
    residency: Residency
    has_pan: bool
    has_form_60: bool
    has_existing_active_account: bool
    mobile_linked_to_aadhaar: bool
    as_of: date

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_of_birth": self.date_of_birth.isoformat(),
            "residency": str(self.residency),
            "has_pan": self.has_pan,
            "has_form_60": self.has_form_60,
            "has_existing_active_account": self.has_existing_active_account,
            "mobile_linked_to_aadhaar": self.mobile_linked_to_aadhaar,
            "as_of": self.as_of.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EligibilityInput:
        return cls(
            date_of_birth=date.fromisoformat(raw["date_of_birth"]),
            residency=Residency(raw["residency"]),
            has_pan=raw["has_pan"],
            has_form_60=raw["has_form_60"],
            has_existing_active_account=raw["has_existing_active_account"],
            mobile_linked_to_aadhaar=raw["mobile_linked_to_aadhaar"],
            as_of=date.fromisoformat(raw["as_of"]),
        )


@dataclass(frozen=True, slots=True)
class Rule:
    """A rule is data too, for the same reason transitions are.

    You can print the ruleset, diff it across versions, and hand the
    descriptions to a translator -- none of which you can do with a function
    full of if-statements.
    """

    rule_id: str
    description: str
    decline_reason: DeclineReason
    predicate: Callable[[EligibilityInput], bool]


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule_id: str
    passed: bool
    decline_reason: DeclineReason | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "passed": self.passed,
            "decline_reason": str(self.decline_reason) if self.decline_reason else None,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuleResult:
        reason = raw["decline_reason"]
        return cls(
            rule_id=raw["rule_id"],
            passed=raw["passed"],
            decline_reason=DeclineReason(reason) if reason else None,
        )


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """The outcome, carrying its own provenance.

    Note there is no timestamp field. A timestamp would make two evaluations of
    identical input compare unequal, which is precisely the property replay
    depends on. When the decision happened belongs in the audit log; what the
    decision *was* belongs here.
    """

    ruleset_version: str
    passed: bool
    results: tuple[RuleResult, ...]

    @property
    def decline_reasons(self) -> tuple[DeclineReason, ...]:
        """Every reason, in ruleset order -- not just the first."""
        return tuple(r.decline_reason for r in self.results if not r.passed and r.decline_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ruleset_version": self.ruleset_version,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EligibilityDecision:
        return cls(
            ruleset_version=raw["ruleset_version"],
            passed=raw["passed"],
            results=tuple(RuleResult.from_dict(r) for r in raw["results"]),
        )


def age_on(date_of_birth: date, as_of: date) -> int:
    """Completed years. Takes the reference date rather than reading a clock."""
    had_birthday = (as_of.month, as_of.day) >= (date_of_birth.month, date_of_birth.day)
    return as_of.year - date_of_birth.year - (0 if had_birthday else 1)


# ---------------------------------------------------------------------------
# THE RULESET -- version 2026-08-13.v1
# ---------------------------------------------------------------------------
# Descriptions are written as plain source-language English, phrased so they
# can be translated for the user. Each maps to exactly one decline reason.
# ---------------------------------------------------------------------------

RULESET: tuple[Rule, ...] = (
    Rule(
        rule_id="AGE_MIN_18",
        description="Applicant must be at least 18 years old.",
        decline_reason=DeclineReason.MINOR_NEEDS_GUARDIAN,
        predicate=lambda i: age_on(i.date_of_birth, i.as_of) >= MINIMUM_AGE_YEARS,
    ),
    Rule(
        rule_id="RESIDENT_INDIAN",
        description="Applicant must be a resident Indian. NRI accounts are a different product.",
        decline_reason=DeclineReason.NRI_DIFFERENT_PRODUCT,
        predicate=lambda i: i.residency is Residency.RESIDENT,
    ),
    Rule(
        rule_id="HAS_PAN_OR_FORM60",
        description="Applicant must have a PAN, or have declared Form 60 in its place.",
        decline_reason=DeclineReason.TAX_ID_MISSING,
        predicate=lambda i: i.has_pan or i.has_form_60,
    ),
    Rule(
        rule_id="NO_EXISTING_ACCOUNT",
        description="Applicant must not already hold an active account. "
                    "Existing customers are served by a different journey.",
        decline_reason=DeclineReason.ALREADY_A_CUSTOMER,
        predicate=lambda i: not i.has_existing_active_account,
    ),
    Rule(
        rule_id="MOBILE_LINKED_TO_AADHAAR",
        description="Applicant's mobile number must be linked to their Aadhaar for OTP verification.",
        decline_reason=DeclineReason.MOBILE_NOT_LINKED,
        predicate=lambda i: i.mobile_linked_to_aadhaar,
    ),
)


def evaluate(applicant: EligibilityInput) -> EligibilityDecision:
    """Run every rule. No short-circuit, by design.

    Stopping at the first failure would be faster and would throw away the
    reason the other four failed -- which is the information you need when
    someone asks why an applicant was turned away.
    """
    results = []
    for rule in RULESET:
        # Called exactly once. Evaluating a predicate twice would still be
        # correct given the purity contract above, but relying on that contract
        # to paper over a double call is how the contract quietly stops holding.
        passed = rule.predicate(applicant)
        results.append(
            RuleResult(
                rule_id=rule.rule_id,
                passed=passed,
                decline_reason=None if passed else rule.decline_reason,
            )
        )
    results = tuple(results)
    return EligibilityDecision(
        ruleset_version=ELIGIBILITY_RULESET_VERSION,
        passed=all(r.passed for r in results),
        results=results,
    )
