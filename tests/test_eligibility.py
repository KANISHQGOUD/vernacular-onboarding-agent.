"""Eligibility: deterministic, versioned, replayable.

Every case pins `as_of` to a fixed date. None of them read a clock, so the age
rule cannot start behaving differently as real time passes -- a test that
quietly changes meaning on someone's birthday is worse than no test.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.flow.eligibility import (
    ELIGIBILITY_RULESET_VERSION,
    RULESET,
    DeclineReason,
    EligibilityDecision,
    EligibilityInput,
    Residency,
    age_on,
    evaluate,
)
from tests.conftest import AS_OF, eligible_applicant


def test_a_qualifying_applicant_passes_every_rule():
    decision = evaluate(eligible_applicant())

    assert decision.passed
    assert decision.decline_reasons == ()
    assert len(decision.results) == len(RULESET)
    assert all(r.passed for r in decision.results)


def test_decision_carries_the_ruleset_version():
    """Provenance. Without this a stored decision cannot be explained later."""
    assert evaluate(eligible_applicant()).ruleset_version == ELIGIBILITY_RULESET_VERSION


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"date_of_birth": date(2010, 1, 1)}, DeclineReason.MINOR_NEEDS_GUARDIAN),
        ({"residency": Residency.NRI}, DeclineReason.NRI_DIFFERENT_PRODUCT),
        ({"residency": Residency.FOREIGN_NATIONAL}, DeclineReason.NRI_DIFFERENT_PRODUCT),
        ({"has_pan": False, "has_form_60": False}, DeclineReason.TAX_ID_MISSING),
        ({"has_existing_active_account": True}, DeclineReason.ALREADY_A_CUSTOMER),
        ({"mobile_linked_to_aadhaar": False}, DeclineReason.MOBILE_NOT_LINKED),
    ],
)
def test_each_rule_declines_for_its_own_reason(overrides, expected_reason):
    decision = evaluate(eligible_applicant(**overrides))

    assert not decision.passed
    assert decision.decline_reasons == (expected_reason,)


def test_form_60_substitutes_for_pan():
    decision = evaluate(eligible_applicant(has_pan=False, has_form_60=True))
    assert decision.passed


def test_all_rules_are_evaluated_not_short_circuited():
    """Three failures must be recorded as three, not as the first one found."""
    decision = evaluate(
        eligible_applicant(
            date_of_birth=date(2015, 6, 1),
            residency=Residency.NRI,
            mobile_linked_to_aadhaar=False,
        )
    )

    assert not decision.passed
    assert len(decision.results) == len(RULESET)
    assert set(decision.decline_reasons) == {
        DeclineReason.MINOR_NEEDS_GUARDIAN,
        DeclineReason.NRI_DIFFERENT_PRODUCT,
        DeclineReason.MOBILE_NOT_LINKED,
    }
    # And the ones that did pass are recorded as passing, not merely omitted.
    passed_ids = {r.rule_id for r in decision.results if r.passed}
    assert passed_ids == {"HAS_PAN_OR_FORM60", "NO_EXISTING_ACCOUNT"}


def test_rule_ids_are_unique():
    ids = [rule.rule_id for rule in RULESET]
    assert len(ids) == len(set(ids))


def test_every_rule_has_a_translatable_description():
    for rule in RULESET:
        assert rule.description.strip()
        assert rule.description.strip().endswith(".")


# ---------------------------------------------------------------------------
# Age, the rule most likely to go impure.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dob,expected",
    [
        (date(2008, 8, 13), 18),   # birthday is today -> eligible
        (date(2008, 8, 14), 17),   # birthday tomorrow -> not yet
        (date(2008, 8, 12), 18),
        (date(1990, 1, 1), 36),
        (date(2009, 2, 28), 17),
    ],
)
def test_age_is_computed_against_the_supplied_date(dob, expected):
    assert age_on(dob, AS_OF) == expected


def test_turning_eighteen_flips_the_decision_on_the_right_day():
    dob = date(2008, 8, 13)
    assert evaluate(eligible_applicant(date_of_birth=dob, as_of=date(2026, 8, 12))).passed is False
    assert evaluate(eligible_applicant(date_of_birth=dob, as_of=date(2026, 8, 13))).passed is True


# ---------------------------------------------------------------------------
# Replay: the property that makes this an audit trail rather than a log.
# ---------------------------------------------------------------------------


def test_the_same_input_always_produces_an_identical_decision():
    applicant = eligible_applicant(residency=Residency.NRI)
    assert evaluate(applicant) == evaluate(applicant)


def test_a_decision_survives_a_storage_round_trip():
    original = evaluate(eligible_applicant(has_pan=False, has_form_60=False))
    restored = EligibilityDecision.from_dict(original.to_dict())

    assert restored == original
    assert restored.decline_reasons == (DeclineReason.TAX_ID_MISSING,)


def test_an_input_survives_a_storage_round_trip():
    """Store the input and you can re-derive the decision years later."""
    original = eligible_applicant(residency=Residency.FOREIGN_NATIONAL)
    restored = EligibilityInput.from_dict(original.to_dict())

    assert restored == original
    assert evaluate(restored) == evaluate(original)
