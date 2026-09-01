"""Field rules, and the escape that stops them trapping anybody.

Pure and offline, like `test_eligibility.py`. Every rule below is checked
against the specific recognition failure it exists for, rather than against
made-up strings -- a rule whose test does not name the bug it prevents tends
to be deleted by the next person who finds it annoying.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.audit import AuditKind, InMemoryAuditSink
from app.flow.actions import Actions, FieldRejected
from app.flow.fields import (
    MAX_FIELD_ATTEMPTS,
    RULES,
    as_date,
    digits_only,
    normalised,
    validate,
)
from app.flow.machine import Machine
from app.flow.states import Event, State
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import MockKycProvider
from app.providers.payments import MockPaymentRail
from app.session import REQUIRED_APPLICANT_FIELDS, Session
from tests.conftest import AS_OF

# ---------------------------------------------------------------------------
# Normalisation the rules depend on.
# ---------------------------------------------------------------------------


def test_a_number_read_aloud_in_devanagari_is_the_same_number():
    """Sarvam returns numbers in either script, and a naive isdigit() check
    would reject the Devanagari one -- rejecting a correct answer because of
    how the vendor rendered it."""
    assert digits_only("९८१२३४५६७८") == "9812345678"
    assert digits_only("98123 45678") == "9812345678"
    assert digits_only("+91 98123-45678") == "919812345678"


def test_a_date_arrives_iso_because_the_extractor_makes_it_so():
    assert as_date("1990-01-01") == date(1990, 1, 1)
    assert as_date("एक जनवरी") is None
    assert as_date("") is None


def test_only_the_mobile_is_rewritten_before_storing():
    """It is looked up in core banking and receives an OTP, so its stored
    shape matters. A name must be stored as given."""
    assert normalised("mobile", "९८१२३ ४५६७८") == "9812345678"
    assert normalised("full_name", "  Asha Devi  ") == "Asha Devi"


# ---------------------------------------------------------------------------
# The rules, each against the failure it exists for.
# ---------------------------------------------------------------------------


def test_a_dropped_digit_is_caught():
    """The failure nothing else can catch: nine digits is a plausible
    sentence, a valid string, and an unreachable phone."""
    assert validate("mobile", "981234567", as_of=AS_OF) is not None
    assert validate("mobile", "98123456789", as_of=AS_OF) is not None
    assert validate("mobile", "9812345678", as_of=AS_OF) is None


def test_a_landline_shaped_number_is_not_a_mobile():
    assert validate("mobile", "2212345678", as_of=AS_OF) is not None


def test_nineteen_ninety_misheard_as_twenty_ninety():
    """A birth year in the future. Until now nothing in the system parsed
    this field at all, so it would have been stored and screened against."""
    assert validate("date_of_birth", "2090-01-01", as_of=AS_OF) is not None
    assert validate("date_of_birth", "1990-01-01", as_of=AS_OF) is None


def test_a_child_and_a_centenarian_are_both_refused():
    assert validate("date_of_birth", "2015-06-30", as_of=AS_OF) is not None
    assert validate("date_of_birth", "1850-01-01", as_of=AS_OF) is not None


def test_a_date_that_never_parsed_is_refused():
    """If extraction did not happen, the raw words arrive. Storing them means
    an unparseable string sits in a date field forever."""
    assert validate("date_of_birth", "पहली जनवरी", as_of=AS_OF) is not None


def test_a_yes_landing_in_the_address_field_is_caught():
    """Happens when a turn is attributed to the wrong field."""
    assert validate("address", "haan", as_of=AS_OF) is not None
    assert validate("address", "14 Gandhi Marg, Jaipur", as_of=AS_OF) is None


def test_silence_transcribed_as_punctuation_is_not_a_name():
    assert validate("full_name", "।", as_of=AS_OF) is not None
    assert validate("full_name", "", as_of=AS_OF) is not None
    assert validate("full_name", "आशा देवी", as_of=AS_OF) is None


def test_an_indic_name_is_not_rejected_for_being_indic():
    """The rules must not quietly require Latin script from a product whose
    entire point is that people do not have to use it."""
    for name in ("आशा देवी राठौड़", "அஷா தேவி", "আশা দেবী"):
        assert validate("full_name", name, as_of=AS_OF) is None, name


def test_a_field_with_no_rule_passes():
    """New fields must be addable without every one needing a rule in the
    same commit, and silently rejecting an unruled field is the worst of
    both."""
    assert validate("middle_name", "x", as_of=AS_OF) is None


def test_every_required_field_has_a_rule():
    """Not enforced by the code -- unruled fields pass -- so it is enforced
    here, where forgetting one is visible."""
    ruled = {rule.field for rule in RULES}
    assert REQUIRED_APPLICANT_FIELDS <= ruled


# ---------------------------------------------------------------------------
# The escape. Rules are guesses, and each is wrong about somebody.
# ---------------------------------------------------------------------------


def rig():
    audit = InMemoryAuditSink()
    machine = Machine(audit)
    actions = Actions(
        machine=machine,
        kyc=MockKycProvider(),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )
    session = Session(session_id="fields", state=State.COLLECT_DATA)
    return actions, session, audit


def test_a_bad_value_is_refused_rather_than_stored():
    actions, session, _ = rig()

    with pytest.raises(FieldRejected) as caught:
        actions.save_applicant_field(session, "mobile", "12345", as_of=AS_OF)

    assert "mobile" not in session.applicant
    assert caught.value.attempts == 1
    assert str(caught.value.problem)


def test_the_applicant_gets_the_last_word():
    """Every rule here is a guess about what a real answer looks like, and
    each is wrong about somebody: an unusual name, a number that does not fit
    the Indian mobile shape. A flow whose purpose is not turning people away
    must not invent a new way to turn them away, so after the budget the
    answer is stored regardless -- flagged, not discarded.
    """
    actions, session, audit = rig()

    for _ in range(MAX_FIELD_ATTEMPTS - 1):
        with pytest.raises(FieldRejected):
            actions.save_applicant_field(session, "mobile", "12345", as_of=AS_OF)

    # The last attempt is accepted.
    actions.save_applicant_field(session, "mobile", "12345", as_of=AS_OF)

    assert session.applicant["mobile"] == "12345"
    assert session.notes["unvalidated:mobile"]
    assert AuditKind.FIELD_UNVALIDATED in [r.kind for r in audit.for_session("fields")]


def test_the_attempt_counter_is_operational_not_personal():
    """It describes the conversation rather than the person, so it survives
    erasure -- and "this rule fires constantly" stays answerable after the
    applicant is gone."""
    actions, session, _ = rig()

    with pytest.raises(FieldRejected):
        actions.save_applicant_field(session, "mobile", "12345", as_of=AS_OF)
    session.applicant["full_name"] = "Asha Devi"

    session.erase_personal_data()

    assert session.applicant == {}
    assert session.field_attempts("mobile") == 1


def test_attempts_are_counted_per_field():
    """Struggling with an address must not spend the budget for a name."""
    actions, session, _ = rig()

    with pytest.raises(FieldRejected):
        actions.save_applicant_field(session, "mobile", "1", as_of=AS_OF)

    assert session.field_attempts("mobile") == 1
    assert session.field_attempts("full_name") == 0


def test_a_good_value_never_counts_against_them():
    actions, session, _ = rig()
    actions.save_applicant_field(session, "mobile", "9812345678", as_of=AS_OF)

    assert session.applicant["mobile"] == "9812345678"
    assert session.field_attempts("mobile") == 0
    assert "unvalidated:mobile" not in session.notes


def test_the_last_field_still_completes_the_set():
    """The completeness check runs after a successful save, and the escape
    path must not skip it -- otherwise a session that hit the budget on its
    final field would sit in COLLECT_DATA forever."""
    actions, session, _ = rig()
    from tests.conftest import complete_applicant_data

    data = complete_applicant_data()
    for name, value in data.items():
        actions.save_applicant_field(session, name, value, as_of=AS_OF)

    assert session.state is State.KYC_HANDOFF
