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
from tests.conftest import AS_OF, save_field

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

    # The last attempt is accepted -- and still read back, because a value
    # we could not validate is the one most worth confirming with the only
    # party who knows.
    actions.save_applicant_field(session, "mobile", "12345", as_of=AS_OF)
    assert session.pending_field == "mobile"
    actions.confirm_pending_field(session)

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
    save_field(actions, session, "mobile", "9812345678")

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
        save_field(actions, session, name, value)

    assert session.state is State.KYC_HANDOFF


# ---------------------------------------------------------------------------
# Read-back. The confidence the provider does not report, asked for instead.
# ---------------------------------------------------------------------------


def test_a_high_stakes_field_is_not_written_down_until_it_is_agreed():
    """Staged, not stored. A "no" should never have to un-write anything."""
    actions, session, _ = rig()

    actions.save_applicant_field(session, "full_name", "Asha Devi", as_of=AS_OF)

    assert session.pending_field == "full_name"
    assert session.pending_value == "Asha Devi"
    assert "full_name" not in session.applicant


def test_a_lower_stakes_field_is_written_straight_down():
    """Five read-backs is a real cost in a flow built for people who already
    find forms long. An address is re-askable; a name on an account is not."""
    actions, session, _ = rig()

    actions.save_applicant_field(session, "address", "14 Gandhi Marg, Jaipur", as_of=AS_OF)

    assert session.pending_field is None
    assert session.applicant["address"] == "14 Gandhi Marg, Jaipur"


def test_saying_no_discards_it_rather_than_offering_it_back():
    """A value they have already denied is not a starting point for a guess.
    Keeping it as a suggestion would be arguing with them."""
    actions, session, _ = rig()
    actions.save_applicant_field(session, "full_name", "Asha Devri", as_of=AS_OF)

    actions.reject_pending_field(session)

    assert session.pending_field is None
    assert session.pending_value is None
    assert "full_name" not in session.applicant
    assert session.corrections("full_name") == 1


def test_confirming_the_last_field_still_completes_the_set():
    """The completion check lives on the commit path, so it must fire whether
    the value arrived confirmed or not -- otherwise a session whose final
    field needed a read-back sits in COLLECT_DATA forever."""
    actions, session, _ = rig()
    from tests.conftest import complete_applicant_data

    data = complete_applicant_data()
    for name in ("address", "occupation", "mobile", "date_of_birth"):
        save_field(actions, session, name, data[name])

    assert session.state is State.COLLECT_DATA

    actions.save_applicant_field(session, "full_name", data["full_name"], as_of=AS_OF)
    assert session.state is State.COLLECT_DATA, "completed before it was agreed"

    actions.confirm_pending_field(session)
    assert session.state is State.KYC_HANDOFF


def test_corrections_survive_erasure_but_the_value_does_not():
    """How often a read-back was denied is operational; what was read back is
    the person. They are stored on opposite sides of that line."""
    actions, session, _ = rig()
    actions.save_applicant_field(session, "full_name", "Asha Devri", as_of=AS_OF)
    actions.reject_pending_field(session)
    actions.save_applicant_field(session, "full_name", "Asha Devi", as_of=AS_OF)

    session.erase_personal_data()

    assert session.pending_field is None
    assert session.pending_value is None
    assert session.corrections("full_name") == 1


# ---------------------------------------------------------------------------
# When asking is not working, stop asking the same way.
# ---------------------------------------------------------------------------


def test_the_tactic_changes_rather_than_repeating():
    """Somebody read back the wrong value twice has already said it correctly
    twice. A third identical question is the machine insisting."""
    from app.flow.hearing import next_tactic

    assert next_tactic(0) == "ask"
    assert next_tactic(1) == "ask"
    assert next_tactic(2) == "spell"
    assert next_tactic(3) == "escalate"
    assert next_tactic(9) == "escalate"


def test_three_denials_reach_a_person_rather_than_a_wall():
    """The end of the budget is a handoff, not a refusal. Same shape as the
    KYC retry budget: a machine that cannot do this says so and gets somebody
    who can."""
    from datetime import date

    from app.conversation import Conversation
    from app.llm.phrasing import Phraser
    from tests.test_conversation import Classification, FakeReader

    audit = InMemoryAuditSink()
    machine = Machine(audit)
    actions = Actions(
        machine=machine,
        kyc=MockKycProvider(),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )
    readings = []
    for _ in range(3):
        readings += [
            Classification(Event.DATA_PROVIDED, "full_name", "Asha Devri"),
            Classification(Event.FIELD_CORRECTED),
        ]
    conversation = Conversation(
        actions, FakeReader(*readings), Phraser(client=None), as_of=date(2026, 9, 2)
    )

    session = Session(session_id="ladder", state=State.COLLECT_DATA)
    replies = []
    for _ in range(3):
        conversation.respond(session, "my name is ...")
        replies.append(conversation.respond(session, "no that is wrong"))

    # Ask, then spell, then a person.
    assert "one letter" in replies[1].text.lower()
    assert session.state is State.HUMAN_HANDOFF
    assert "full_name" in session.notes["escalation_reason"]


def test_the_spelling_request_does_not_blame_them():
    """They have said it correctly twice and been told it was wrong. The
    difficulty belongs to the machine."""
    from app.llm.speech_acts import SPEECH_ACTS, SpeechActKind

    spec = SPEECH_ACTS[SpeechActKind.ASK_TO_SPELL]
    spoken = spec.fallback.lower()

    assert "sorry" in spoken
    assert "i'm still not getting" in spoken or "not getting that right" in spoken
    for blame in ("you said", "you were", "your mistake", "speak up"):
        assert blame not in spoken
