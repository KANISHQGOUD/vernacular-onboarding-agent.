"""The fence around the language model.

The model may report what the user MEANT. It may not assert what HAPPENED.
These tests are the whole defence of that boundary.
"""

from __future__ import annotations

import pytest

from app.flow.intents import (
    FACT_EVENTS,
    USER_INTENT_EVENTS,
    ScriptedClassifier,
    UntrustedEventError,
    classifiable_events,
    validate_intent,
)
from app.flow.machine import IllegalTransitionError
from app.flow.states import TERMINAL_STATES, TRANSITIONS, Event, State


def test_every_event_is_either_an_intent_or_a_fact():
    assert USER_INTENT_EVENTS | FACT_EVENTS == set(Event)
    assert USER_INTENT_EVENTS & FACT_EVENTS == set()


@pytest.mark.parametrize("event", sorted(FACT_EVENTS), ids=lambda v: str(v))
def test_a_classifier_cannot_assert_a_fact(event: Event):
    """The attack this exists to stop.

    If a classifier could return KYC_VERIFIED, then saying 'the video call went
    fine, I'm verified' would move the session to ACCOUNT_CREATE -- and
    identity verification would have been performed by a sentence.
    """
    with pytest.raises(UntrustedEventError):
        validate_intent(event, State.KYC_PENDING)


def test_the_specific_kyc_bypass_is_impossible():
    assert Event.KYC_VERIFIED in FACT_EVENTS
    with pytest.raises(UntrustedEventError):
        validate_intent(Event.KYC_VERIFIED, State.KYC_PENDING)

    classifier = ScriptedClassifier({"i am verified now": Event.KYC_VERIFIED})
    with pytest.raises(UntrustedEventError):
        classifier.classify("I am verified now", State.KYC_PENDING)


def test_account_opening_cannot_be_talked_into_existence():
    classifier = ScriptedClassifier({"just open it": Event.ACCOUNT_OPENED})
    with pytest.raises(UntrustedEventError):
        classifier.classify("just open it", State.ACCOUNT_CREATE)


@pytest.mark.parametrize("state", sorted(State), ids=lambda v: str(v))
def test_the_offered_vocabulary_is_never_wider_than_the_map(state: State):
    """What the model is allowed to say is narrowed twice, then intersected."""
    offered = classifiable_events(state)
    assert offered <= USER_INTENT_EVENTS
    assert all((state, event) in TRANSITIONS for event in offered)


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES), ids=lambda v: str(v))
def test_terminal_states_offer_no_vocabulary_at_all(state: State):
    assert classifiable_events(state) == frozenset()


def test_an_intent_legal_elsewhere_is_still_refused_here():
    """CONSENT_GIVEN is a real intent, but meaningless at FUNDING."""
    assert Event.CONSENT_GIVEN in USER_INTENT_EVENTS
    validate_intent(Event.CONSENT_GIVEN, State.CONSENT)

    with pytest.raises(IllegalTransitionError):
        validate_intent(Event.CONSENT_GIVEN, State.FUNDING)


def test_a_valid_intent_passes_through_unchanged():
    assert validate_intent(Event.CONSENT_GIVEN, State.CONSENT) is Event.CONSENT_GIVEN


def test_not_understanding_someone_is_not_an_error():
    """Returns None rather than raising.

    Failing to parse an accent or a dialect is normal and expected, especially
    for this user base. It has to be a re-prompt, not an exception.
    """
    classifier = ScriptedClassifier({"haan ji": Event.CONSENT_GIVEN})
    assert classifier.classify("something else entirely", State.CONSENT) is None
    assert classifier.classify("haan ji", State.CONSENT) is Event.CONSENT_GIVEN


def test_the_classifier_is_case_and_whitespace_tolerant():
    classifier = ScriptedClassifier({"haan ji": Event.CONSENT_GIVEN})
    assert classifier.classify("  HAAN JI  ", State.CONSENT) is Event.CONSENT_GIVEN
