"""The speech seam, and the judgement that sits behind it.

Offline. Nothing here reaches a vendor, which is the point of the seam
existing before any vendor is chosen.
"""

from __future__ import annotations

import pytest

from app.flow.hearing import (
    CONFIDENCE_FLOOR,
    DEFAULT_FLOOR,
    floor_for,
    is_confident_enough,
)
from app.flow.states import State
from app.speech import Audio, MockSpeech, SpeechProvider, Transcript

# ---------------------------------------------------------------------------
# The vocabulary.
# ---------------------------------------------------------------------------


def test_a_percentage_that_was_never_divided_is_rejected():
    """The bug this guard exists for is silent, which is why it is a guard.

    A provider reporting 0-100 would hand back 87.0, which clears every floor
    in hearing.py by a factor of a hundred -- so the consent gate would be
    wide open and every test would still pass.
    """
    with pytest.raises(ValueError, match="0-100"):
        Transcript(text="haan", language="hi-IN", confidence=87.0)

    with pytest.raises(ValueError):
        Transcript(text="haan", language="hi-IN", confidence=-0.1)


def test_confidence_may_be_unreported():
    """None is not zero, and must not be treated as a low score."""
    heard = Transcript(text="haan", language="hi-IN", confidence=None)
    assert heard.confidence is None


def test_silence_is_recognisable_as_silence():
    assert Transcript(text="   ", language="hi-IN").is_empty
    assert not Transcript(text="haan", language="hi-IN").is_empty


def test_the_mock_satisfies_the_protocol():
    assert isinstance(MockSpeech(), SpeechProvider)


# ---------------------------------------------------------------------------
# The mock, which every other test in the suite depends on staying boring.
# ---------------------------------------------------------------------------


def test_the_mock_hears_what_it_was_told_to_hear():
    mock = MockSpeech(
        hears=[
            Transcript(text="mmm", language="hi-IN", confidence=0.2),
            Transcript(text="haan", language="hi-IN", confidence=0.97),
        ]
    )
    clip = Audio(data=b"\x00\x01", media_type="audio/webm")

    first = mock.transcribe(clip, "hi-IN")
    second = mock.transcribe(clip, "hi-IN")
    third = mock.transcribe(clip, "hi-IN")

    assert (first.text, first.confidence) == ("mmm", 0.2)
    assert (second.text, second.confidence) == ("haan", 0.97)
    # Script exhausted: falls back to the default rather than raising, so a
    # test that only cares about the first two turns does not have to script
    # every turn after them.
    assert third.text == "haan"


def test_different_sentences_never_synthesise_to_the_same_audio():
    """Otherwise a wrong-language bug and a caching bug both look like success.

    If synthesize() returned a constant, a test asserting "we spoke the reply"
    would pass while speaking the wrong reply, in the wrong language, from a
    stale cache entry.
    """
    mock = MockSpeech()

    hindi = mock.synthesize("aapka khaata khul gaya", "hi-IN")
    tamil = mock.synthesize("aapka khaata khul gaya", "ta-IN")
    other = mock.synthesize("kuch aur", "hi-IN")

    assert hindi.data != tamil.data, "same text, different language, same bytes"
    assert hindi.data != other.data
    assert hindi.media_type == "audio/wav"
    assert mock.spoken == [
        ("aapka khaata khul gaya", "hi-IN"),
        ("aapka khaata khul gaya", "ta-IN"),
        ("kuch aur", "hi-IN"),
    ]


# ---------------------------------------------------------------------------
# How sure we have to be, which is a per-state question.
# ---------------------------------------------------------------------------


def test_every_state_has_a_floor():
    """Including ones added later -- floor_for falls back rather than raising."""
    for state in State:
        assert 0.0 <= floor_for(state) <= 1.0


def test_consent_demands_more_certainty_than_a_name_does():
    """The counterintuitive one, pinned so it cannot be 'fixed' by accident.

    A misheard name is read back, has to survive KYC, and the applicant is
    there to correct it. A misheard consent is caught by nobody and makes
    everything after it unlawful processing that looks entirely normal.

    The other half matters just as much: a high floor on COLLECT_DATA would
    punish unusual names, regional pronunciations and heavy accents -- which
    is precisely the population this product exists for.
    """
    assert floor_for(State.CONSENT) > floor_for(State.COLLECT_DATA)
    assert floor_for(State.CONSENT) > DEFAULT_FLOOR
    assert floor_for(State.COLLECT_DATA) < DEFAULT_FLOOR


def test_money_states_are_held_to_the_consent_standard():
    for state in (State.FUNDING, State.FIRST_TRANSACTION):
        assert floor_for(state) >= 0.8


def test_a_mumbled_yes_at_the_consent_gate_is_not_acted_on():
    assert not is_confident_enough(0.5, State.CONSENT)
    assert is_confident_enough(0.9, State.CONSENT)


def test_the_same_mumble_is_accepted_while_collecting_a_name():
    """Same audio, different state, different answer. That is the whole idea."""
    mumbled = 0.5
    assert not is_confident_enough(mumbled, State.CONSENT)
    assert is_confident_enough(mumbled, State.COLLECT_DATA)


def test_an_unreported_confidence_passes_and_that_is_deliberate():
    """Documented in hearing.py: rejecting None would make quiet providers
    unusable, so a quiet provider instead disables this protection -- which
    makes "does it report confidence" a selection criterion, not a detail."""
    for state in State:
        assert is_confident_enough(None, state)


def test_the_floors_are_all_plausible_probabilities():
    for state, floor in CONFIDENCE_FLOOR.items():
        assert 0.0 < floor <= 1.0, f"{state} has a nonsense floor"
