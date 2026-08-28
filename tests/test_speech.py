"""The speech seam, and the judgement that sits behind it.

Offline. Nothing here reaches a vendor, which is the point of the seam
existing before any vendor is chosen.
"""

from __future__ import annotations

import os
import unicodedata

import pytest

from app.flow.hearing import (
    CONFIDENCE_FLOOR,
    DEFAULT_FLOOR,
    floor_for,
    is_confident_enough,
)
from app.flow.states import State
from app.speech import Audio, MockSpeech, SarvamSpeech, SpeechProvider, Transcript
from app.speech.sarvam import from_vendor, to_vendor

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


# ---------------------------------------------------------------------------
# Sarvam. Offline by default -- the live tests cost real money.
# ---------------------------------------------------------------------------


class FakeSarvamClient:
    """Enough of the vendor SDK to exercise the adapter without a network."""

    def __init__(self, transcript="मेरा नाम आशा देवी है", language_code="hi-IN"):
        self.speech_to_text = self._Stt(transcript, language_code)
        self.text_to_speech = self._Tts()

    class _Stt:
        def __init__(self, transcript, language_code):
            self.transcript_value = transcript
            self.language_code = language_code
            self.calls = []

        def transcribe(self, **kwargs):
            self.calls.append(kwargs)
            import types

            return types.SimpleNamespace(
                transcript=self.transcript_value,
                language_code=self.language_code,
                # The vendor really does return this, and it is NOT a
                # transcription confidence. Present here so a test can prove
                # the adapter ignores it.
                language_probability=0.98,
            )

    class _Tts:
        def __init__(self):
            self.calls = []

        def convert(self, **kwargs):
            import base64
            import types

            self.calls.append(kwargs)
            return types.SimpleNamespace(audios=[base64.b64encode(b"RIFFfake").decode()])


def fake_sarvam(**kwargs) -> SarvamSpeech:
    provider = SarvamSpeech(api_key="not-a-real-key", **kwargs)
    provider._client = FakeSarvamClient()
    return provider


def test_sarvam_satisfies_the_protocol():
    assert isinstance(fake_sarvam(), SpeechProvider)


def test_odia_is_spelled_the_vendor_way_only_at_the_boundary():
    """Sarvam spells Odia `od-IN`; ISO 639-1, and this project, say `or-IN`.

    The translation belongs in the adapter. One vendor's spelling is not a
    reason to make the shared vocabulary wrong, and the next provider will
    have a different quirk in a different place.
    """
    assert to_vendor("or-IN") == "od-IN"
    assert from_vendor("od-IN") == "or-IN"

    # Everything else passes straight through, both ways.
    for code in ("hi-IN", "ta-IN", "en-IN"):
        assert to_vendor(code) == code
        assert from_vendor(code) == code

    provider = fake_sarvam()
    provider.synthesize("ନମସ୍କାର", "or-IN")
    assert provider.client.text_to_speech.calls[0]["target_language_code"] == "od-IN"


def test_the_project_never_learns_the_vendor_spelling():
    """`or-IN` is what the rest of the codebase speaks, and stays that way."""
    from app.languages import LANGUAGE_CODES

    assert "or-IN" in LANGUAGE_CODES
    assert "od-IN" not in LANGUAGE_CODES


def test_language_probability_is_not_used_as_a_confidence():
    """The gap that matters, pinned so it cannot be closed by accident.

    Sarvam returns `language_probability` -- how sure it is the audio was
    Hindi -- and no transcription confidence at all. Passing that number
    through would put a value into hearing.py's floors that does not mean what
    they assume, and the consent gate would then be comparing 0.85 against an
    answer to a different question.

    So confidence is None, `is_confident_enough` lets the turn through, and
    the protection genuinely does not exist on this provider. Better named
    than quietly faked.
    """
    heard = fake_sarvam().transcribe(Audio(data=b"x", media_type="audio/wav"), "hi-IN")

    assert heard.text
    assert heard.confidence is None, "language_probability leaked in as confidence"
    assert not SarvamSpeech(api_key="x").reports_confidence

    from app.flow.hearing import is_confident_enough
    from app.flow.states import State

    assert is_confident_enough(heard.confidence, State.CONSENT), (
        "documented consequence: this provider cannot guard the consent gate"
    )


def test_the_transcript_reports_the_language_that_came_back():
    """Not the one we asked for. They can differ, and the caller should see it."""
    provider = fake_sarvam()
    provider._client = FakeSarvamClient(transcript="hello", language_code="en-IN")

    heard = provider.transcribe(Audio(data=b"x", media_type="audio/wav"), "hi-IN")
    assert heard.language == "en-IN"
    assert heard.provider == "sarvam"


def test_the_verified_defaults_are_the_ones_that_answered():
    """The SDK's Literal[...] lists disagree with the live API in both
    directions -- bulbul:v2 is deprecated, anushka is invalid for v3, and
    niharika exists only on the API side. These are the measured values."""
    provider = SarvamSpeech(api_key="x")
    assert provider.tts_model == "bulbul:v3"
    assert provider.speaker != "anushka"

    fake = fake_sarvam()
    fake.synthesize("hi", "hi-IN")
    sent = fake.client.text_to_speech.calls[0]
    assert sent["model"] == "bulbul:v3"
    assert sent["speaker"] == provider.speaker


def test_constructing_it_opens_no_connection():
    """The client is built on first use, so the suite can construct providers
    freely and `sarvamai` need not be installed for the rest to run."""
    provider = SarvamSpeech(api_key="x")
    assert provider._client is None


# ---------------------------------------------------------------------------
# The live ones. Opt IN, not opt out -- these spend a finite prepaid balance.
# ---------------------------------------------------------------------------

LIVE = os.getenv("SARVAM_LIVE", "").strip() == "1"

live_only = pytest.mark.skipif(
    not LIVE,
    reason="set SARVAM_LIVE=1 to spend real credit on these",
)


@live_only
def test_a_hindi_sentence_survives_a_round_trip():
    """Synthesise, then transcribe what came back.

    Note what this does NOT prove: Sarvam transcribing its own clean synthesis
    is far easier than a person in a room with a fan on. This is a check that
    the plumbing works, not evidence about accuracy -- that needs the WER
    comparison against real recordings.
    """
    provider = SarvamSpeech(api_key=os.environ["SARVAM_API_KEY"])
    said = "मेरा नाम आशा देवी है"

    audio = provider.synthesize(said, "hi-IN")
    assert len(audio) > 1000
    assert audio.data[:4] == b"RIFF"

    heard = provider.transcribe(audio, "hi-IN")

    # NOT an exact match, and not a substring check either. The first version
    # of this asserted `said in heard.text` and passed, then failed on the very
    # next run: recognition is not deterministic, and it had added a danda one
    # time and not the next. A flaky test on the speech path is worse than no
    # test, because people learn to re-run it until it goes green.
    #
    # So this asserts what it claims to: the plumbing carried real Devanagari
    # there and back. Accuracy is the WER comparison's job, against recordings
    # of actual people rather than the vendor's own clean synthesis.
    # Strip PUNCTUATION, not "anything not alnum". The first attempt used
    # `ch.isalnum()` and turned आशा into आश, because Devanagari vowel signs
    # are combining marks (category Mn) and isalnum() is False for them.
    # A filter written that way quietly mutilates every script this project
    # exists to serve, and would have looked like a vendor accuracy problem.
    stripped = "".join(
        ch for ch in heard.text if not unicodedata.category(ch).startswith("P")
    )
    assert stripped.split(), f"nothing came back: {heard.text!r}"
    assert "आशा" in stripped, f"lost the name entirely: {heard.text!r}"
    assert heard.language == "hi-IN"
    assert heard.confidence is None, "the vendor started reporting confidence"
