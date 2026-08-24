"""The seam a speech vendor drops into -- and the vocabulary that outlives it.

WHY THIS IS NOT JUST `transcribe(audio) -> str`
-----------------------------------------------
Speech can happen in two places, and they are not the same shape:

    client-side   the browser listens, transcribes, and sends text.
                  The server never receives a byte of audio.
    server-side   the browser sends audio, and a provider here transcribes it.

A protocol written only for the second is dead code on the first, and the
first is what ships without an API key. So the thing both have in common is
made the centre of this file: whoever did the listening, the server ends up
holding a `Transcript`.

CONFIDENCE IS THE POINT
-----------------------
`Transcript.confidence` is the field this seam exists for. Text alone loses the
one thing that matters -- speech recognition does not fail loudly. It returns a
plausible sentence with no error, and "nahi" heard as "haan" at the consent
gate records a consent that was never given. That is a correctness failure, and
the only signal that it might have happened is a number the recogniser already
computes and a text-only interface throws away.

What to DO about a low number is not decided here. `app/flow/hearing.py` owns
that, because how sure you need to be depends on what the answer will be used
for -- and the machine is the thing that knows. This file only reports.

NO STREAMING YET, DELIBERATELY
------------------------------
`is_final` exists because every real recogniser emits interim guesses that
change as you keep talking, and a design that pretends otherwise has to be torn
up when turn-taking arrives. Nothing consumes interim results yet. The field is
here so that adding VAD later is a new caller, not a new protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Below this, a float is almost certainly a mistake -- a percentage that was
# never divided, or a provider reporting 0-100. Caught rather than trusted,
# because a confidence of 87.0 silently passes every floor in hearing.py.
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.0


class SpeechError(RuntimeError):
    """A speech provider could not do what was asked."""


@dataclass(frozen=True, slots=True)
class Audio:
    """Encoded audio, with the encoding named rather than assumed.

    `media_type` is not decoration. Browsers hand out `audio/webm;codecs=opus`,
    phones hand out `audio/mp4`, and Indic ASR models generally want 16 kHz
    mono WAV. A pipeline that passes bytes around without saying what they are
    fails at the vendor boundary, in production, on someone else's device.
    """

    data: bytes
    media_type: str
    sample_rate_hz: int | None = None

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was heard, and how sure the recogniser was.

    `confidence` is None when the provider does not report one. That is not the
    same as zero, and conflating them would either reject everything from a
    provider that stays quiet, or silently trust it. `hearing.py` decides which
    way that falls; here it stays honest about not knowing.
    """

    text: str
    language: str
    confidence: float | None = None
    is_final: bool = True
    provider: str = ""

    def __post_init__(self) -> None:
        if self.confidence is None:
            return
        if not MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE:
            raise ValueError(
                f"confidence must be within {MIN_CONFIDENCE}..{MAX_CONFIDENCE}, "
                f"got {self.confidence!r} -- a 0-100 scale reaching hearing.py "
                f"would clear every floor it has"
            )

    @property
    def is_empty(self) -> bool:
        """Silence, or a recogniser that heard nothing it could render."""
        return not self.text.strip()


@runtime_checkable
class SpeechProvider(Protocol):
    """Two methods, and neither of them decides anything.

    There is no `detect_language`, on purpose. The language is chosen once, by
    the applicant, through a state the machine controls -- and re-detecting it
    per utterance would let a mumbled sentence silently switch the conversation
    into a language the caller does not read. Same argument as the tool
    registry: the risk is closed by the capability being absent.
    """

    name: str

    def transcribe(self, audio: Audio, language: str) -> Transcript: ...

    def synthesize(self, text: str, language: str) -> Audio: ...


@dataclass
class MockSpeech:
    """Offline stand-in, so the suite never needs a key or a network.

    Scripted rather than random, for the reason `MockKycProvider` gives: a mock
    that returned a low confidence one time in five would make the re-prompt
    test flaky, and a flaky test on a path that guards consent is worse than no
    test, because people learn to re-run it until it passes.

    Each entry in `hears` is consumed by one `transcribe` call, so a script of
    [uncertain, clear] reproduces the ordinary case -- the first attempt is
    mumbled, the second is not.
    """

    name: str = "mock"
    hears: list[Transcript] = field(default_factory=list)
    default_text: str = "haan"
    default_confidence: float | None = 0.95

    _heard_index: int = 0
    spoken: list[tuple[str, str]] = field(default_factory=list)

    def transcribe(self, audio: Audio, language: str) -> Transcript:
        if self._heard_index < len(self.hears):
            scripted = self.hears[self._heard_index]
            self._heard_index += 1
            return scripted

        return Transcript(
            text=self.default_text,
            language=language,
            confidence=self.default_confidence,
            provider=self.name,
        )

    def synthesize(self, text: str, language: str) -> Audio:
        """Deterministic bytes, not silence.

        Derived from the text so two different sentences never produce equal
        audio -- otherwise a caching bug or a wrong-language bug would look
        exactly like correct behaviour to every test that compares output.
        """
        self.spoken.append((text, language))
        digest = hashlib.sha256(f"{language}:{text}".encode()).digest()
        return Audio(data=digest, media_type="audio/wav", sample_rate_hz=16_000)
