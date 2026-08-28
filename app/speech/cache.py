"""Synthesis costs money; saying the same sentence twice should not.

WHY THIS EXISTS
---------------
Measured against Sarvam's live pricing: a twenty-turn conversation is about
Rs 9.83, of which Rs 9.00 -- **92%** -- is text-to-speech. On a one-time Rs 100
credit that is ten conversations, and this project needs to be demonstrated
more than ten times.

But most of what the agent says is not new. Every `fallback` in
`speech_acts.py` is a fixed literal. The consent explanation, the KYC handoff,
"your identity check hasn't finished yet" -- these are spoken in every single
conversation, in the same language, with the same voice. Synthesising them
again each time is paying repeatedly for an identical file.

WHAT IS NOT CACHED, AND WHY
---------------------------
Transcription. Two recordings of a person saying the same words are never
byte-identical, so the cache would never hit -- and a key loose enough to hit
would be a key that returns one caller's words for another caller's audio.
The failure would be silent, wrong, and in the one direction this project
cannot afford. `transcribe` delegates, always.

A WRAPPER, NOT A FEATURE OF THE VENDOR
--------------------------------------
`CachedSpeech` satisfies `SpeechProvider` and holds another one. Sarvam does
not know it is being cached, the next provider will not either, and neither
had to be modified. This is what the seam was for.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.speech.provider import Audio, SpeechProvider, Transcript

log = logging.getLogger(__name__)

# Sarvam's bulbul:v3 rate, for the estimate this reports. Wrong for another
# vendor, which is why the number is called an estimate and the character
# count -- which is exact -- is reported beside it.
RUPEES_PER_10K_CHARS = 30.0

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".speech-cache"


@dataclass
class SpeechSpend:
    """What was synthesised, and what was not paid for twice."""

    hits: int = 0
    misses: int = 0
    chars_synthesised: int = 0
    chars_served_from_cache: int = 0

    @property
    def rupees_spent(self) -> float:
        return self.chars_synthesised / 10_000 * RUPEES_PER_10K_CHARS

    @property
    def rupees_saved(self) -> float:
        return self.chars_served_from_cache / 10_000 * RUPEES_PER_10K_CHARS

    def summary(self) -> str:
        return (
            f"{self.hits} cached / {self.misses} synthesised; "
            f"~Rs {self.rupees_spent:.2f} spent, ~Rs {self.rupees_saved:.2f} saved"
        )


@dataclass
class CachedSpeech:
    """Satisfies `SpeechProvider`. Caches synthesis on disk, forever."""

    inner: SpeechProvider
    directory: Path = CACHE_DIR
    spend: SpeechSpend = field(default_factory=SpeechSpend)

    @property
    def name(self) -> str:
        return f"cached:{getattr(self.inner, 'name', 'speech')}"

    def _key(self, text: str, language: str) -> str:
        """Everything that changes the audio goes into the key.

        Model and speaker included: changing the voice must not keep serving
        the old one, and that is a bug you would hear rather than see in a
        test, which is the worst way to find it.
        """
        parts = [
            getattr(self.inner, "name", "speech"),
            getattr(self.inner, "tts_model", ""),
            getattr(self.inner, "speaker", ""),
            language,
            text,
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def synthesize(self, text: str, language: str) -> Audio:
        path = self.directory / f"{self._key(text, language)}.wav"

        if path.exists():
            self.spend.hits += 1
            self.spend.chars_served_from_cache += len(text)
            return Audio(data=path.read_bytes(), media_type="audio/wav")

        audio = self.inner.synthesize(text, language)

        self.spend.misses += 1
        self.spend.chars_synthesised += len(text)

        # Written after the call succeeded, never before. Caching a failure
        # would make one bad response permanent.
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio.data)
        except OSError:
            # A cache that cannot write is slow, not broken. The caller still
            # gets their audio.
            log.warning("could not write speech cache to %s", path, exc_info=True)

        return audio

    def transcribe(self, audio: Audio, language: str) -> Transcript:
        """Never cached. See the module docstring."""
        return self.inner.transcribe(audio, language)

    # -- Warming ------------------------------------------------------------

    def warm(self, phrases: list[tuple[str, str]]) -> SpeechSpend:
        """Synthesise a list of (text, language) pairs up front.

        Meant for the fixed fallbacks, which are known before anyone calls and
        are spoken in every conversation. Paying for them once, deliberately,
        beats discovering the bill one live demo at a time.
        """
        for text, language in phrases:
            self.synthesize(text, language)
        return self.spend
