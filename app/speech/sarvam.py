"""Sarvam AI behind the speech seam.

Indic speech from a vendor that trains on Indian accents and code-mixing,
which is the half of the problem the browser cannot do: Windows ships no Indic
voices at all, so `web/index.html` can only read Devanagari in an English one.

EVERYTHING HERE WAS MEASURED, NOT READ
--------------------------------------
The installed SDK's type hints disagree with the live API in three places, all
found by calling it:

    bulbul:v2   the SDK offers it; the API returns 400 "has been deprecated".
    anushka     the SDK lists it first; the API rejects it for bulbul:v3.
    niharika    the API offers it for v3; the SDK has never heard of it.

So the defaults below are the ones that answered, not the ones documented, and
`scripts/introspect_sdk.py` is the thing to re-run after an upgrade rather than
trusting `Literal[...]` in a signature.

THE CONFIDENCE PROBLEM, STATED PLAINLY
--------------------------------------
`SpeechToTextResponse` has no transcription confidence. It has
`language_probability`, which answers "how sure am I this was Hindi" -- a
different question, and using it as a stand-in would put a number into
`hearing.py` that does not mean what the floors assume it means.

So `Transcript.confidence` is None here, honestly, and
`is_confident_enough` therefore lets every Sarvam turn through. That is a real
hole and it is worth naming: the consent gate's protection against "nahi"
heard as "haan" does not exist on this provider. `reports_confidence` is False
so a deployment can detect that rather than discover it.

Two ways out, neither built yet: read the answer back at the regulated gates
and require a second confirmation, or keep browser STT (which does report
confidence) and use Sarvam only for synthesis. The second is also the cheaper
arrangement, since TTS is most of the bill.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

from app.speech.provider import Audio, SpeechError, Transcript

log = logging.getLogger(__name__)

# Verified against the live API on 2026-08-28. The SDK's Literal[...] lists
# are stale in both directions -- see the module docstring.
DEFAULT_TTS_MODEL = "bulbul:v3"
DEFAULT_STT_MODEL = "saaras:v4"
DEFAULT_SPEAKER = "ritu"

# Sarvam spells Odia `od-IN`. The ISO 639-1 code is `or`, which is what
# `languages.py` uses and what every other part of this project speaks.
#
# The mapping lives here, at the vendor boundary, rather than in the shared
# vocabulary. One vendor's spelling is not a reason to make the domain wrong,
# and the next provider will have a different quirk in a different place.
TO_SARVAM: dict[str, str] = {"or-IN": "od-IN"}
FROM_SARVAM: dict[str, str] = {v: k for k, v in TO_SARVAM.items()}


def to_vendor(code: str) -> str:
    return TO_SARVAM.get(code, code)


def from_vendor(code: str | None) -> str | None:
    if code is None:
        return None
    return FROM_SARVAM.get(code, code)


@dataclass
class SarvamSpeech:
    """Satisfies `SpeechProvider`. One vendor, two methods, no decisions.

    There is no `detect_language` here for the same reason the protocol has
    none: the applicant chooses their language once, through a state the
    machine owns.
    """

    api_key: str
    name: str = "sarvam"
    tts_model: str = DEFAULT_TTS_MODEL
    stt_model: str = DEFAULT_STT_MODEL
    speaker: str = DEFAULT_SPEAKER

    #: This provider cannot fill in `Transcript.confidence`. See the module
    #: docstring -- it is a gap in a safety check, not a missing nicety.
    reports_confidence: bool = False

    _client: Any = field(default=None, repr=False, compare=False)

    @property
    def client(self) -> Any:
        """Built on first use, so constructing the provider costs nothing.

        Imported lazily too: the rest of the project runs, and the whole test
        suite passes, without `sarvamai` installed.
        """
        if self._client is None:
            from sarvamai import SarvamAI

            self._client = SarvamAI(api_subscription_key=self.api_key)
        return self._client

    # -- Speech in ----------------------------------------------------------

    def transcribe(self, audio: Audio, language: str) -> Transcript:
        try:
            response = self.client.speech_to_text.transcribe(
                file=("audio", audio.data, audio.media_type),
                model=self.stt_model,
                language_code=to_vendor(language),
            )
        except Exception as exc:  # noqa: BLE001 - the vendor's tree is wide
            raise SpeechError(f"sarvam transcription failed: {exc}") from exc

        return Transcript(
            text=response.transcript or "",
            # What came back, not what we asked for -- they can differ, and the
            # caller should see the answer rather than its own question.
            language=from_vendor(getattr(response, "language_code", None)) or language,
            # Deliberately None, never `language_probability`. See the module
            # docstring: that number answers a different question.
            confidence=None,
            provider=self.name,
        )

    # -- Speech out ---------------------------------------------------------

    def synthesize(self, text: str, language: str) -> Audio:
        try:
            response = self.client.text_to_speech.convert(
                text=text,
                target_language_code=to_vendor(language),
                speaker=self.speaker,
                model=self.tts_model,
            )
        except TypeError:
            # Older SDKs name it `language_code`. Both spellings have shipped.
            response = self.client.text_to_speech.convert(
                text=text,
                language_code=to_vendor(language),
                speaker=self.speaker,
                model=self.tts_model,
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"sarvam synthesis failed: {exc}") from exc

        audios = getattr(response, "audios", None)
        if not audios:
            raise SpeechError("sarvam returned no audio")

        return Audio(
            data=base64.b64decode(audios[0]),
            media_type="audio/wav",
        )
