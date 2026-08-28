"""Speech in and speech out, behind a seam like every other vendor here."""

from app.speech.provider import (
    Audio,
    MockSpeech,
    SpeechError,
    SpeechProvider,
    Transcript,
)
from app.speech.sarvam import SarvamSpeech

__all__ = [
    "Audio",
    "MockSpeech",
    "SpeechError",
    "SarvamSpeech",
    "SpeechProvider",
    "Transcript",
]
