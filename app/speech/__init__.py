"""Speech in and speech out, behind a seam like every other vendor here."""

from app.speech.provider import (
    Audio,
    MockSpeech,
    SpeechError,
    SpeechProvider,
    Transcript,
)

__all__ = [
    "Audio",
    "MockSpeech",
    "SpeechError",
    "SpeechProvider",
    "Transcript",
]
