"""The languages this agent will speak, as a closed set.

Constrained for the same reason everything else here is constrained: the
classifier picks the language from this list, so it cannot return a locale the
speech layer has never heard of. An unconstrained model asked "what language
is this?" will happily answer "Hinglish" or "Marwari" or "hi" -- all
reasonable answers, none of which a text-to-speech vendor accepts.

The set is deliberately short. Ten languages the agent can actually be tested
in beats twenty-two it claims and mispronounces, and a wrong language on a
call to someone who cannot read is not a cosmetic failure.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

DEFAULT_LANGUAGE = "en-IN"

# BCP-47 code -> the name to show a human. Codes match what Indic TTS vendors
# expect, which is why they carry the -IN region rather than the bare tag.
SUPPORTED_LANGUAGES: Mapping[str, str] = MappingProxyType(
    {
        "en-IN": "English",
        "hi-IN": "हिन्दी (Hindi)",
        "bn-IN": "বাংলা (Bengali)",
        "ta-IN": "தமிழ் (Tamil)",
        "te-IN": "తెలుగు (Telugu)",
        "mr-IN": "मराठी (Marathi)",
        "gu-IN": "ગુજરાતી (Gujarati)",
        "kn-IN": "ಕನ್ನಡ (Kannada)",
        "ml-IN": "മലയാളം (Malayalam)",
        "pa-IN": "ਪੰਜਾਬੀ (Punjabi)",
        "or-IN": "ଓଡ଼ିଆ (Odia)",
    }
)

LANGUAGE_CODES: tuple[str, ...] = tuple(SUPPORTED_LANGUAGES)


def is_supported(code: str | None) -> bool:
    return code in SUPPORTED_LANGUAGES


def display_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, code)


def normalise(code: str | None) -> str:
    """Best-effort tidy-up of a language code, falling back to the default.

    Accepts a bare tag ("hi") and returns the regional one the speech layer
    wants ("hi-IN"), because that mistake is easy to make and cheap to fix.
    """
    if not code:
        return DEFAULT_LANGUAGE
    code = code.strip()
    if code in SUPPORTED_LANGUAGES:
        return code
    bare = code.split("-", 1)[0].lower()
    for supported in SUPPORTED_LANGUAGES:
        if supported.split("-", 1)[0] == bare:
            return supported
    return DEFAULT_LANGUAGE
