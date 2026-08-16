"""The one place that knows a language model vendor exists.

Everything above this file talks to protocols -- `IntentClassifier`, and the
phrasing seam. Nothing in `app/flow/` or `app/providers/` imports this module,
which is why moving from one provider to another touched three docstrings and
a requirements file rather than the state machine.

VERIFIED AGAINST THE INSTALLED SDK, NOT THE DOCS
------------------------------------------------
google-genai 2.18.1. These are read off the actual installed package:

    client       genai.Client(api_key=...)
    call         client.models.generate_content(model=, contents=, config=)
    config       types.GenerateContentConfig(
                     system_instruction=, response_mime_type=,
                     response_schema=, thinking_config=,
                     max_output_tokens=, temperature=, safety_settings=)
    schema       types.Schema(type=types.Type.STRING, enum=[...])
    thinking     types.ThinkingConfig(thinking_budget=int,
                                      thinking_level=MINIMAL|LOW|MEDIUM|HIGH)
    response     .text (property), .parsed (field), .function_calls
    errors       APIError <- ClientError, ServerError

Re-run `scripts/introspect_sdk.py` after any SDK upgrade rather than trusting
this comment. A comment is a claim; the introspection is a measurement.

MEASURED AGAINST THE LIVE FREE TIER, 2026-08-15
-----------------------------------------------
    gemini-2.5-flash      DEAD -- "no longer available"
    gemini-3.7-flash      exists, but free-tier quota rejects it
    gemini-3.5-flash      works. all thinking levels. ~1.0-1.4s per call
    free tier limit       5 requests per minute, per model
    enum mode             works, and classified Hindi correctly

That 5 RPM ceiling is the constraint the whole design bends around. Two model
calls per conversational turn would cap the agent at 2.5 turns a minute, which
is not a conversation. So:

  - classification calls the model, because it genuinely cannot be cached
  - phrasing mostly does NOT, because the same speech act in the same language
    is the same sentence every time (see phrasing.py)

A 429 is therefore expected traffic on this tier, not an exception. It is
caught and degraded like any other failure -- never retried mid-call, because
the retry-after is measured in tens of seconds and there is a person waiting.

DEGRADING, NOT CRASHING
-----------------------
A bank agent that raises on a network blip has hung up on somebody mid-
application. Every call here returns `None` on failure instead. The callers
already treat `None` as "I could not tell" -- which the conversation layer
turns into a re-prompt, and which was designed in from the start precisely so
that a model failure and a misheard sentence take the same path.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import errors, types

log = logging.getLogger(__name__)

# Overridable without editing code, because model availability on the free tier
# moves faster than this repo does -- gemini-2.5-flash was the obvious default
# and is already retired.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Both jobs -- classifying an utterance and phrasing one sentence -- are short
# and latency-critical on a voice call. Neither needs deep reasoning; the
# reasoning already happened, deterministically, in the state machine.
CLASSIFY_THINKING = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
PHRASING_THINKING = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

# Turned off for CLASSIFICATION ONLY, and the argument is narrow enough to be
# worth stating rather than assuming.
#
# This call cannot emit harmful content. Its output space is an enum of at most
# seven fixed strings that this repo wrote. A content filter on that output has
# nothing to protect against -- there is no sentence it could block, because
# there is no sentence.
#
# What the filter CAN do is fail closed on the input, and that has a real cost.
# Applicants describe their own circumstances to explain why they need an
# account: debt, illness, a death in the family, domestic difficulty. A filter
# tripping on that returns nothing, which this code correctly reads as "I could
# not understand you" -- so a distressed caller gets asked to repeat themselves,
# and nobody ever learns why.
#
# Phrasing keeps the defaults, because phrasing does generate free text.
_CLASSIFY_SAFETY = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


class MissingApiKeyError(RuntimeError):
    """No key configured. Raised at construction, never mid-conversation."""


def load_api_key(explicit: str | None = None) -> str:
    key = explicit or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise MissingApiKeyError(
            "GOOGLE_API_KEY is empty. Get a free key at "
            "https://aistudio.google.com/apikey and paste it into .env"
        )
    return key


def api_key_available() -> bool:
    """True when a live call is possible.

    The suite runs entirely offline against fakes -- 5 requests a minute makes
    a live test suite both slow and flaky. This exists so a caller can decide
    whether a live path is worth attempting at all, without catching an
    exception to find out.
    """
    return bool(os.getenv("GOOGLE_API_KEY", "").strip())


@dataclass
class LlmClient:
    """Thin wrapper. Owns the vendor client and the failure policy."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    _client: genai.Client = field(init=False)

    def __post_init__(self) -> None:
        self._client = genai.Client(api_key=load_api_key(self.api_key))

    # -- The classifier's call: one constrained object ----------------------

    def extract(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema: types.Schema,
    ) -> dict[str, Any] | None:
        """Return one object matching `schema`, or None.

        Every field the caller cares about is constrained by the schema rather
        than requested in the prompt -- the intent is an enum, the field name
        is an enum, the language is an enum. The model is not asked what should
        happen; it is handed a shape and asked to fill it in. That narrowing is
        the whole reason a language model is safe to have in a banking flow.

        One call rather than two, deliberately. Classifying the utterance and
        extracting the value out of it are the same reading of the same
        sentence, and on a 5-requests-per-minute budget a second round trip to
        re-read a sentence we have already read is not affordable.

        The caller still re-checks everything that comes back. The schema is
        the wall; the re-check is the proof the wall held.
        """
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=CLASSIFY_THINKING,
                    safety_settings=_CLASSIFY_SAFETY,
                    temperature=0.0,
                    max_output_tokens=1024,
                ),
            )
        except errors.APIError as exc:
            # 429 lands here and is ORDINARY on a 5-RPM free tier, not
            # exceptional. Deliberately not retried: the API's own retry-after
            # is tens of seconds, and a caller mid-sentence will not wait.
            # "Say that again" costs one turn; a 44-second silence loses them.
            log.warning("classification call failed: %s", exc)
            return None

        parsed = response.parsed
        if parsed is None:
            # `.parsed` is populated by the SDK when a schema is set, but a
            # truncated or malformed body leaves it empty. Fall back to the
            # raw text rather than treating a parser quirk as a refusal.
            try:
                parsed = json.loads(response.text or "")
            except (ValueError, TypeError):
                log.warning("model returned a body that was not valid JSON")
                return None

        if not isinstance(parsed, dict):
            log.warning("model returned %s, expected an object", type(parsed).__name__)
            return None
        return parsed

    # -- The phrasing call: say this, in this language -----------------------

    def say(
        self,
        *,
        system_instruction: str,
        instruction: str,
        max_output_tokens: int = 512,
    ) -> str | None:
        """Render one short utterance. Returns None so callers can fall back.

        There is no structured output here on purpose: the machine has already
        decided WHAT is said, and the only open question is how to word it in
        the applicant's language.
        """
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=instruction,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    thinking_config=PHRASING_THINKING,
                    temperature=0.3,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except errors.APIError as exc:
            log.warning("phrasing call failed: %s", exc)
            return None

        text = (response.text or "").strip()
        return text or None
