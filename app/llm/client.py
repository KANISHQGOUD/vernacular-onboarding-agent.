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

import logging
import os
from dataclasses import dataclass, field

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

    # -- The classifier's call: one value from a closed list ----------------

    def choose_one(
        self,
        *,
        system_instruction: str,
        utterance: str,
        options: list[str],
    ) -> str | None:
        """Return exactly one member of `options`, or None.

        `options` becomes an `enum` on the response schema, so the constraint
        is enforced by the API rather than requested in the prompt. The model
        is not asked what should happen -- it is handed four strings and asked
        which one this was. That narrowing is the whole reason a language model
        is safe to have in a banking flow at all.

        The caller still re-checks the answer through `validate_intent`. The
        schema is the wall; the re-check is the proof the wall held.
        """
        if not options:
            return None

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=utterance,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="text/x.enum",
                    response_schema=types.Schema(
                        type=types.Type.STRING, enum=list(options)
                    ),
                    thinking_config=CLASSIFY_THINKING,
                    safety_settings=_CLASSIFY_SAFETY,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
        except errors.APIError as exc:
            # 429 lands here and is ORDINARY on a 5-RPM free tier, not
            # exceptional. Deliberately not retried: the API's own retry-after
            # is tens of seconds, and a caller mid-sentence will not wait.
            # "Say that again" costs one turn; a 44-second silence loses them.
            log.warning("classification call failed: %s", exc)
            return None

        answer = (response.text or "").strip()

        # Belt and braces. The schema should make this impossible; if it ever
        # is not, an out-of-vocabulary answer must be dropped rather than
        # passed along as though the model had authority to invent one.
        if answer not in options:
            log.warning("model returned %r, not in the offered set", answer)
            return None
        return answer

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
