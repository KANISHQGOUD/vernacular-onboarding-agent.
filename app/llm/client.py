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
    gemini-3.5-flash      works. all thinking levels.
    latency               ~3.1-4.6s per classification, measured over a
                          full conversation. Earlier lower figures were taken
                          during a quiet period and did not hold up.

    QUOTA -- two separate ceilings, and the second one is the real constraint:
      GenerateRequestsPerMinutePerProjectPerModel-FreeTier   5 per minute
      GenerateRequestsPerDayPerProjectPerModel-FreeTier     20 per DAY

Twenty a day is roughly three complete conversations. Not a development
budget, and worth knowing before planning around it. Both quotas are scoped
per project PER MODEL, so switching GEMINI_MODEL moves to a fresh bucket --
which is the practical reason that setting is an environment variable rather
than a constant.

Those ceilings are why the design spends calls the way it does:

  - classification calls the model, because it genuinely cannot be cached --
    every caller says something new
  - phrasing mostly does NOT, because the same speech act in the same language
    is the same sentence every time (see phrasing.py). Measured over a full
    live conversation in English: eight speech acts, zero calls.

A 429 is therefore expected traffic on this tier, not an exception -- and it
has been observed in anger, alongside 503s. Both are caught and degraded like
any other failure, and never retried mid-call, because the retry-after is
measured in tens of seconds and there is a person waiting.

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

import httpx
from google import genai
from google.genai import errors, types

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """The model could not be reached, or refused to answer.

    Distinct from "the model answered and understood nothing", which is a
    reading of the caller. This is a statement about us.
    """

# Overridable without editing code, because model availability on the free tier
# moves faster than this repo does -- gemini-2.5-flash was the obvious default
# and is already retired.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Both jobs -- classifying an utterance and phrasing one sentence -- are short
# and latency-critical on a voice call. Neither needs deep reasoning; the
# reasoning already happened, deterministically, in the state machine.
# How long one model call may take before we give up and say so.
#
# The SDK's default is NO timeout. Measured on 2026-11-19 against the live
# API: under load, gemini-3.5-flash held a request open for about 130 seconds
# and then answered 503 "high demand". On a voice call that is two minutes of
# silence followed by an apology -- the caller has hung up long before it.
# Same trap as libpq's default; see CONNECT_TIMEOUT_SECONDS in postgres.py.
#
# 10s, and it cannot be lower. The SDK forwards this to Google as a server
# deadline, and the API rejects anything under 10s with 400 INVALID_ARGUMENT
# ("Minimum allowed deadline is 10s") -- found by setting 8s, which passed
# every offline test and broke every live call. That is still about three
# times the slowest healthy median measured (3.4s). A turn makes at most two
# calls -- classify, then phrase -- so a caller's worst wait is ~20s rather
# than unbounded.
MODEL_TIMEOUT_MS = 10_000

# One attempt, stated rather than inherited. The SDK does not retry by
# default today, but a default is a thing that changes in a minor release, and
# a retry hidden inside the client would silently multiply MODEL_TIMEOUT_MS.
# Retrying belongs to the caller ("say that again"), never to the transport.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=MODEL_TIMEOUT_MS,
    retry_options=types.HttpRetryOptions(attempts=1),
)

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
    # Injectable so a test can point the client at a socket that never answers
    # and prove the timeout holds, without spending a request.
    http_options: types.HttpOptions = field(default_factory=lambda: _HTTP_OPTIONS)
    _client: genai.Client = field(init=False)

    def __post_init__(self) -> None:
        self._client = genai.Client(
            api_key=load_api_key(self.api_key), http_options=self.http_options
        )

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
        except (errors.APIError, httpx.TransportError) as exc:  # noqa: PERF203 - see below
            # TransportError covers the timeout above and a dropped connection.
            # Neither is an APIError, so before this they escaped as a generic
            # exception and the caller got "something went wrong" instead of
            # the honest "that's not you".
            #
            # 429 lands here and is ORDINARY on a 5-RPM free tier, not
            # exceptional. Deliberately not retried: the API's own retry-after
            # is tens of seconds, and a caller mid-sentence will not wait.
            # "Say that again" costs one turn; a 44-second silence loses them.
            #
            # RAISED, not returned as None. Returning None made a vendor
            # outage indistinguishable from a caller mumbling, and the agent
            # answered both with "sorry, I didn't catch that" -- telling
            # somebody to speak more clearly while the fault was entirely
            # ours. For an applicant who already expects machines not to
            # understand them, that is the exact humiliation this product
            # exists to remove, and it is unanswerable: no amount of speaking
            # clearly fixes a 503.
            log.warning("classification call failed: %s", exc)
            raise ModelUnavailableError(str(exc)) from exc

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
        except (errors.APIError, httpx.TransportError) as exc:
            log.warning("phrasing call failed: %s", exc)
            return None

        text = (response.text or "").strip()
        return text or None
