"""Turning a decided speech act into a sentence someone can hear.

The machine has already chosen WHAT to say. This file only chooses HOW, and
in which language. It cannot introduce a subject the machine did not pick,
because a `SpeechAct` is the only input it accepts.

BUILT AROUND 5 REQUESTS PER MINUTE
----------------------------------
The free tier allows 5 model calls a minute. Two calls per conversational turn
would cap this agent at 2.5 turns a minute -- not a conversation. So phrasing
avoids the model wherever it honestly can:

  1. English needs no call at all. The fallback sentences ARE English, written
     by a human, and already correct. A model call to rewrite correct English
     into different correct English spends quota to gain nothing.

  2. Everything else is cached by (kind, language, data). The same speech act
     in the same language is the same sentence every time -- there is no reason
     to buy it twice.

  3. Static acts can be pre-warmed before the conversation starts, so the
     applicant never waits on a translation that was always predictable.

What is left is a handful of calls per language, once, ever. Classification --
which genuinely cannot be cached, because every caller says something new --
gets the remaining budget.

THE HALLUCINATION GUARD
-----------------------
`verbatim` fields must survive into the output character for character. A model
that renders "Rs 125.50" as "about a hundred and twenty-five rupees" has
misstated a bank balance. If a required value did not survive, the rendering is
discarded and the human-written fallback is spoken instead. Losing the
translation is a bad outcome; misreporting an account number is a worse one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm.client import LlmClient
from app.llm.speech_acts import SpeechAct, SpeechActKind

log = logging.getLogger(__name__)

# Languages the fallback text is already written in. No call needed.
NATIVE_LANGUAGES = frozenset({"en", "en-IN", "en-US", "en-GB"})

SYSTEM_INSTRUCTION = """\
You word one sentence for a bank's account-opening voice agent, to be read \
aloud to a caller.

You are given the exact meaning to convey and the language to say it in. Your \
only job is the wording.

Rules, in order of importance:

1. Say only what you were asked to convey. Never add an offer, a product, a \
suggestion, or a recommendation of any kind. If it was not in the instruction, \
it does not go in the sentence.
2. Reproduce every number, account number, amount and reference exactly as \
given. Do not round them, spell them out, convert currencies, or approximate.
3. Write it to be spoken, not read. Short sentences. No markdown, no bullet \
points, no emoji, no formatting of any kind.
4. Use plain, warm, everyday language. Many callers are opening their first \
bank account, may be elderly, and may not read. Avoid banking jargon.
5. Return only the sentence itself. No preamble, no quotes, no explanation.\
"""

# Acts with no `data` produce the same sentence forever, per language. These
# are the ones worth translating once at startup instead of mid-call.
STATIC_ACTS: frozenset[SpeechActKind] = frozenset(
    {
        SpeechActKind.ASK_LANGUAGE,
        SpeechActKind.EXPLAIN_CONSENT,
        SpeechActKind.EXPLAIN_KYC_HANDOFF,
        SpeechActKind.KYC_STILL_WAITING,
        SpeechActKind.KYC_RETRY,
        SpeechActKind.ASK_FOR_DEPOSIT,
        SpeechActKind.GUIDE_FIRST_PAYMENT,
        SpeechActKind.HANDING_TO_HUMAN,
        SpeechActKind.DID_NOT_UNDERSTAND,
    }
)

CacheKey = tuple[SpeechActKind, str, tuple[tuple[str, str], ...]]


def _cache_key(act: SpeechAct) -> CacheKey:
    return (act.kind, act.language, tuple(sorted(act.data.items())))


def build_instruction(act: SpeechAct) -> str:
    """What to convey, plus the values that must come through untouched."""
    lines = [f"Say this in {act.language}:", act.spec.intent]

    if act.data:
        lines.append("")
        lines.append("Details to include:")
        lines.extend(f"- {key}: {value}" for key, value in sorted(act.data.items()))

    if act.spec.verbatim:
        required = ", ".join(
            act.data[k] for k in act.spec.verbatim if k in act.data
        )
        if required:
            lines.append("")
            lines.append(
                f"These must appear exactly as written, unchanged: {required}"
            )

    return "\n".join(lines)


@dataclass
class Phraser:
    """Renders speech acts. Never raises; always returns something speakable."""

    client: LlmClient | None = None
    _cache: dict[CacheKey, str] = field(default_factory=dict, init=False)

    # -- Instrumentation, so the quota story is measurable rather than assumed.
    calls_made: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    calls_avoided_native: int = field(default=0, init=False)
    fallbacks_used: int = field(default=0, init=False)

    def say(self, act: SpeechAct) -> str:
        """The sentence to speak. Guaranteed non-empty."""
        # 1. English is already written. Spend nothing.
        if act.language in NATIVE_LANGUAGES:
            self.calls_avoided_native += 1
            return act.fallback_text()

        key = _cache_key(act)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        # 2. No client configured (tests, offline, missing key) -> fallback.
        if self.client is None:
            self.fallbacks_used += 1
            return act.fallback_text()

        self.calls_made += 1
        rendered = self.client.say(
            system_instruction=SYSTEM_INSTRUCTION,
            instruction=build_instruction(act),
        )

        # 3. Model unreachable, rate-limited, or empty -> fallback.
        if not rendered:
            self.fallbacks_used += 1
            return act.fallback_text()

        # 4. The hallucination guard. A dropped account number or a rounded
        #    amount means we throw the translation away and speak the English.
        missing = act.missing_verbatim(rendered)
        if missing:
            log.warning(
                "discarding %s rendering: %s did not survive verbatim",
                act.kind,
                ", ".join(missing),
            )
            self.fallbacks_used += 1
            return act.fallback_text()

        self._cache[key] = rendered
        return rendered

    def prewarm(self, language: str) -> int:
        """Translate the static acts once, before anyone is waiting.

        Returns how many calls it made. Run this at startup, not mid-call: on
        a 5-RPM budget the difference between paying now and paying while
        somebody holds the phone is the whole user experience.
        """
        if language in NATIVE_LANGUAGES or self.client is None:
            return 0

        before = self.calls_made
        for kind in sorted(STATIC_ACTS):
            self.say(SpeechAct(kind=kind, language=language))
        return self.calls_made - before

    def stats(self) -> dict[str, int]:
        return {
            "calls_made": self.calls_made,
            "cache_hits": self.cache_hits,
            "calls_avoided_native": self.calls_avoided_native,
            "fallbacks_used": self.fallbacks_used,
            "cached_entries": len(self._cache),
        }
