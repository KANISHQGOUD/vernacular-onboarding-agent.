"""How sure we have to be before acting on what we heard.

The speech layer reports a number. This decides whether the number is good
enough -- and the answer depends on the state, which is why this lives in
`app/flow/` and takes a bare float rather than importing anything from
`app/speech/`. The seam stays a seam; the machine keeps the judgement.

THE FLOOR IS NOT "HOW IMPORTANT IS THIS FIELD"
----------------------------------------------
The tempting rule is to demand most certainty where the data matters most, and
it is wrong. What matters is the cost of acting on a mishearing, which is a
different question:

    A misheard NAME is caught later. It gets read back, it has to survive KYC,
    and the applicant is sitting right there to correct it. Cheap and visible.

    A misheard CONSENT is caught by nobody. "Nahi" heard as "haan" produces a
    lawful-looking record of a consent that was never given, and every
    subsequent step -- the screening, the data collection, the account -- is
    then unlawful processing that looks perfectly normal in the audit trail.

So COLLECT_DATA gets a LOW floor and CONSENT gets a high one, which reads
backwards until you notice that a high floor on names would punish exactly the
people this product exists for. Unusual names, regional pronunciations and
heavy accents are where recognisers lose confidence, and a strict floor there
means an applicant repeating their own name six times to a machine that keeps
saying it did not catch it. That is the failure mode this project is supposed
to remove, not reproduce.

SHORT ANSWERS ARE THE DANGEROUS ONES
------------------------------------
The yes/no gates carry both properties at once: "haan" and "nahi" are about
300 milliseconds of audio with few acoustic cues to separate them, and they
are the utterances with irreversible consequences. High floors there are not
caution, they are the whole reason this file exists.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from app.flow.fields import next_field
from app.flow.states import State
from app.speech.provider import Expectation

# Applied where a state has no entry of its own. Deliberately middling: low
# enough not to block ordinary speech, high enough to catch a recogniser that
# is guessing.
DEFAULT_FLOOR = 0.5

CONFIDENCE_FLOOR: Mapping[State, float] = MappingProxyType(
    {
        # Irreversible and invisible if wrong. See the module docstring.
        State.CONSENT: 0.85,
        # Money. Same argument, and the amount is spoken aloud as a number,
        # which is the other thing recognisers routinely mangle.
        State.FUNDING: 0.80,
        State.FIRST_TRANSACTION: 0.80,
        # Wrong language is bad but loud: the very next sentence arrives in a
        # language the caller does not read, and they say so. Recoverable.
        State.LANGUAGE_SELECT: 0.60,
        # Names, addresses, occupations -- exactly where recognisers lose
        # confidence on the accents this product is for, and exactly where a
        # mistake gets read back and corrected. Low on purpose.
        State.COLLECT_DATA: 0.35,
    }
)


# Where the answer is a proper noun rather than a sentence. People give
# names, streets and occupations in English constantly while speaking Hindi --
# "मेरा नाम Steve है" is not unusual, it is ordinary -- and a recogniser told
# to expect pure Hindi transliterates it into Devanagari, or worse, guesses.
#
# Measured on this project's corpus: code-mix mode kept `account` as
# `account`, and spelled आशा देवी राठौड़ correctly where the default produced
# राठौर. Better on both halves of the problem, so it is not a trade.
# Within COLLECT_DATA it depends on the FIELD, not the state. Measured:
# code-mix mode preserved `account` and spelled राठौड़ correctly, but the
# default mode's normalisation is what turned spoken words into `1 जनवरी 1990`
# and `₹500` -- which is exactly what the extractor downstream wants. So a
# name wants code-mix and a date does not, and one setting for the whole state
# has to be wrong about one of them.
CODE_MIXED_FIELDS: frozenset[str] = frozenset({"full_name", "address", "occupation"})

# Read back before being written down. Three, not five: the ones that are hard
# to fix afterwards. The mobile receives the OTP, the date of birth drives the
# age rule, and the name goes on the account -- an address or an occupation is
# re-askable later by somebody at a branch.
#
# This is also the practical answer to Sarvam reporting no confidence. A
# floor compares a number the provider does not give us; a read-back asks the
# only party who actually knows.
CONFIRMED_FIELDS: frozenset[str] = frozenset(
    {"mobile", "date_of_birth", "full_name"}
)


def needs_confirming(field: str) -> bool:
    return field in CONFIRMED_FIELDS


# How many denied read-backs before changing tactic, and before giving up.
# Two is where a person stops believing the machine can hear them; asking a
# third time in the same way is asking them to fail again.
#
#   after 1   ask again, normally
#   after 2   ask them to spell it -- letter by letter is a different task,
#             not a louder version of the one that failed
#   after 3   a person takes over. Same shape as MAX_KYC_ATTEMPTS: a budget,
#             and the end of it is a handoff rather than a wall.
SPELL_AFTER_CORRECTIONS = 2
ESCALATE_AFTER_CORRECTIONS = 3


def next_tactic(corrections: int) -> str:
    """What to do after this many denied read-backs of one field."""
    if corrections >= ESCALATE_AFTER_CORRECTIONS:
        return "escalate"
    if corrections >= SPELL_AFTER_CORRECTIONS:
        return "spell"
    return "ask"
def expectation_for(state: State, field: str | None = None) -> Expectation:
    """What kind of speech to tell the recogniser to expect.

    Deliberately NOT "detect the language". Letting the vendor auto-detect
    flips the output script entirely -- the same applicant comes back as
    आशा देवी राठौड़ or as Ashadevi Rathod depending on which way it guessed,
    and a bank record whose script depends on a coin toss is worse than one
    that is occasionally misspelled. The conversation's language stays the
    language; only the expectation of English *inside* it changes.
    """
    if state is not State.COLLECT_DATA:
        return Expectation.PLAIN
    # No field named means nothing specific is being asked for yet.
    return Expectation.CODE_MIXED if field in CODE_MIXED_FIELDS else Expectation.PLAIN


def floor_for(state: State) -> float:
    return CONFIDENCE_FLOOR.get(state, DEFAULT_FLOOR)


def is_confident_enough(confidence: float | None, state: State) -> bool:
    """Should we act on this utterance, or ask them to say it again?

    `None` means the provider does not report confidence, and it passes. That
    is a real decision with a real cost, so it is stated rather than left to
    fall out of the comparison:

    Rejecting it would make any provider that stays quiet unusable, including
    ones whose accuracy is fine. Accepting it means a silent provider disables
    this protection entirely -- so "does it report confidence" is a selection
    criterion for a real deployment, not a nice-to-have. The browser's Web
    Speech API does report it. A provider that does not should be treated as
    one you cannot run the consent gate on.
    """
    if confidence is None:
        return True
    return confidence >= floor_for(state)
