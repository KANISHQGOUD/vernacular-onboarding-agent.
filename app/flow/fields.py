"""Whether what we heard could possibly be what they said.

WHY THIS EXISTS
---------------
Speech recognition does not fail loudly. It returns a plausible sentence with
no error, and until now `save_applicant_field` accepted whatever came back:

    session.applicant[field_name] = value

Any string was a mobile number. Any string was a date of birth. Worse, the
collected `date_of_birth` was never parsed into a date anywhere in the system
-- `Conversation._screen` passes a hardcoded one to eligibility -- so a
recogniser hearing "twenty ninety" instead of "nineteen ninety" produced a
birth year in the future and nothing noticed.

The provider we use reports no transcription confidence (see
`app/speech/sarvam.py`), so `hearing.py`'s floors are inert against it. These
rules are the replacement, and they are better than a confidence score for
this job: a number tells you the recogniser was unsure, while a rule tells you
the answer is impossible.

RULES AS DATA, LIKE ELIGIBILITY
------------------------------
Same shape as `app/flow/eligibility.py`, for the same reasons: you can print
the ruleset, diff it across versions, and hand the descriptions to a
translator -- none of which you can do with a function full of if-statements.
And like eligibility, nothing here reads a clock. `as_of` is passed in, so a
rejection is reproducible.

THE RULES ARE WRONG ABOUT SOMEBODY
----------------------------------
Every rule below is a guess about what a real answer looks like, and each will
be wrong about a real person: an unusual name, a number that does not fit the
Indian mobile shape. That is why `actions.save_applicant_field` gives up after
two attempts and stores the value anyway with a flag, rather than refusing
forever. A flow whose entire purpose is not turning people away must not
invent a new way to turn them away.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Callable

from app.flow.eligibility import age_on
from app.speech.wer import DIGIT_MAP

FIELD_RULESET_VERSION = "2026-09-02.v1"

# How many times to re-ask before accepting the answer anyway. Three, not
# because three is special, but because two re-asks is where a person stops
# believing the machine can hear them -- and past that point the rule is more
# likely to be wrong than the applicant is. Same shape as MAX_KYC_ATTEMPTS:
# a budget, not a wall.
MAX_FIELD_ATTEMPTS = 3

# Indian mobile numbers are ten digits beginning 6-9. Older series (2-5) were
# landline and are not issued to mobiles.
MOBILE_DIGITS = 10
MOBILE_FIRST_DIGITS = frozenset("6789")

# The age rule eligibility already enforces. Repeated here as an upper bound
# too, because a date parsed from speech can land centuries out and "aged 900"
# is a transcription failure rather than an eligibility question.
MIN_AGE = 18
MAX_AGE = 120


class FieldProblem(str):
    """Why a value could not be used, in words a caller could be told.

    A string subclass so it can be logged, compared and put straight into a
    speech act's data without unwrapping.
    """


def digits_only(value: str) -> str:
    """Just the digits, with Devanagari folded to ASCII.

    `९८१२३४५६७८` and `9812345678` are the same phone number, and a caller
    reading digits aloud may be transcribed either way -- Sarvam normalises
    numbers, which is helpful right up until a naive `str.isdigit()` check
    rejects the result. Reuses the fold from `app/speech/wer.py` rather than
    repeating the table.
    """
    return "".join(ch for ch in value.translate(DIGIT_MAP) if ch in "0123456789")


# `1990-1-1` -- a real date that `fromisoformat` refuses over zero padding.
_LOOSE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")

# A bare year, and the most important shape after ISO. See `as_date`.
_YEAR_ONLY = re.compile(r"(\d{4})")


def as_date(value: str) -> date | None:
    """ISO first, because that is what the extractor produces.

    The classifier returns dates already structured -- a date spoken entirely
    in Hindi words comes back as `1990-01-01` with no date parser involved --
    so ISO is the expected shape, and the three fallbacks below are for when
    extraction half-worked rather than for parsing free text.

    DIGITS ARE FOLDED FIRST
    -----------------------
    `१९९०-०१-०१` is the same date as `1990-01-01`, and Sarvam transcribing
    spoken Hindi numerals may return either. This is the same fold
    `digits_only` applies to mobile numbers, and leaving it out here meant
    a date in Devanagari digits was rejected as "not a date" -- exactly the
    failure this module's own docstring warns about for phones.

    A BARE YEAR IS A DATE
    ---------------------
    Ask an older applicant their date of birth and "1990" is a common and
    honest answer: a large share of Indian records carry 1 January precisely
    because the exact day was never recorded. Refusing it three times and then
    storing it unvalidated turns that into a failure, on the population this
    product exists for. So a year alone becomes 1 January of that year, which
    is the convention those records already use.

    The inference is not silent -- `Actions.save_applicant_field` notes it, so
    "was this day real or assumed" stays answerable afterwards.
    """
    if not isinstance(value, str):
        return None

    text = value.strip().translate(DIGIT_MAP)

    try:
        return date.fromisoformat(text)
    except ValueError:
        pass

    loose = _LOOSE_ISO.fullmatch(text)
    if loose:
        try:
            return date(int(loose[1]), int(loose[2]), int(loose[3]))
        except ValueError:
            return None

    year = _YEAR_ONLY.fullmatch(text)
    if year:
        try:
            return date(int(year[1]), 1, 1)
        except ValueError:
            return None

    return None


def is_year_only(value: str) -> bool:
    """Whether the day and month in a parsed date were assumed rather than said.

    Kept beside `as_date` so the two cannot disagree about what a bare year
    looks like.
    """
    return bool(
        isinstance(value, str) and _YEAR_ONLY.fullmatch(value.strip().translate(DIGIT_MAP))
    )


# -- The rules ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldRule:
    """One check, as data. `check` returns a problem or None."""

    field: str
    description: str
    check: Callable[[str, date], FieldProblem | None]


def _check_mobile(value: str, as_of: date) -> FieldProblem | None:
    digits = digits_only(value)
    if len(digits) != MOBILE_DIGITS:
        return FieldProblem(
            f"that came through as {len(digits)} digits rather than {MOBILE_DIGITS}"
        )
    if digits[0] not in MOBILE_FIRST_DIGITS:
        return FieldProblem("that does not look like a mobile number")
    return None


def _check_date_of_birth(value: str, as_of: date) -> FieldProblem | None:
    born = as_date(value)
    if born is None:
        return FieldProblem("that did not come through as a date")
    if born > as_of:
        return FieldProblem("that date is in the future")

    age = age_on(born, as_of)
    if age < MIN_AGE:
        return FieldProblem(f"that would make the applicant {age}")
    if age > MAX_AGE:
        return FieldProblem(f"that would make the applicant {age}")
    return None


def _check_name(value: str, as_of: date) -> FieldProblem | None:
    text = value.strip()
    if len(text) < 2:
        return FieldProblem("that came through as almost nothing")
    if not any(ch.isalpha() for ch in text):
        return FieldProblem("that came through with no letters in it")
    return None


def _check_address(value: str, as_of: date) -> FieldProblem | None:
    # Short enough to catch "haan" landing in the address field, long enough
    # not to reject a genuinely terse rural address.
    if len(value.strip()) < 10:
        return FieldProblem("that came through too short to be an address")
    return None


def _check_occupation(value: str, as_of: date) -> FieldProblem | None:
    text = value.strip()
    if len(text) < 2 or not any(ch.isalpha() for ch in text):
        return FieldProblem("that came through as almost nothing")
    return None


RULES: tuple[FieldRule, ...] = (
    FieldRule("mobile", "Ten digits, beginning 6 to 9.", _check_mobile),
    FieldRule("date_of_birth", "A real past date, aged 18 to 120.", _check_date_of_birth),
    FieldRule("full_name", "At least two characters, containing a letter.", _check_name),
    FieldRule("address", "At least ten characters.", _check_address),
    FieldRule("occupation", "At least two characters, containing a letter.", _check_occupation),
)

_BY_FIELD = {rule.field: rule for rule in RULES}

# The order they are asked in. Easiest first: a name is one word and builds
# some confidence that the machine can hear you at all, before a ten-digit
# number that is the hardest thing in the flow to get right by voice.
FIELD_ORDER: tuple[str, ...] = (
    "full_name",
    "date_of_birth",
    "address",
    "occupation",
    "mobile",
)


def next_field(missing: frozenset[str] | set[str]) -> str | None:
    """Which field is being asked for now.

    Two callers need this -- the phrasing layer, to name the field, and the
    speech layer, to know what kind of answer to expect -- and they must
    agree. Scanning FIELD_ORDER in two places is how they stop agreeing.
    """
    return next((f for f in FIELD_ORDER if f in missing), None)


def validate(field: str, value: str, *, as_of: date) -> FieldProblem | None:
    """The problem with this value, or None if there is none.

    An unknown field passes. New fields should be addable without every one of
    them needing a rule on the same commit, and silently rejecting a field
    nobody wrote a rule for would be the worst of both.
    """
    rule = _BY_FIELD.get(field)
    if rule is None:
        return None
    if value is None:
        return FieldProblem("nothing came through")
    return rule.check(value, as_of)


def normalised(field: str, value: str) -> str:
    """What to store, once the value is accepted.

    Only the mobile is rewritten: it is the one field where the stored form
    matters downstream -- core banking is looked up by it, and an OTP is sent
    to it -- and where a caller reading digits aloud can produce spaces,
    dashes or Devanagari numerals for the same number.
    """
    if field == "mobile":
        return digits_only(value)
    return value.strip()
