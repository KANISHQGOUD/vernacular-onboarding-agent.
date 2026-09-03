"""Word and character error rate, for Indic text specifically.

WHY NOT A LIBRARY
-----------------
The metric is twenty lines. The normalisation is the whole problem, and every
off-the-shelf implementation makes assumptions that are wrong here.

WHAT NORMALISATION HAS TO GET RIGHT
-----------------------------------
**Combining marks.** The obvious way to strip punctuation is to keep what is
`isalnum()`. Doing that turns आशा into आश, because Devanagari vowel signs are
combining marks (category Mn) and `isalnum()` is False for them. It silently
shortens every word in every Indic script, and the resulting error rate looks
like a vendor problem. This was not hypothetical -- it cost a debugging round
in this repo. Punctuation is removed by *category*, never by keeping alnum.

**Composition.** क + ि and the precomposed form are the same word and
different code points. Everything is NFC-normalised before comparison, or two
identical transcripts score as an error.

**Digits.** A caller reading a mobile number may be transcribed as 9812345678
or ९८१२३४५६७८. Those are the same answer. Devanagari digits fold to ASCII,
because scoring them as errors would make every number-carrying utterance look
worse than it is -- and numbers are most of what this agent collects.

**Danda.** । and ॥ are punctuation, and recognisers add them inconsistently.
Removed with the rest.

WHY BOTH WER AND CER
--------------------
Word error rate assumes words are separated by spaces. That is a shakier
assumption in Indic text than in English -- compounds and postpositions get
split or joined differently by different engines, and a single disagreement
about spacing can register as two substitutions. Character error rate does not
care, so it is the more honest headline number here; WER is reported beside it
because it is what everyone expects to see.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Devanagari digits, in order, so index == value.
DEVANAGARI_DIGITS = "०१२३४५६७८९"
DIGIT_MAP = {ord(d): str(i) for i, d in enumerate(DEVANAGARI_DIGITS)}


def normalise(text: str, *, fold_digits: bool = True) -> str:
    """Make two transcripts comparable without damaging either.

    Order matters: compose first, then strip punctuation, then fold digits,
    then collapse whitespace. Stripping before composing can orphan a
    combining mark.
    """
    text = unicodedata.normalize("NFC", text)

    # By category, never by `isalnum()`. See the module docstring.
    text = "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("P")
    )

    if fold_digits:
        text = text.translate(DIGIT_MAP)

    return " ".join(text.casefold().split())


def tokens(text: str) -> list[str]:
    return normalise(text).split()


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """One comparison, with the breakdown rather than just the total.

    The three counts matter separately. A model that mostly *substitutes* is
    mishearing; one that mostly *deletes* is dropping audio, usually at the
    ends; one that *inserts* is hallucinating into silence. They are different
    faults with different fixes, and a single percentage hides which you have.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference unit. Can exceed 1.0 when a recogniser
        hallucinates more than was said, which is not a bug in the metric."""
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.rate)


def _levenshtein(reference: list, hypothesis: list) -> ErrorRate:
    """Standard edit distance, kept for its backtrace rather than its total."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitute
                    cost[i - 1][j],      # delete
                    cost[i][j - 1],      # insert
                )

    # Walk back to separate the three kinds. Ties are resolved towards
    # substitution, which is the conventional choice and keeps the counts
    # comparable with other tools.
    subs = dels = ins = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return ErrorRate(subs, dels, ins, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return _levenshtein(tokens(reference), tokens(hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """Spacing-insensitive, which matters more in Indic text than in English."""
    ref = list(normalise(reference).replace(" ", ""))
    hyp = list(normalise(hypothesis).replace(" ", ""))
    return _levenshtein(ref, hyp)
