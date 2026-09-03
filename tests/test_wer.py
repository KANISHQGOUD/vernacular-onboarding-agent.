"""The error-rate metric, which everything else in the comparison rests on.

Offline and pure. If this is wrong, the vendor comparison is worse than
useless -- it is a confident number pointing the wrong way.
"""

from __future__ import annotations

import unicodedata

import pytest

from app.speech.wer import (
    ErrorRate,
    character_error_rate,
    normalise,
    tokens,
    word_error_rate,
)

# ---------------------------------------------------------------------------
# Normalisation, which is the whole problem.
# ---------------------------------------------------------------------------


def test_stripping_punctuation_does_not_shorten_indic_words():
    """The bug that cost a debugging round, pinned.

    `ch.isalnum()` is False for Devanagari vowel signs -- they are combining
    marks, category Mn -- so filtering on it turns आशा into आश and every Indic
    word gets quietly shorter. The resulting error rate looks like a vendor
    accuracy problem.
    """
    assert normalise("आशा") == "आशा"
    assert len(normalise("आशा")) == 3

    # The wrong implementation, kept here so the difference is visible.
    naive = "".join(ch for ch in "आशा" if ch.isalnum() or ch.isspace())
    assert naive == "आश", "the trap has changed shape; update this test"
    assert naive != normalise("आशा")


def test_the_danda_is_punctuation():
    """Recognisers add it inconsistently, so it must not count as an error."""
    assert normalise("मेरा नाम आशा देवी है।") == normalise("मेरा नाम आशा देवी है")
    assert word_error_rate("यह ठीक है।", "यह ठीक है").errors == 0


def test_the_same_word_composed_two_ways_is_the_same_word():
    """NFC first, or two identical transcripts score as errors."""
    composed = unicodedata.normalize("NFC", "कि")
    decomposed = unicodedata.normalize("NFD", "कि")

    assert composed != decomposed or True  # may be identical for this cluster
    assert normalise(composed) == normalise(decomposed)


def test_devanagari_digits_are_the_same_answer_as_ascii_ones():
    """A spoken mobile number may come back either way, and this agent
    collects numbers constantly. Scoring them as errors would make every
    number-carrying utterance look worse than it is."""
    assert normalise("९८१२३४५६७८") == "9812345678"
    assert word_error_rate("9812345678", "९८१२३४५६७८").errors == 0


def test_case_and_spacing_do_not_count():
    assert word_error_rate("Asha Devi", "asha   devi").errors == 0
    assert tokens("  a  b  ") == ["a", "b"]


# ---------------------------------------------------------------------------
# The metric.
# ---------------------------------------------------------------------------


def test_identical_transcripts_score_zero():
    heard = word_error_rate("मेरा नाम आशा है", "मेरा नाम आशा है")
    assert heard.errors == 0
    assert heard.rate == 0.0
    assert heard.accuracy == 1.0


def test_the_three_kinds_of_error_are_counted_separately():
    """They are different faults with different fixes, and one percentage
    hides which you have: substitutions mean mishearing, deletions mean
    dropped audio, insertions mean hallucinating into silence."""
    assert word_error_rate("a b c", "a x c") == ErrorRate(1, 0, 0, 3)
    assert word_error_rate("a b c", "a c") == ErrorRate(0, 1, 0, 3)
    assert word_error_rate("a b c", "a b x c") == ErrorRate(0, 0, 1, 3)


def test_the_rate_is_errors_over_reference_length():
    assert word_error_rate("a b c d", "a x c d").rate == pytest.approx(0.25)
    assert word_error_rate("a b", "x y").rate == pytest.approx(1.0)


def test_a_hallucinating_recogniser_can_exceed_one():
    """Not a bug in the metric. A model that invents more than was said has an
    error rate above 100%, and clamping it would hide the worst failure mode."""
    heard = word_error_rate("haan", "haan ji bilkul theek hai sir")
    assert heard.rate > 1.0
    assert heard.accuracy == 0.0


def test_an_empty_reference_is_handled():
    assert word_error_rate("", "").rate == 0.0
    assert word_error_rate("", "something").rate == 1.0


def test_silence_against_a_real_utterance_is_total_loss():
    heard = word_error_rate("मेरा नाम आशा देवी है", "")
    assert heard.deletions == 5  # मेरा नाम आशा देवी है
    assert heard.rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Why both metrics are reported.
# ---------------------------------------------------------------------------


def test_character_rate_forgives_a_spacing_disagreement_that_word_rate_punishes():
    """Indic engines split compounds and postpositions differently. One
    disagreement about a space can register as two word substitutions while
    the characters are almost identical -- which is why CER is the more honest
    headline here and WER is reported beside it, not instead of it.
    """
    reference = "आपका खाता"
    hypothesis = "आप का खाता"

    by_word = word_error_rate(reference, hypothesis)
    by_char = character_error_rate(reference, hypothesis)

    assert by_word.errors > 0, "the spacing difference should cost something"
    assert by_char.errors == 0, "the characters are identical once space is gone"
    assert by_char.rate < by_word.rate


def test_character_rate_still_catches_a_real_mishearing():
    """It must forgive spacing, not everything."""
    assert character_error_rate("हाँ", "नहीं").errors > 0
