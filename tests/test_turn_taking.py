"""How long to wait before deciding somebody has finished speaking.

Offline and pure. The detector runs in the browser; this is the judgement it
is given, and the judgement is the part worth pinning.
"""

from __future__ import annotations

from app.flow.states import State
from app.flow.turn_taking import (
    DEFAULT_SILENCE_MS,
    INTERRUPTION_IS_MATERIAL,
    interruption_matters,
    patience_ms,
    policy_for,
    silence_ms,
)

# The window most voice agents use, tuned on people who are fluent, unhurried
# and used to talking to machines.
CONVENTIONAL_MS = 800


def test_every_state_has_a_window():
    for state in State:
        assert silence_ms(state) > 0
        assert patience_ms(state) > 0


def test_the_consent_gate_waits_longest():
    """The counterintuitive one, pinned so it cannot be "optimised" later.

    "हाँ" is one short word, so a short window looks obviously right. It is
    not, because the second half of a sentence can reverse the first:

        "हाँ..."  endpoint here and consent is recorded
        "...नहीं, रुकिए"  which is what they were actually saying

    Endpointing early does not clip a transcript. It records a consent that
    was being withdrawn while we stopped listening.
    """
    assert silence_ms(State.CONSENT) == max(silence_ms(s) for s in State)
    assert silence_ms(State.CONSENT) > DEFAULT_SILENCE_MS


def test_recall_gets_longer_than_answering():
    """Addresses and names arrive in pieces, with thinking in between. Being
    cut off there is being cut off saying your own name."""
    assert silence_ms(State.COLLECT_DATA) > silence_ms(State.LANGUAGE_SELECT)
    assert silence_ms(State.COLLECT_DATA) > DEFAULT_SILENCE_MS


def test_the_windows_are_deliberately_slower_than_convention():
    """Every state where somebody is composing an answer waits longer than the
    500-800ms most agents use. Those numbers are tuned on people this product
    is explicitly not for -- fluent, unhurried, and used to machines."""
    composing = (
        State.CONSENT,
        State.COLLECT_DATA,
        State.FUNDING,
        State.FIRST_TRANSACTION,
    )
    for state in composing:
        assert silence_ms(state) > CONVENTIONAL_MS

    # The exception, and it is an exception on purpose: one word, and getting
    # it wrong is loud and instantly fixable.
    assert silence_ms(State.LANGUAGE_SELECT) <= CONVENTIONAL_MS


def test_money_is_not_hurried():
    for state in (State.FUNDING, State.FIRST_TRANSACTION):
        assert silence_ms(state) >= 1_400


def test_interrupting_is_always_allowed():
    """A person must be able to interrupt a machine. An agent that finishes
    its sentence while somebody objects is the support line everyone hates."""
    for state in State:
        assert policy_for(state)["barge_in"] is True


def test_only_the_consent_notice_makes_an_interruption_material():
    """Allowed everywhere, worth writing down in one place.

    DPDP requires notice as well as consent, and "हाँ" four seconds into a
    twenty-second explanation is a weak claim to have received it. This does
    not block anything -- it makes the question answerable later.
    """
    assert interruption_matters(State.CONSENT)
    assert INTERRUPTION_IS_MATERIAL == {State.CONSENT}

    for state in State:
        if state is not State.CONSENT:
            assert not interruption_matters(state)


def test_the_policy_is_the_shape_the_browser_is_sent():
    policy = policy_for(State.CONSENT)
    assert set(policy) == {"silence_ms", "patience_ms", "barge_in", "report_interruption"}
    assert policy["silence_ms"] == silence_ms(State.CONSENT)
    assert policy["report_interruption"] is True

    assert policy_for(State.COLLECT_DATA)["report_interruption"] is False
