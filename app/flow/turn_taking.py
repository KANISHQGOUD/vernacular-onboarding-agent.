"""How long a pause has to be before we decide someone has finished.

The detector runs in the browser, because that is where the microphone is.
This is the part that does not: how patient to be, which is a judgement about
the conversation rather than about audio, and the machine owns those.

Same split as `hearing.py`. The client measures, the machine decides.

WHY A FIXED THRESHOLD IS THE WRONG DEFAULT
------------------------------------------
Most voice agents endpoint at 500-800ms of silence, which is tuned on people
who are fluent, unhurried, and used to talking to machines. This product is
explicitly for people who are none of those things. Someone recalling their
own address pauses. Someone unsure what a machine wants pauses. Someone
elderly pauses. Cutting them off mid-sentence and answering the fragment is
the exact experience of being rushed through a form by a stranger, which is
what this is supposed to replace.

So the windows here are longer than is conventional, and longest where people
are recalling something rather than answering.

THE CONSENT GATE IS THE LONGEST, WHICH IS NOT OBVIOUS
-----------------------------------------------------
"हाँ" is one short word, so the tempting setting is a short window. It is
wrong, because the second half of a sentence can reverse the first:

    "हाँ..."            (endpoint here and consent is recorded)
    "...नहीं, रुकिए"     (what they were actually saying)

Endpointing early there does not produce a slightly clipped transcript. It
produces a lawful-looking record of a consent that was being withdrawn as we
stopped listening. Waiting an extra second costs nothing; not waiting costs
the one thing this flow cannot get wrong.

BARGE-IN IS ALWAYS ALLOWED
--------------------------
A person must be able to interrupt a machine. An agent that finishes its
sentence while somebody is objecting is the failure everyone has met on a
support line.

But interrupting the CONSENT explanation is worth writing down: DPDP requires
notice *and* consent, and someone who says "हाँ" four seconds into a
twenty-second notice has arguably not received it. This module does not decide
what to do about that -- it flags the states where an interruption is
material, and `actions.record_interruption` puts it in the trail.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from app.flow.states import State

# Longer than the 500-800ms most agents use. See the module docstring: the
# conventional numbers are tuned on people this product is not for.
DEFAULT_SILENCE_MS = 1000

# How long to keep listening when nobody has said anything at all. Generous,
# because "I did not hear a question I understood" and "I am thinking" look
# identical from here, and the recovery for the second is to wait.
DEFAULT_PATIENCE_MS = 12_000

SILENCE_MS: Mapping[State, int] = MappingProxyType(
    {
        # The longest, for the reason in the docstring: a sentence can reverse
        # itself, and this is the one place that is unrecoverable.
        State.CONSENT: 1_800,
        # Recall, not answering. Addresses and names arrive in pieces with
        # thinking in between, and this is where being cut off is most
        # insulting -- it is somebody's own name.
        State.COLLECT_DATA: 1_600,
        # Numbers, said carefully, often digit by digit.
        State.FUNDING: 1_500,
        State.FIRST_TRANSACTION: 1_500,
        # One word, and getting it wrong is loud and immediately fixable --
        # the next sentence arrives in a language they cannot read and they
        # say so. Snappy is fine here.
        State.LANGUAGE_SELECT: 800,
    }
)

# States where interrupting what the agent was saying is worth recording.
# Not a prohibition -- see the docstring -- a note in the trail.
INTERRUPTION_IS_MATERIAL: frozenset[State] = frozenset({State.CONSENT})


def silence_ms(state: State) -> int:
    """How much quiet ends their turn, in this state."""
    return SILENCE_MS.get(state, DEFAULT_SILENCE_MS)


def patience_ms(state: State) -> int:
    """How long to wait for them to begin at all."""
    return DEFAULT_PATIENCE_MS


def interruption_matters(state: State) -> bool:
    return state in INTERRUPTION_IS_MATERIAL


def policy_for(state: State) -> dict[str, object]:
    """The whole policy, as the shape the browser is sent.

    Sent per turn rather than configured once, because it changes with the
    state and the page must not have to know why.
    """
    return {
        "silence_ms": silence_ms(state),
        "patience_ms": patience_ms(state),
        # Always. A person must be able to interrupt a machine.
        "barge_in": True,
        "report_interruption": interruption_matters(state),
    }
