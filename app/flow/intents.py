"""The seam the language model plugs into -- and the fence around it.

The model's entire job in this system is one narrow question: given what the
user just said, which of these few events is it? It picks from a closed list.
It does not name a state, does not decide what happens next, and -- the part
this file enforces -- does not get to assert facts about the world.

That last distinction is the important one, so it is worth being precise.

Event splits in two:

  USER-INTENT events  what the user MEANT. CONSENT_GIVEN, USER_QUIT,
                      LANGUAGE_CHOSEN. Only a language model can read these
                      out of speech, so this is exactly the work it should do.

  FACT events         what HAPPENED. KYC_VERIFIED, ACCOUNT_OPENED,
                      DEPOSIT_CONFIRMED. These come from a provider response
                      or our own code. A model must never emit one.

Collapse that distinction and the architecture is gone. If a classifier could
return KYC_VERIFIED, then someone saying "the video call went fine, I'm
verified" would move the session to ACCOUNT_CREATE -- and identity
verification would have been performed by a chat message. `validate_intent`
below is a single membership check, and it is the entire defence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.flow.machine import FlowError
from app.flow.states import TRANSITIONS, Event, State

# The only events a classifier may ever produce.
USER_INTENT_EVENTS: frozenset[Event] = frozenset(
    {
        Event.LANGUAGE_CHOSEN,
        Event.LANGUAGE_CHANGE_REQUESTED,
        Event.CONSENT_GIVEN,
        Event.CONSENT_REFUSED,
        Event.DATA_PROVIDED,
        Event.USER_WANTS_HUMAN,
        Event.USER_QUIT,
        Event.DEPOSIT_REQUESTED,
        Event.PAYMENT_REQUESTED,
        Event.FIELD_CONFIRMED,
        Event.FIELD_CORRECTED,
    }
)

# Everything else: produced by tool results or our own code, never by a model.
FACT_EVENTS: frozenset[Event] = frozenset(Event) - USER_INTENT_EVENTS


class UntrustedEventError(FlowError):
    """A classifier tried to emit a fact event."""

    def __init__(self, event: Event) -> None:
        self.event = event
        super().__init__(
            f"{event} is a fact about the world, not a reading of what the user meant. "
            f"It cannot come from a classifier."
        )


def classifiable_events(state: State) -> frozenset[Event]:
    """The closed vocabulary offered to the model for one turn.

    Two narrowings, intersected: the events a classifier may ever produce, and
    the events this particular state accepts. At KYC_PENDING that leaves a
    handful; at a terminal state it leaves none.

    In Phase 2 this list goes into the prompt as the allowed answers -- so the
    model is not asked "what should happen?" but "which of these four?".
    """
    legal_here = frozenset(event for (s, event) in TRANSITIONS if s is state)
    return USER_INTENT_EVENTS & legal_here


def validate_intent(event: Event, state: State) -> Event:
    """Gate every classifier output. Raises rather than returning a flag.

    Checked in this order because the two failures mean different things: a
    fact event is a privilege violation and should be investigated, while a
    merely out-of-place intent is a routine misunderstanding that the
    conversation layer handles by re-prompting.
    """
    if event not in USER_INTENT_EVENTS:
        raise UntrustedEventError(event)

    if event not in classifiable_events(state):
        from app.flow.machine import IllegalTransitionError

        raise IllegalTransitionError(state, event, classifiable_events(state))

    return event


class IntentClassifier(Protocol):
    """A language model sits behind this; Phase 1 uses the scripted fake below.

    Deliberately says nothing about WHICH model. Nothing downstream of this
    protocol knows or cares -- which is why swapping providers touches this
    file's implementations and nothing else in the state machine.

    Returning None means "I could not tell" -- an honest outcome that the
    conversation layer turns into a re-prompt. It is deliberately not an
    exception: not understanding someone is normal, especially across accents
    and dialects, and must not be handled as an error condition.
    """

    def classify(self, utterance: str, state: State) -> Event | None: ...


@dataclass
class ScriptedClassifier:
    """A fake with no model behind it, so Phase 1 runs with no API key.

    Because it satisfies the same protocol and passes through the same
    `validate_intent`, the tests written against it keep their meaning when a
    real model replaces it.
    """

    script: dict[str, Event] = field(default_factory=dict)

    def classify(self, utterance: str, state: State) -> Event | None:
        event = self.script.get(utterance.strip().lower())
        if event is None:
            return None
        return validate_intent(event, state)
