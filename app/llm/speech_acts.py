"""What the machine is allowed to ask to be said.

The claim this project rests on is that the state machine decides and the
model only phrases. `intents.py` enforces half of that -- the model may not
assert facts. This file enforces the other half: the model may not choose the
subject.

A `SpeechAct` is the machine saying "tell them their account is open, the
number is 00012345678". The model's entire freedom is the wording and the
language. It does not get to decide that now is a good moment to mention a
loan, because there is no speech act for mentioning a loan, and a speech act
is the only thing the phrasing layer will render.

THREE PROPERTIES WORTH NOTICING
-------------------------------
`fallback` is not optional. Every act ships a pre-written sentence used when
the model is unreachable. An agent that goes silent because an API returned
503 has hung up on someone halfway through opening a bank account.

`verbatim` names the data keys that must survive into the output unchanged --
account numbers, amounts, references. A model that rounds Rs 125.50 to "about
a hundred and twenty-five rupees" has misreported a bank balance. A test
asserts these appear character-for-character.

`data` values are strings, already formatted by the machine. Money arrives
here as "Rs 125.50" from `format_rupees`, never as a number for the model to
render -- because rendering is exactly where a model would improvise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class SpeechActKind(StrEnum):
    """The complete set of things this agent can ever say.

    Note what is missing: nothing here offers a product, suggests an upgrade,
    or recommends anything. Same argument as the tool registry -- the risk is
    closed by absence, not by an instruction asking the model to behave.
    """

    ASK_LANGUAGE = "ASK_LANGUAGE"
    EXPLAIN_CONSENT = "EXPLAIN_CONSENT"
    ASK_FIELD = "ASK_FIELD"
    ANNOUNCE_DECLINE = "ANNOUNCE_DECLINE"
    EXPLAIN_KYC_HANDOFF = "EXPLAIN_KYC_HANDOFF"
    KYC_STILL_WAITING = "KYC_STILL_WAITING"
    KYC_RETRY = "KYC_RETRY"
    ANNOUNCE_ACCOUNT_OPENED = "ANNOUNCE_ACCOUNT_OPENED"
    ASK_FOR_DEPOSIT = "ASK_FOR_DEPOSIT"
    GUIDE_FIRST_PAYMENT = "GUIDE_FIRST_PAYMENT"
    CONFIRM_COMPLETED = "CONFIRM_COMPLETED"
    HANDING_TO_HUMAN = "HANDING_TO_HUMAN"
    DID_NOT_UNDERSTAND = "DID_NOT_UNDERSTAND"


@dataclass(frozen=True, slots=True)
class SpeechActSpec:
    """How to say one thing, and what to say if the model cannot be reached."""

    kind: SpeechActKind
    intent: str
    fallback: str
    verbatim: tuple[str, ...] = ()


SPEECH_ACTS: Mapping[SpeechActKind, SpeechActSpec] = MappingProxyType(
    {
        spec.kind: spec
        for spec in (
            SpeechActSpec(
                kind=SpeechActKind.ASK_LANGUAGE,
                intent=(
                    "Greet them briefly and ask which language they would like to "
                    "continue in. Name two or three common options."
                ),
                fallback="Which language would you like to continue in?",
            ),
            SpeechActSpec(
                kind=SpeechActKind.EXPLAIN_CONSENT,
                intent=(
                    "Explain that to open the account you need to collect and use "
                    "some personal details, and ask whether that is alright. Be "
                    "plain. Do not minimise it and do not pressure them."
                ),
                fallback=(
                    "To open your account I need to collect some personal details "
                    "and use them for verification. Is that alright?"
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.ASK_FIELD,
                intent=(
                    "Ask for the one piece of information named in field_label. "
                    "One question only."
                ),
                fallback="Could you tell me your {field_label}?",
            ),
            SpeechActSpec(
                kind=SpeechActKind.ANNOUNCE_DECLINE,
                intent=(
                    "Tell them this account cannot be opened, and give the reason "
                    "in reason_text as the reason. Say what they can do instead if "
                    "next_step is given. Do not apologise repeatedly and do not "
                    "invent any reason beyond reason_text."
                ),
                fallback="I'm not able to open this account. {reason_text}",
            ),
            SpeechActSpec(
                kind=SpeechActKind.EXPLAIN_KYC_HANDOFF,
                intent=(
                    "Explain that the identity check is a short video call with a "
                    "bank officer, that it is a legal requirement, and that you "
                    "will pick up where you left off afterwards -- even if that is "
                    "hours later. Reassure them their progress is saved."
                ),
                fallback=(
                    "The next step is a short video call with a bank officer to "
                    "verify your identity. It's required by law. Your progress is "
                    "saved, so we can continue right here afterwards."
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.KYC_STILL_WAITING,
                intent=(
                    "Tell them the identity check has not finished yet and there "
                    "is nothing they need to do right now. Do not rush them."
                ),
                fallback=(
                    "Your identity check hasn't finished yet. There's nothing you "
                    "need to do — we'll continue as soon as it's done."
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.KYC_RETRY,
                intent=(
                    "Tell them the video call didn't go through and it is a common "
                    "problem, usually the network or the lighting -- not anything "
                    "they did wrong. Offer to try again."
                ),
                fallback=(
                    "That video call didn't go through. It usually just means a "
                    "weak network or poor lighting. Shall we try once more?"
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
                intent=(
                    "Tell them the account is open and give the account number in "
                    "account_number. Read the number out clearly. Then say the "
                    "next step is putting some money in."
                ),
                fallback=(
                    "Your account is open. The number is {account_number}. "
                    "The next step is adding some money to it."
                ),
                verbatim=("account_number",),
            ),
            SpeechActSpec(
                kind=SpeechActKind.ASK_FOR_DEPOSIT,
                intent=(
                    "Ask them to add some money to the account so it can be used. "
                    "Keep it short and practical."
                ),
                fallback="Let's add some money to the account so you can start using it.",
            ),
            SpeechActSpec(
                kind=SpeechActKind.GUIDE_FIRST_PAYMENT,
                intent=(
                    "Encourage them to make one small first payment so they have "
                    "actually used the account once. Suggest something ordinary "
                    "like paying a shop. Keep it warm and low-pressure."
                ),
                fallback=(
                    "Now let's make one small payment, so you've used the account "
                    "once. Even paying a shop a small amount is enough."
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.CONFIRM_COMPLETED,
                intent=(
                    "Confirm the payment of amount_text went through, and that "
                    "their account is now open and working. Congratulate them "
                    "briefly."
                ),
                fallback=(
                    "That payment of {amount_text} went through. Your account is "
                    "open and working — you're all set."
                ),
                verbatim=("amount_text",),
            ),
            SpeechActSpec(
                kind=SpeechActKind.HANDING_TO_HUMAN,
                intent=(
                    "Tell them you are passing them to a member of staff, and that "
                    "the person will already have their details so they will not "
                    "have to start over."
                ),
                fallback=(
                    "I'm passing you to a member of our staff. They'll have your "
                    "details already, so you won't need to start again."
                ),
            ),
            SpeechActSpec(
                kind=SpeechActKind.DID_NOT_UNDERSTAND,
                intent=(
                    "Say you did not catch that and ask them to say it again. "
                    "Put it on yourself, not on them. One short sentence."
                ),
                fallback="Sorry, I didn't catch that — could you say it again?",
            ),
        )
    }
)


@dataclass(frozen=True, slots=True)
class SpeechAct:
    """One thing to say, decided by the machine.

    `data` carries only values the machine supplied. There is no free-text
    field a caller could stuff a new instruction into, which matters because
    anything reaching the phrasing prompt is, by construction, trusted.
    """

    kind: SpeechActKind
    language: str = "en-IN"
    data: Mapping[str, str] = field(default_factory=dict)

    @property
    def spec(self) -> SpeechActSpec:
        return SPEECH_ACTS[self.kind]

    def fallback_text(self) -> str:
        """The pre-written sentence, used when the model is unreachable.

        Missing keys leave their placeholder rather than raising: a slightly
        awkward sentence is a far better failure than silence on a phone call.
        """
        try:
            return self.spec.fallback.format(**self.data)
        except (KeyError, IndexError):
            return self.spec.fallback

    def missing_verbatim(self, rendered: str) -> tuple[str, ...]:
        """Which required values did NOT survive into the rendered sentence.

        Empty tuple means the rendering is safe to speak.
        """
        return tuple(
            key
            for key in self.spec.verbatim
            if key in self.data and self.data[key] not in rendered
        )
