"""The map: every state, every legal transition, every tool binding.

This file is a SPECIFICATION, not logic. Nothing here executes a decision --
it only declares which decisions are legal. The engine that reads this map
lives in `machine.py` and is deliberately small and dumb.

Read this file to answer questions like:
  - "Can the agent open an account before KYC passes?"      -> look at TRANSITIONS
  - "Can the agent take someone's PAN during screening?"    -> look at STATE_TOOLS
  - "What happens if video KYC fails three times?"          -> look at TRANSITIONS

You should never have to read `machine.py` to answer those.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    # Guards are typed as taking a Session, but Session stores a State -- a
    # circular import. Deferring it to type-check time keeps the annotation
    # honest without a runtime cycle. `from __future__ import annotations`
    # above is what makes the deferral work.
    from app.session import Session


# How many times a user may retry video KYC before we stop and fetch a human.
# Real VCIP sessions fail for mundane reasons -- bad light, dropped 4G, the
# officer's queue timing out -- so a single failure must not end the journey.
MAX_KYC_ATTEMPTS = 3


class State(StrEnum):
    """Where the conversation is. Exactly one at a time, always persisted.

    StrEnum (not Enum) because these values go straight into the audit log and
    the session row. `State.KYC_PENDING` serialises as the string
    "KYC_PENDING" with no encoder, and reads back with no decoder.
    """

    START = "START"
    LANGUAGE_SELECT = "LANGUAGE_SELECT"
    CONSENT = "CONSENT"
    ELIGIBILITY = "ELIGIBILITY"
    COLLECT_DATA = "COLLECT_DATA"
    KYC_HANDOFF = "KYC_HANDOFF"
    KYC_PENDING = "KYC_PENDING"
    ACCOUNT_CREATE = "ACCOUNT_CREATE"
    FUNDING = "FUNDING"
    FIRST_TRANSACTION = "FIRST_TRANSACTION"

    # Terminal states. Nothing transitions out of these.
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    # The bank said no (DECLINED) and the person said no (CONSENT_REFUSED) are
    # different outcomes and are kept apart. So is "stopped answering"
    # (ABANDONED) -- see the note on the CONSENT_REFUSED transition below.
    CONSENT_REFUSED = "CONSENT_REFUSED"
    # They said stop. A decision, and an answer.
    #
    # Note what is NOT here: a state for going quiet. Silence is not an
    # outcome, so it does not get one -- see the note on lapsing below.
    ABANDONED = "ABANDONED"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"


TERMINAL_STATES: frozenset[State] = frozenset(
    {
        State.COMPLETED,
        State.DECLINED,
        State.CONSENT_REFUSED,
        State.ABANDONED,
        State.HUMAN_HANDOFF,
    }
)

# LAPSING IS NOT A TRANSITION
# ---------------------------
# Going quiet used to fire SESSION_TIMEOUT into a TIMED_OUT state. Both are
# gone, and the reason is worth keeping.
#
# Making that state resumable means going back to where the session was, and a
# transition targets a constant -- so "go back" becomes one guarded edge per
# state a timeout can happen in. Nine of them, and they are new entrances into
# ELIGIBILITY, COLLECT_DATA and ACCOUNT_CREATE, each of which is asserted to
# have exactly one. The guard would keep it behaviourally safe while turning
# "illegal is absent from the table" into "illegal unless a guard says
# otherwise". In a flow where one of those entrances opens a bank account,
# that is the wrong thing to trade for convenience.
#
# So a lapse does not move the session at all. Somebody who stops answering in
# COLLECT_DATA is still in COLLECT_DATA -- they are simply not talking -- and
# coming back needs no edge, because they never left. Zero new entrances, and
# all three regulatory walls stay structural.
#
# It also stops overstating what a timeout knows. The state machine cannot
# tell a person who gave up from one whose train went into a tunnel, and this
# product's callers are disproportionately the second. `Actions.record_lapse`
# writes the fact to the audit trail, where an operational question belongs,
# and `session.lapses` counts it -- which is more than the old state carried,
# because it survives the resume and says WHERE people go quiet.


class Event(StrEnum):
    """The only things that can ever happen. A closed vocabulary.

    This enum is the narrow gate the language model speaks through. The model
    never names a State -- it classifies an utterance into one of the
    USER-INTENT events below, and the map decides what that means. If the
    model returns anything outside this enum, or an event this state does not
    accept, the machine refuses and we re-prompt.
    """

    # -- User intent: produced by classifying an utterance (LLM in Phase 2) --
    LANGUAGE_CHOSEN = "LANGUAGE_CHOSEN"
    LANGUAGE_CHANGE_REQUESTED = "LANGUAGE_CHANGE_REQUESTED"
    CONSENT_GIVEN = "CONSENT_GIVEN"
    CONSENT_REFUSED = "CONSENT_REFUSED"
    DATA_PROVIDED = "DATA_PROVIDED"
    USER_WANTS_HUMAN = "USER_WANTS_HUMAN"
    # Answers to "I have written down X -- is that right?". Intents, because
    # they are readings of what the caller meant; a model may emit them, and
    # nothing about the world changes until one is acted on.
    FIELD_CONFIRMED = "FIELD_CONFIRMED"
    FIELD_CORRECTED = "FIELD_CORRECTED"
    USER_QUIT = "USER_QUIT"

    # -- Facts: produced by our own code or a tool result, never by the LLM --
    SESSION_STARTED = "SESSION_STARTED"
    ELIGIBILITY_CHECKED = "ELIGIBILITY_CHECKED"
    REQUIRED_DATA_COMPLETE = "REQUIRED_DATA_COMPLETE"
    KYC_DISPATCHED = "KYC_DISPATCHED"
    KYC_VERIFIED = "KYC_VERIFIED"
    KYC_FAILED = "KYC_FAILED"
    ACCOUNT_OPENED = "ACCOUNT_OPENED"
    ACCOUNT_CREATION_FAILED = "ACCOUNT_CREATION_FAILED"
    DEPOSIT_CONFIRMED = "DEPOSIT_CONFIRMED"
    PAYMENT_CONFIRMED = "PAYMENT_CONFIRMED"


class ToolName(StrEnum):
    """Every tool that exists in the entire system.

    Note what is absent: there is no offer_loan, no suggest_credit_card, no
    upsell_anything. The mis-selling risk is closed by the tool simply not
    existing, not by a prompt asking the model to behave. A test asserts this
    registry stays free of cross-sell tools.
    """

    SET_LANGUAGE = "set_language"
    RECORD_CONSENT = "record_consent"
    CHECK_ELIGIBILITY = "check_eligibility"
    SAVE_APPLICANT_FIELD = "save_applicant_field"
    START_VIDEO_KYC = "start_video_kyc"
    CHECK_KYC_STATUS = "check_kyc_status"
    OPEN_ACCOUNT = "open_account"
    INITIATE_DEPOSIT = "initiate_deposit"
    MAKE_FIRST_PAYMENT = "make_first_payment"
    ESCALATE_TO_HUMAN = "escalate_to_human"


# A guard is a QUESTION ABOUT THE SESSION, and it must be pure: no clock, no
# network, no I/O, no randomness. The same session must always give the same
# answer, or the audit log stops being replayable. Anything time-dependent
# (an applicant's age) takes an explicit as_of date instead of reading today().
Guard = Callable[["Session"], bool]


@dataclass(frozen=True, slots=True)
class Transition:
    """One legal edge in the graph.

    `to` is always a literal State -- never computed. That is what lets this
    table be read as a complete spec: if a target could be decided at runtime,
    the table would no longer tell you where you can end up.
    """

    to: State
    guard: Guard | None = None
    why: str = ""


# ---------------------------------------------------------------------------
# THE TRANSITION TABLE
# ---------------------------------------------------------------------------
# Keyed by (current_state, event). The value is an ORDERED tuple of candidates;
# the first one whose guard passes wins. A candidate with guard=None always
# passes, so it acts as a fallback and must be last -- anything after it would
# be dead code, and a test asserts that never happens.
#
# A pair that is absent from this dict is ILLEGAL. That is the whole point:
# the machine's refusal behaviour is defined by the complement of this table,
# which means the tests can enumerate every illegal move automatically.
# ---------------------------------------------------------------------------

TRANSITIONS: dict[tuple[State, Event], tuple[Transition, ...]] = {
    # -- START ---------------------------------------------------------------
    (State.START, Event.SESSION_STARTED): (
        Transition(State.LANGUAGE_SELECT, why="Nothing can happen before we know the language."),
    ),

    # -- LANGUAGE_SELECT -----------------------------------------------------
    (State.LANGUAGE_SELECT, Event.LANGUAGE_CHOSEN): (
        Transition(State.CONSENT, why="Language settled; now the DPDP notice, in that language."),
    ),

    # -- CONSENT -------------------------------------------------------------
    # DPDP Act 2023: we may not process personal data without notice + consent,
    # so this gate sits before any question that collects any.
    (State.CONSENT, Event.CONSENT_GIVEN): (
        Transition(State.ELIGIBILITY, why="Consent recorded; screening may begin."),
    ),
    # Its own terminal state, not ABANDONED. Three reasons, and the third is
    # the one that actually changes behaviour:
    #
    #   1. Measurement. The success metric is first-transaction completion, so
    #      every exit has to be attributable. Folded into ABANDONED, "people
    #      refuse the consent notice" is indistinguishable from "people give up
    #      mid-form" and from "people go quiet", and those need different
    #      fixes -- rewrite the notice, shorten the form, handle silence.
    #   2. Law. A refusal is a lawful stop with a decision behind it. Walking
    #      away is the absence of a decision. Only one of them is an answer.
    #   3. What we say next. ABANDONED is phrased as HANDING_TO_HUMAN, so
    #      routing a refusal here meant answering "do not process my data" with
    #      "let me get someone for you" -- escalating at exactly the moment
    #      somebody has said no. That is the bug this fixes.
    #
    # Nothing needs erasing on this edge: consent gates ELIGIBILITY and
    # COLLECT_DATA, so no personal data has been collected yet.
    (State.CONSENT, Event.CONSENT_REFUSED): (
        Transition(
            State.CONSENT_REFUSED,
            why="No consent, no processing. A clean, lawful stop.",
        ),
    ),

    # -- ELIGIBILITY ---------------------------------------------------------
    # The branch that decides whether a person proceeds. Note the LLM has no
    # say here: it cannot emit ELIGIBILITY_CHECKED, and the guard reads a
    # decision produced by versioned rules in eligibility.py.
    (State.ELIGIBILITY, Event.ELIGIBILITY_CHECKED): (
        Transition(
            State.COLLECT_DATA,
            guard=lambda s: s.eligibility_passed,
            why="All rules in the pinned ruleset passed.",
        ),
        Transition(
            State.DECLINED,
            why="At least one rule failed. The reason comes from the rule id, "
                "never from free text the model composed.",
        ),
    ),

    # -- COLLECT_DATA --------------------------------------------------------
    (State.COLLECT_DATA, Event.DATA_PROVIDED): (
        Transition(State.COLLECT_DATA, why="One field captured; stay here for the rest."),
    ),
    # Reading a value back and being told yes or no. Both stay in
    # COLLECT_DATA: confirming is not progress through the flow, it is the
    # same step finishing properly.
    #
    # Guarded on there being something pending, then an unconditional
    # fallback -- because every row here must be total. That rule exists to
    # keep `NoGuardMatchedError` structurally unreachable, and a first draft
    # of this pair leaned on that exception to reject a stray confirmation.
    # Wrong way round: a session must never be stranded by a guard, so the
    # map always answers and the conversation decides what it meant.
    #
    # These events are legal here on every turn, because `classifiable_events`
    # is per state and cannot see whether a value is waiting. So "yes" with
    # nothing pending is a real possibility, and it lands on the fallback,
    # which changes nothing.
    (State.COLLECT_DATA, Event.FIELD_CONFIRMED): (
        Transition(
            State.COLLECT_DATA,
            guard=lambda s: s.pending_field is not None,
            why="Read back and agreed; the value can be written down.",
        ),
        Transition(State.COLLECT_DATA, why="Nothing was waiting to be confirmed."),
    ),
    (State.COLLECT_DATA, Event.FIELD_CORRECTED): (
        Transition(
            State.COLLECT_DATA,
            guard=lambda s: s.pending_field is not None,
            why="Read back and denied; ask for it again.",
        ),
        Transition(State.COLLECT_DATA, why="Nothing was waiting to be corrected."),
    ),
    (State.COLLECT_DATA, Event.REQUIRED_DATA_COMPLETE): (
        Transition(State.KYC_HANDOFF, why="Everything the KYC provider needs is on file."),
    ),

    # -- KYC_HANDOFF ---------------------------------------------------------
    # The regulatory boundary. RBI requires video KYC to simulate real-time
    # face-to-face contact, and since Aug 2025 those sessions must detect
    # AI-generated faces. An AI agent therefore cannot perform this step -- it
    # can only prepare for it, hand off, and pick the thread back up after.
    (State.KYC_HANDOFF, Event.KYC_DISPATCHED): (
        Transition(State.KYC_PENDING, why="Control has left our process. We now wait."),
    ),

    # -- KYC_PENDING ---------------------------------------------------------
    # Suspension is a STATE, not a flag on some other state, because what is
    # legal here is genuinely different: no data collection, no account
    # opening, and -- see below -- no timeout.
    (State.KYC_PENDING, Event.KYC_VERIFIED): (
        Transition(State.ACCOUNT_CREATE, why="Identity established by a human-supervised process."),
    ),
    (State.KYC_PENDING, Event.KYC_FAILED): (
        Transition(
            State.KYC_HANDOFF,
            guard=lambda s: s.kyc_attempts < MAX_KYC_ATTEMPTS,
            why="Retries are budgeted; bad light is not a rejection.",
        ),
        Transition(
            State.HUMAN_HANDOFF,
            why="Budget spent. A person takes over rather than the loop continuing.",
        ),
    ),
    # Nothing times this state out. Nothing times any state out any more --
    # see the note on lapsing above -- but this is where that started. This
    # state can
    # legitimately last hours -- the user closes the app and finishes video KYC
    # at a branch in the evening. Timing out here would destroy exactly the
    # journey the handoff exists to protect.

    # -- ACCOUNT_CREATE ------------------------------------------------------
    (State.ACCOUNT_CREATE, Event.ACCOUNT_OPENED): (
        Transition(State.FUNDING, why="An account exists. It is not yet a success."),
    ),
    (State.ACCOUNT_CREATE, Event.ACCOUNT_CREATION_FAILED): (
        Transition(State.HUMAN_HANDOFF, why="Core banking rejected us; the agent cannot fix that."),
    ),

    # -- FUNDING -------------------------------------------------------------
    # This state and the next are why the map does not end at ACCOUNT_CREATE.
    # An open, empty, untouched account IS the problem this product exists to
    # solve, so the graph refuses to call that finished.
    (State.FUNDING, Event.DEPOSIT_CONFIRMED): (
        Transition(State.FIRST_TRANSACTION, why="Money is in. Now it has to actually get used."),
    ),

    # -- FIRST_TRANSACTION ---------------------------------------------------
    (State.FIRST_TRANSACTION, Event.PAYMENT_CONFIRMED): (
        Transition(State.COMPLETED, why="The success metric. Not 'account opened' -- money moved."),
    ),

    # -- Language switching --------------------------------------------------
    # Self-loops: change the language, stay exactly where you were. A speaker
    # who is struggling is MOST likely to ask for a different language deep in
    # the flow, so restricting this to the opening states would fail the very
    # users the product is for.
    #
    # These are written out one row at a time rather than generated in a loop.
    # Generating them would be four lines shorter and would stop this table
    # being something you can simply read.
    (State.START, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.START),),
    (State.CONSENT, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.CONSENT),),
    (State.ELIGIBILITY, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.ELIGIBILITY),),
    (State.COLLECT_DATA, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.COLLECT_DATA),),
    (State.KYC_HANDOFF, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.KYC_HANDOFF),),
    (State.KYC_PENDING, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.KYC_PENDING),),
    (State.ACCOUNT_CREATE, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.ACCOUNT_CREATE),),
    (State.FUNDING, Event.LANGUAGE_CHANGE_REQUESTED): (Transition(State.FUNDING),),
    (State.FIRST_TRANSACTION, Event.LANGUAGE_CHANGE_REQUESTED): (
        Transition(State.FIRST_TRANSACTION),
    ),
    # LANGUAGE_SELECT is absent on purpose: asking to change language while
    # choosing a language is just LANGUAGE_CHOSEN.

    # -- Escape hatches ------------------------------------------------------
    # Available from every non-terminal state. "Get me a person" must never be
    # a move the machine refuses -- SBI has 10,000 staff for exactly this, and
    # an agent that traps someone is worse than no agent.
    (State.START, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.LANGUAGE_SELECT, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.CONSENT, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.ELIGIBILITY, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.COLLECT_DATA, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.KYC_HANDOFF, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.KYC_PENDING, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.ACCOUNT_CREATE, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.FUNDING, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),
    (State.FIRST_TRANSACTION, Event.USER_WANTS_HUMAN): (Transition(State.HUMAN_HANDOFF),),

    (State.START, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.LANGUAGE_SELECT, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.CONSENT, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.ELIGIBILITY, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.COLLECT_DATA, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.KYC_HANDOFF, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.KYC_PENDING, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.ACCOUNT_CREATE, Event.USER_QUIT): (Transition(State.ABANDONED),),
    # Quitting at FUNDING or FIRST_TRANSACTION leaves an open, unused account:
    # the precise failure this product exists to prevent. Same target, but
    # these two are the ones worth counting.
    (State.FUNDING, Event.USER_QUIT): (Transition(State.ABANDONED),),
    (State.FIRST_TRANSACTION, Event.USER_QUIT): (Transition(State.ABANDONED),),

}


# ---------------------------------------------------------------------------
# TOOL BINDING
# ---------------------------------------------------------------------------
# What the agent is physically able to DO in each state, as opposed to what it
# is told to do. In Phase 2 only these schemas get sent to the model, so an
# unbound tool is not discouraged -- it is absent from the model's world. The
# engine re-checks membership before executing anything, because a prompt is
# not a trust boundary.
#
# Terminal states bind nothing at all. A finished session has no moves.
# ---------------------------------------------------------------------------

# Bound everywhere the conversation is still live. Neither can mis-sell
# anything: one changes which language the TTS speaks, the other fetches a
# human. Wide binding is correct for these, and the contrast with
# OPEN_ACCOUNT below is the whole lesson.
_ALWAYS = frozenset({ToolName.SET_LANGUAGE, ToolName.ESCALATE_TO_HUMAN})

STATE_TOOLS: dict[State, frozenset[ToolName]] = {
    State.START: _ALWAYS,
    State.LANGUAGE_SELECT: _ALWAYS,
    State.CONSENT: _ALWAYS | {ToolName.RECORD_CONSENT},
    # CHECK_ELIGIBILITY but NOT SAVE_APPLICANT_FIELD: screening runs on what
    # the applicant has already told us. If the agent could collect new
    # personal data here it would be collecting it before the eligibility
    # decision that justifies collecting it.
    State.ELIGIBILITY: _ALWAYS | {ToolName.CHECK_ELIGIBILITY},
    State.COLLECT_DATA: _ALWAYS | {ToolName.SAVE_APPLICANT_FIELD},
    State.KYC_HANDOFF: _ALWAYS | {ToolName.START_VIDEO_KYC},
    # Only status polling. No data collection, no account creation. Whatever
    # the user says during these hours, the machine's hands are tied.
    State.KYC_PENDING: _ALWAYS | {ToolName.CHECK_KYC_STATUS},
    # OPEN_ACCOUNT lives here and nowhere else, and this state is reachable
    # only via (KYC_PENDING, KYC_VERIFIED). No prompt wording can open an
    # account for someone whose identity was never established.
    State.ACCOUNT_CREATE: _ALWAYS | {ToolName.OPEN_ACCOUNT},
    State.FUNDING: _ALWAYS | {ToolName.INITIATE_DEPOSIT},
    State.FIRST_TRANSACTION: _ALWAYS | {ToolName.MAKE_FIRST_PAYMENT},

    State.COMPLETED: frozenset(),
    State.DECLINED: frozenset(),
    State.CONSENT_REFUSED: frozenset(),
    State.ABANDONED: frozenset(),
    State.HUMAN_HANDOFF: frozenset(),
}
