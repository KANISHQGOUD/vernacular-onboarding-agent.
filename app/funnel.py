"""Where people stop, computed from the audit trail.

WHY THIS HAS TO EXIST
---------------------
The whole project is justified by one number -- first-transaction completion
rate -- and until now nothing computed it. Every design argument here appeals
to a metric the code could not report, which is a comfortable position to
argue from and a bad one to be in.

It is also what blocks the user study. Ten people can sit down with this
tomorrow and, without a funnel, the result is ten anecdotes. "Six of ten
stopped at the KYC handoff" is a finding; "it seemed to go alright" is not.

READS THE TRAIL, NOT THE SESSIONS
---------------------------------
`sessions.state` says where somebody is now. That answers nothing about the
journey: a session sitting in ABANDONED does not say whether they gave up
before consent or after their account was open, and those are opposite
problems with opposite fixes.

The audit trail has the whole path, and it is append-only, so the funnel
cannot be quietly improved by editing history. That property was built for
regulators; it turns out to be exactly what makes a drop-off number
trustworthy.

PURE, SO IT CAN BE TESTED WITHOUT A DATABASE
--------------------------------------------
Everything here works on lists of `AuditEvent`. `scripts/funnel.py` does the
fetching. That split is why the funnel logic can be exercised against
synthetic journeys -- including the awkward ones, like somebody who lapsed
four times or who reached FUNDING and quit -- without a Postgres or ten
volunteers.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.audit import AuditEvent, AuditKind
from app.flow.states import TERMINAL_STATES, State

# The path a successful applicant walks, in order. Not derived from the
# transition map on purpose: the map is a graph with retries and escape
# hatches, and a funnel needs a spine. Keeping it explicit also means adding a
# state to the machine does not silently reshape the metric.
JOURNEY: tuple[State, ...] = (
    State.START,
    State.LANGUAGE_SELECT,
    State.CONSENT,
    State.ELIGIBILITY,
    State.COLLECT_DATA,
    State.KYC_HANDOFF,
    State.KYC_PENDING,
    State.ACCOUNT_CREATE,
    State.FUNDING,
    State.FIRST_TRANSACTION,
    State.COMPLETED,
)

# The step the product exists to get people past. An account that opens and is
# never used is the failure being solved, so the interesting drop is here.
ACTIVATION_STEP = State.FUNDING


@dataclass(frozen=True, slots=True)
class Step:
    state: State
    reached: int
    share: float
    lost_here: int
    loss_rate: float
    median_seconds: float | None

    @property
    def is_worst(self) -> bool:  # filled in by Funnel; see `worst_step`
        return False


@dataclass
class Funnel:
    sessions: int
    steps: list[Step] = field(default_factory=list)
    exits: Counter = field(default_factory=Counter)
    lapses: Counter = field(default_factory=Counter)
    interruptions: Counter = field(default_factory=Counter)
    refusals: Counter = field(default_factory=Counter)

    @property
    def completed(self) -> int:
        return self.exits.get(str(State.COMPLETED), 0)

    @property
    def completion_rate(self) -> float:
        """The metric. Completed journeys over journeys started."""
        return self.completed / self.sessions if self.sessions else 0.0

    @property
    def opened_an_account(self) -> int:
        """Reached FUNDING, i.e. the account exists. The denominator for the
        question this product is actually about."""
        for step in self.steps:
            if step.state is ACTIVATION_STEP:
                return step.reached
        return 0

    @property
    def dormant(self) -> int:
        """Accounts opened that never made a first transaction.

        The number the whole design is aimed at. SBI's problem stated as one
        integer: people who were successfully onboarded and never used it.
        """
        return max(0, self.opened_an_account - self.completed)

    @property
    def worst_step(self) -> Step | None:
        """Where the most people are lost, in absolute terms.

        Absolute rather than rate, deliberately: a 50% loss on a step four
        people reached is noise, and chasing it wastes the study.
        """
        candidates = [s for s in self.steps if s.lost_here > 0]
        return max(candidates, key=lambda s: s.lost_here) if candidates else None


def _reached(trail: Sequence[AuditEvent]) -> dict[str, AuditEvent | None]:
    """First arrival at each state, by name. None means "reached, when unknown".

    START is implicit -- nothing transitions *into* it, so a trail that exists
    at all belongs to somebody who started. But the trail's first entry is the
    transition OUT of START, so when they arrived there is not recorded
    anywhere, and it maps to None rather than to that timestamp.

    Using the first event's time instead would report every session as taking
    zero seconds on the first step -- a confident number for something the
    data cannot answer, which is worse than an empty cell.
    """
    first: dict[str, AuditEvent | None] = {}
    for record in trail:
        if record.kind is not AuditKind.TRANSITION_ACCEPTED or not record.to_state:
            continue
        first.setdefault(record.to_state, record)
    if trail:
        first.setdefault(str(State.START), None)
    return first


def _final_state(trail: Sequence[AuditEvent]) -> str | None:
    terminal = {str(s) for s in TERMINAL_STATES}
    for record in reversed(trail):
        if record.kind is AuditKind.TRANSITION_ACCEPTED and record.to_state in terminal:
            return record.to_state
    return None


def build(trails: Iterable[Sequence[AuditEvent]]) -> Funnel:
    """One funnel from many journeys."""
    trails = [t for t in trails if t]
    funnel = Funnel(sessions=len(trails))
    if not trails:
        return funnel

    arrivals = [_reached(t) for t in trails]

    for trail in trails:
        end = _final_state(trail)
        if end:
            funnel.exits[end] += 1

        for record in trail:
            if record.kind is AuditKind.SESSION_LAPSED and record.from_state:
                funnel.lapses[record.from_state] += 1
            elif record.kind is AuditKind.CALLER_INTERRUPTED and record.from_state:
                funnel.interruptions[record.from_state] += 1
            elif record.kind is AuditKind.TRANSITION_REFUSED and record.from_state:
                funnel.refusals[record.from_state] += 1

    previous = None
    for index, state in enumerate(JOURNEY):
        name = str(state)
        here = [a for a in arrivals if name in a]
        reached = len(here)

        # Time from arriving at the previous step to arriving at this one.
        # Median, not mean: one person who left the tab open overnight would
        # otherwise dominate every number in the column.
        durations = []
        if index > 0:
            before = str(JOURNEY[index - 1])
            for a in here:
                # Both endpoints have to be timed. `a[before]` is None for
                # START, so the first step honestly reports nothing.
                if a.get(before) is None or a.get(name) is None:
                    continue
                delta = (a[name].recorded_at - a[before].recorded_at).total_seconds()
                if delta >= 0:
                    durations.append(delta)

        lost = max(0, (previous if previous is not None else reached) - reached)
        funnel.steps.append(
            Step(
                state=state,
                reached=reached,
                share=reached / funnel.sessions,
                lost_here=lost,
                loss_rate=(lost / previous) if previous else 0.0,
                median_seconds=statistics.median(durations) if durations else None,
            )
        )
        previous = reached

    return funnel
