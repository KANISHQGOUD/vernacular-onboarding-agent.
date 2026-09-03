"""The drop-off report, against journeys made up on purpose.

Pure and offline. The point of computing this from audit events rather than
from a database is that the awkward journeys -- somebody who lapsed four
times, somebody whose account opened and was never used -- can be written down
rather than waited for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.audit import AuditEvent, AuditKind
from app.flow.states import Event, State
from app.funnel import ACTIVATION_STEP, JOURNEY, build

START_AT = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)


def journey(*states, session="s", seconds_each=30, extras=()):
    """A trail that walks the given states in order, one transition each."""
    trail = []
    at = START_AT
    for index, (origin, destination) in enumerate(zip(states, states[1:])):
        at = at + timedelta(seconds=seconds_each)
        trail.append(
            AuditEvent(
                session_id=session,
                sequence=index,
                kind=AuditKind.TRANSITION_ACCEPTED,
                recorded_at=at,
                from_state=str(origin),
                to_state=str(destination),
                event=str(Event.SESSION_STARTED),
                tool=None,
                detail="",
            )
        )
    for kind, state in extras:
        trail.append(
            AuditEvent(
                session_id=session,
                sequence=len(trail),
                kind=kind,
                recorded_at=at,
                from_state=str(state),
                to_state=str(state),
                event=None,
                tool=None,
                detail="",
            )
        )
    return trail


HAPPY = JOURNEY  # START ... COMPLETED


def test_no_sessions_is_not_a_division_by_zero():
    empty = build([])
    assert empty.sessions == 0
    assert empty.completion_rate == 0.0
    assert empty.worst_step is None


def test_one_finished_journey_is_a_hundred_percent():
    funnel = build([journey(*HAPPY)])
    assert funnel.sessions == 1
    assert funnel.completed == 1
    assert funnel.completion_rate == 1.0
    assert funnel.dormant == 0


def test_the_metric_is_first_transaction_not_account_opened():
    """The number this project exists for, and the reason it is not
    "accounts opened".

    Three people get an account. One of them uses it. A funnel that stopped at
    ACCOUNT_CREATE would call that a 100% success and would be describing
    precisely the dormant-account problem this is meant to solve.
    """
    upto_funding = HAPPY[: HAPPY.index(ACTIVATION_STEP) + 1]
    trails = [
        journey(*HAPPY, session="finished"),
        journey(*upto_funding, session="opened-and-stopped-1"),
        journey(*upto_funding, session="opened-and-stopped-2"),
    ]
    funnel = build(trails)

    assert funnel.opened_an_account == 3
    assert funnel.completed == 1
    assert funnel.dormant == 2, "two accounts opened and never used"
    assert funnel.completion_rate < 0.4


def test_the_worst_step_is_absolute_not_a_rate():
    """A 50% loss on a step four people reached is noise, and chasing it
    wastes the study. The worst step is where the most PEOPLE are lost."""
    upto_consent = HAPPY[: HAPPY.index(State.CONSENT) + 1]
    upto_kyc = HAPPY[: HAPPY.index(State.KYC_HANDOFF) + 1]

    trails = [journey(*upto_consent, session=f"early-{i}") for i in range(2)]
    trails += [journey(*upto_kyc, session=f"mid-{i}") for i in range(20)]
    trails += [journey(*HAPPY, session="done")]

    worst = build(trails).worst_step
    assert worst is not None
    assert worst.state is State.KYC_PENDING, f"picked {worst.state}"


def test_reaching_a_state_is_recorded_once_however_many_retries():
    """KYC retries loop back. A funnel that counted arrivals rather than
    whether somebody ever arrived would report more than 100%."""
    trail = journey(State.KYC_HANDOFF, State.KYC_PENDING, State.KYC_HANDOFF, State.KYC_PENDING)
    funnel = build([trail])

    for step in funnel.steps:
        assert step.reached <= funnel.sessions
        assert step.share <= 1.0


def test_where_people_go_quiet_is_counted_separately_from_where_they_stop():
    """Lapsing is not an exit -- the session stays where it is -- so it has
    its own column. "They go silent while giving their address" is a different
    finding from "they abandon at consent"."""
    trail = journey(
        *HAPPY[: HAPPY.index(State.COLLECT_DATA) + 1],
        extras=[
            (AuditKind.SESSION_LAPSED, State.COLLECT_DATA),
            (AuditKind.SESSION_LAPSED, State.COLLECT_DATA),
            (AuditKind.CALLER_INTERRUPTED, State.CONSENT),
        ],
    )
    funnel = build([trail])

    assert funnel.lapses[str(State.COLLECT_DATA)] == 2
    assert funnel.interruptions[str(State.CONSENT)] == 1
    # And none of it counted as an ending.
    assert funnel.exits == {}


def test_exits_are_broken_out_by_reason():
    """Five ways to stop, and lumping them together is the mistake three
    commits were spent undoing."""
    trails = [
        journey(*HAPPY, session="done"),
        journey(State.START, State.LANGUAGE_SELECT, State.CONSENT, State.CONSENT_REFUSED, session="no"),
        journey(State.START, State.LANGUAGE_SELECT, State.ABANDONED, session="quit"),
        journey(State.START, State.LANGUAGE_SELECT, State.CONSENT, State.ELIGIBILITY, State.DECLINED, session="declined"),
    ]
    funnel = build(trails)

    assert funnel.exits[str(State.COMPLETED)] == 1
    assert funnel.exits[str(State.CONSENT_REFUSED)] == 1
    assert funnel.exits[str(State.ABANDONED)] == 1
    assert funnel.exits[str(State.DECLINED)] == 1


def test_how_long_each_step_took():
    """Median, not mean: one person who left the tab open overnight would
    otherwise dominate every number in the column."""
    trails = [
        journey(*HAPPY, session="quick", seconds_each=10),
        journey(*HAPPY, session="slow", seconds_each=20),
        journey(*HAPPY, session="asleep", seconds_each=100_000),
    ]
    funnel = build(trails)

    timed = [s for s in funnel.steps if s.median_seconds is not None]
    assert timed, "no step reported a duration"
    assert all(s.median_seconds == 20 for s in timed), [s.median_seconds for s in timed]


def test_a_session_that_never_transitioned_still_counts_as_started():
    """Somebody who opened the page and said nothing is a data point, and the
    most important kind -- they are the drop-off before anything happened."""
    lone = [
        AuditEvent(
            session_id="silent",
            sequence=0,
            kind=AuditKind.SESSION_LAPSED,
            recorded_at=START_AT,
            from_state=str(State.LANGUAGE_SELECT),
            to_state=str(State.LANGUAGE_SELECT),
            event=None,
            tool=None,
            detail="",
        )
    ]
    funnel = build([lone])

    assert funnel.sessions == 1
    assert funnel.completed == 0
    assert funnel.lapses[str(State.LANGUAGE_SELECT)] == 1
