"""Property tests: drive the machine randomly and assert what must never break.

The other suites test the paths somebody thought of. This one tests the paths
nobody thought of, by walking the graph at random tens of thousands of times
and checking invariants after every single step.

Seeded `random` rather than a property-testing library, deliberately: no new
dependency, and a failure reproduces exactly from the seed printed in the
assertion. Determinism matters more here than shrinking.

The invariants are the claims the whole project rests on:

    - a terminal state is absorbing
    - COMPLETED is unreachable without a confirmed payment
    - ACCOUNT_CREATE is unreachable without verified identity
    - the audit sequence is gapless, whatever happened
    - a session survives storage at any point in any walk
    - refusals never mutate anything
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from app.audit import InMemoryAuditSink
from app.flow.eligibility import EligibilityDecision, EligibilityInput, Residency, evaluate
from app.flow.machine import FlowError, Machine
from app.flow.states import TERMINAL_STATES, Event, State
from app.providers.payments import (
    UPI_PER_TRANSACTION_LIMIT_PAISE,
    MockPaymentRail,
    format_rupees,
)
from app.session import InMemorySessionStore, Session

WALKS = 400
STEPS = 60
SEEDS = range(12)


def _random_session(rng: random.Random, session_id: str) -> Session:
    """A session with arbitrary but type-correct contents."""
    session = Session(session_id=session_id)
    session.language = rng.choice([None, "hi-IN", "ta-IN", "en-IN", "bn-IN"])
    session.consent_recorded = rng.choice([True, False])
    session.kyc_attempts = rng.randint(0, 6)
    session.kyc_reference = rng.choice([None, "vckyc-abc123"])
    session.account_number = rng.choice([None, "00012345678"])
    session.deposit_reference = rng.choice([None, "dep-1"])
    session.first_payment_reference = rng.choice([None, "upi-1"])
    for name in ("full_name", "address", "occupation", "mobile", "date_of_birth"):
        if rng.random() < 0.6:
            session.applicant[name] = f"value-{rng.randint(0, 999)}"
    if rng.random() < 0.5:
        session.notes["last_kyc_failure"] = "NETWORK_DROPPED"
    if rng.random() < 0.5:
        applicant = _random_eligibility_input(rng)
        session.eligibility_input = applicant
        session.eligibility = evaluate(applicant)
    return session


def _random_eligibility_input(rng: random.Random) -> EligibilityInput:
    as_of = date(2026, 8, 13)
    return EligibilityInput(
        date_of_birth=as_of - timedelta(days=rng.randint(1, 40_000)),
        residency=rng.choice(list(Residency)),
        has_pan=rng.choice([True, False]),
        has_form_60=rng.choice([True, False]),
        has_existing_active_account=rng.choice([True, False]),
        mobile_linked_to_aadhaar=rng.choice([True, False]),
        as_of=as_of,
    )


# ---------------------------------------------------------------------------
# Random walks through the graph.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_random_walks_hold_every_invariant(seed: int):
    """The core fuzz. Fire random events at random sessions, forever."""
    rng = random.Random(seed)
    events = list(Event)

    for walk in range(WALKS // len(SEEDS) + 1):
        audit = InMemoryAuditSink()
        machine = Machine(audit)
        session_id = f"s{seed}-{walk}"
        session = Session(session_id=session_id)

        for step in range(STEPS):
            before = session.state
            event = rng.choice(events)

            try:
                machine.fire(session, event)
            except FlowError:
                # A refusal must change absolutely nothing.
                assert session.state is before, (
                    f"seed={seed} walk={walk} step={step}: refused {event} "
                    f"but state moved {before} -> {session.state}"
                )
                continue

            ctx = f"seed={seed} walk={walk} step={step} {before} --{event}--> {session.state}"

            # A terminal state must never have been the origin of a move.
            assert before not in TERMINAL_STATES, f"{ctx}: escaped a terminal state"

            # Money only moves via a confirmed payment.
            if session.state is State.COMPLETED:
                assert event is Event.PAYMENT_CONFIRMED, f"{ctx}: completed without payment"

            # Identity is established only by the provider.
            if session.state is State.ACCOUNT_CREATE and before is not State.ACCOUNT_CREATE:
                assert event is Event.KYC_VERIFIED, f"{ctx}: reached account creation unverified"

            assert session.state in set(State), ctx

        # However the walk went, the trail is contiguous.
        trail = audit.for_session(session_id)
        assert [e.sequence for e in trail] == list(range(len(trail)))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_terminal_state_absorbs_everything(seed: int):
    rng = random.Random(seed)
    machine = Machine(InMemoryAuditSink())

    for _ in range(200):
        state = rng.choice(sorted(TERMINAL_STATES))
        session = Session(session_id="absorbing", state=state)
        for _ in range(10):
            with pytest.raises(FlowError):
                machine.fire(session, rng.choice(list(Event)))
            assert session.state is state


@pytest.mark.parametrize("seed", SEEDS)
def test_every_reachable_state_survives_storage(seed: int):
    """Serialise at an arbitrary point in an arbitrary walk and come back."""
    rng = random.Random(seed)
    store = InMemorySessionStore()
    machine = Machine(InMemoryAuditSink())

    for walk in range(40):
        session = _random_session(rng, f"rt{seed}-{walk}")
        for _ in range(rng.randint(0, 15)):
            try:
                machine.fire(session, rng.choice(list(Event)))
            except FlowError:
                pass

        store.save(session)
        loaded = store.load(session.session_id)

        assert loaded is not None
        assert loaded.to_dict() == session.to_dict()
        assert loaded.state is session.state
        assert loaded.eligibility == session.eligibility
        assert loaded.eligibility_input == session.eligibility_input


# ---------------------------------------------------------------------------
# Eligibility: determinism and replay, over arbitrary applicants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_eligibility_is_deterministic_for_any_applicant(seed: int):
    rng = random.Random(seed)
    for _ in range(300):
        applicant = _random_eligibility_input(rng)
        assert evaluate(applicant) == evaluate(applicant)


@pytest.mark.parametrize("seed", SEEDS)
def test_any_decision_replays_from_its_stored_input(seed: int):
    """The audit promise: store the input, recover the decision, years later."""
    rng = random.Random(seed)
    for _ in range(300):
        applicant = _random_eligibility_input(rng)
        decision = evaluate(applicant)

        restored_input = EligibilityInput.from_dict(applicant.to_dict())
        restored_decision = EligibilityDecision.from_dict(decision.to_dict())

        assert restored_input == applicant
        assert restored_decision == decision
        assert evaluate(restored_input) == decision


@pytest.mark.parametrize("seed", SEEDS)
def test_a_decline_always_carries_at_least_one_reason(seed: int):
    """No applicant is ever refused without a recorded reason."""
    rng = random.Random(seed)
    for _ in range(300):
        decision = evaluate(_random_eligibility_input(rng))
        if decision.passed:
            assert decision.decline_reasons == ()
        else:
            assert decision.decline_reasons, "declined with no reason"


# ---------------------------------------------------------------------------
# Money: the ledger cannot be broken by arbitrary traffic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_ledger_never_goes_negative_and_never_goes_float(seed: int):
    rng = random.Random(seed)
    rail = MockPaymentRail()
    account = "00012345678"

    for _ in range(500):
        if rng.random() < 0.5:
            rail.initiate_deposit(account, rng.randint(-5_000, 500_000))
        else:
            rail.send_payment(account, "shop@upi", rng.randint(-5_000, 500_000))

        balance = rail.balance_paise(account)
        assert isinstance(balance, int) and not isinstance(balance, bool)
        assert balance >= 0, "a failed payment moved the ledger"


@pytest.mark.parametrize("seed", SEEDS)
def test_rupee_formatting_round_trips_through_paise(seed: int):
    """Formatting is the only place money becomes decimal, so it must not lie."""
    rng = random.Random(seed)
    for _ in range(500):
        paise = rng.randint(0, UPI_PER_TRANSACTION_LIMIT_PAISE)
        text = format_rupees(paise)
        rupees, _, fraction = text.removeprefix("Rs ").partition(".")
        recovered = int(rupees.replace(",", "")) * 100 + int(fraction)
        assert recovered == paise, f"{paise} rendered as {text}"
