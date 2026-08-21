"""The Postgres layer, against a real Postgres.

Skipped entirely unless DATABASE_URL is set, so the suite stays offline by
default. There is deliberately no mock here: the whole value of this layer is
whether the SQL is right, and a fake connection that agrees with my SQL proves
only that I am consistent with myself.

SKIPPING IS A CHOICE, NOT A CONFIGURATION ACCIDENT
--------------------------------------------------
Those are two different situations, and this file treats them differently:

    no DATABASE_URL at all   ->  skip. Running offline, on purpose.
    DATABASE_URL, but empty  ->  error. You meant to configure this and did not.

The second used to skip as well. That is how this whole layer went a commit
without a single line of its SQL ever executing, while the summary read
"822 passed, 14 skipped" and looked healthy. A blank value is not a decision to
run offline; it is a decision that failed halfway.

The skip above still handles the honest case. The dishonest one is caught by
`tests/test_env_config.py`, which lives in its own module precisely because the
`pytestmark` on this one would skip it too -- a guard that skips under the
conditions it exists to detect is not a guard.

    docker run -d --name onboarding-db -p 5432:5432 \\
      -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=onboarding postgres:17
    .venv\\Scripts\\python.exe scripts/db.py apply
    .venv\\Scripts\\python.exe -m pytest tests/test_postgres.py -v
"""

from __future__ import annotations

import collections
import os
import threading
import uuid

import pytest

from app.audit import AuditKind
from app.flow.states import Event, State
from app.session import Session
from app.web.registry import SessionRegistry, hash_token

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL is not set; see the module docstring to start one",
)

psycopg = pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def schema():
    from app.store.postgres import apply_schema

    apply_schema(DATABASE_URL)
    return DATABASE_URL


@pytest.fixture
def store(schema):
    from app.store.postgres import PostgresSessionStore

    return PostgresSessionStore(schema)


@pytest.fixture
def sink(schema):
    from app.store.postgres import PostgresAuditSink

    return PostgresAuditSink(schema)


@pytest.fixture
def tokens(schema):
    from app.store.postgres import PostgresTokenStore

    return PostgresTokenStore(schema)


def fresh_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Sessions round-trip, including everything the in-memory store round-trips.
# ---------------------------------------------------------------------------


def test_a_session_survives_the_database(store):
    session = Session(session_id=fresh_id(), state=State.KYC_PENDING)
    session.language = "hi-IN"
    session.consent_recorded = True
    session.applicant = {"full_name": "Asha Devi", "mobile": "9812345678"}
    session.notes = {"last_kyc_failure": "NETWORK_DROPPED"}
    session.kyc_attempts = 2
    session.kyc_reference = "vckyc-abc"

    store.save(session)
    loaded = store.load(session.session_id)

    assert loaded is not None
    assert loaded.to_dict() == session.to_dict()
    assert loaded.state is State.KYC_PENDING
    assert loaded.notes["last_kyc_failure"] == "NETWORK_DROPPED"


def test_saving_twice_updates_rather_than_duplicating(store):
    session = Session(session_id=fresh_id())
    store.save(session)

    session.language = "ta-IN"
    session.state = State.CONSENT
    store.save(session)

    loaded = store.load(session.session_id)
    assert loaded.language == "ta-IN"
    assert loaded.state is State.CONSENT


def test_an_unknown_session_loads_as_none(store):
    assert store.load(fresh_id()) is None


def test_the_eligibility_decision_survives(store):
    from app.flow.eligibility import evaluate
    from tests.conftest import eligible_applicant

    session = Session(session_id=fresh_id())
    applicant = eligible_applicant()
    session.eligibility_input = applicant
    session.eligibility = evaluate(applicant)

    store.save(session)
    loaded = store.load(session.session_id)

    assert loaded.eligibility == session.eligibility
    assert loaded.eligibility_input == session.eligibility_input
    # The replay promise, across a database round trip.
    assert evaluate(loaded.eligibility_input) == loaded.eligibility


def test_sessions_waiting_at_kyc_can_be_found(store):
    waiting = Session(session_id=fresh_id(), state=State.KYC_PENDING)
    elsewhere = Session(session_id=fresh_id(), state=State.COLLECT_DATA)
    store.save(waiting)
    store.save(elsewhere)

    parked = store.suspended_at_kyc()

    assert waiting.session_id in parked
    assert elsewhere.session_id not in parked


# ---------------------------------------------------------------------------
# The audit trail. Append-only, and gapless.
# ---------------------------------------------------------------------------


def test_the_sequence_is_gapless(sink):
    session_id = fresh_id()
    for index in range(6):
        record = sink.append(
            session_id,
            AuditKind.TRANSITION_ACCEPTED,
            from_state=State.START,
            to_state=State.LANGUAGE_SELECT,
            event=Event.SESSION_STARTED,
            detail=f"step {index}",
        )
        assert record.sequence == index

    trail = sink.for_session(session_id)
    assert [e.sequence for e in trail] == list(range(6))


def test_two_sessions_number_independently(sink):
    first, second = fresh_id(), fresh_id()
    sink.append(first, AuditKind.TRANSITION_ACCEPTED)
    sink.append(second, AuditKind.TRANSITION_ACCEPTED)

    assert sink.for_session(first)[0].sequence == 0
    assert sink.for_session(second)[0].sequence == 0


def test_refusals_are_recorded_too(sink):
    session_id = fresh_id()
    sink.append(
        session_id,
        AuditKind.TRANSITION_REFUSED,
        from_state=State.COLLECT_DATA,
        event=Event.ACCOUNT_OPENED,
        detail="no such edge",
    )

    (record,) = sink.for_session(session_id)
    assert record.kind is AuditKind.TRANSITION_REFUSED
    assert record.to_state is None


def test_the_database_refuses_to_update_the_audit_trail(sink, schema):
    """The claim audit.py makes, enforced where the data actually lives.

    In Python the guarantee is that the sink has no update method, which stops
    accidents and stops nothing else. Here a trigger stops anyone with a psql
    prompt, including the table's owner.
    """
    session_id = fresh_id()
    sink.append(session_id, AuditKind.TRANSITION_ACCEPTED, detail="original")

    with psycopg.connect(schema, autocommit=True) as connection:
        with pytest.raises(psycopg.errors.RaiseException):
            connection.execute(
                "UPDATE audit_events SET detail = %s WHERE session_id = %s",
                ("tampered", session_id),
            )

    (record,) = sink.for_session(session_id)
    assert record.detail == "original"


def test_the_database_refuses_to_delete_from_the_audit_trail(sink, schema):
    session_id = fresh_id()
    sink.append(session_id, AuditKind.TRANSITION_ACCEPTED, detail="keep me")

    with psycopg.connect(schema, autocommit=True) as connection:
        with pytest.raises(psycopg.errors.RaiseException):
            connection.execute(
                "DELETE FROM audit_events WHERE session_id = %s", (session_id,)
            )

    assert len(sink.for_session(session_id)) == 1


# ---------------------------------------------------------------------------
# Tokens. Only the hash is stored.
# ---------------------------------------------------------------------------


def test_the_plaintext_token_is_never_in_the_database(store, tokens, schema):
    """A dump, a backup, or a support engineer with SELECT must not come with
    the ability to attach to live sessions."""
    registry = SessionRegistry(store=store, tokens=tokens)
    session, token = registry.create()

    with psycopg.connect(schema) as connection:
        rows = connection.execute(
            "SELECT token_sha256 FROM session_tokens WHERE session_id = %s",
            (session.session_id,),
        ).fetchall()

    assert rows
    stored = rows[0][0]
    assert stored != token
    assert stored == hash_token(token)


def test_the_registry_still_authorises_across_the_database(store, tokens):
    registry = SessionRegistry(store=store, tokens=tokens)
    session, token = registry.create()

    assert registry.open(session.session_id, token) is not None
    assert registry.open(session.session_id, "wrong") is None
    assert registry.open(session.session_id, "") is None
    assert registry.open(fresh_id(), token) is None


def test_forgetting_a_token_leaves_the_session(store, tokens):
    registry = SessionRegistry(store=store, tokens=tokens)
    session, token = registry.create()

    registry.forget(session.session_id)

    assert registry.open(session.session_id, token) is None
    assert store.load(session.session_id) is not None


# ---------------------------------------------------------------------------
# The thing this whole layer exists for.
# ---------------------------------------------------------------------------


def test_a_session_survives_a_process_restart(schema):
    """Two registries, sharing nothing but the database.

    This is the failure that motivated the work: restarting the server used to
    invalidate every live session, which for anyone parked at the regulatory
    handoff meant losing an application designed to last hours.
    """
    from app.store.postgres import PostgresSessionStore, PostgresTokenStore

    def new_process() -> SessionRegistry:
        return SessionRegistry(
            store=PostgresSessionStore(schema),
            tokens=PostgresTokenStore(schema),
        )

    before = new_process()
    session, token = before.create()
    session.state = State.KYC_PENDING
    session.language = "hi-IN"
    session.kyc_reference = "vckyc-xyz"
    before.save(session)

    del before  # the process ends here

    after = new_process()
    recovered = after.open(session.session_id, token)

    assert recovered is not None
    assert recovered.state is State.KYC_PENDING
    assert recovered.language == "hi-IN"
    assert recovered.kyc_reference == "vckyc-xyz"


# ---------------------------------------------------------------------------
# The claims the module docstring makes, which nothing used to check.
# ---------------------------------------------------------------------------


def test_concurrent_writers_do_not_lose_events(sink):
    """The gapless claim, against writers that actually race.

    `test_the_sequence_is_gapless` appends in a single thread, which cannot
    fail: there is nothing to race. This is the test that found the real bug.
    The original sink computed `MAX(sequence) + 1` optimistically and retried a
    fixed five times; under this load it dropped roughly 2% of events, and the
    drop rate rose with contention -- an audit trail that forgets most when it
    is busiest.

    Twelve writers on one session is far harsher than production, where a
    session carries one conversation. That is deliberate: the mechanism should
    be correct under contention, not merely unlikely to meet it.
    """
    session_id = fresh_id()
    writers, each = 12, 20
    sequences: list[int] = []
    failures: list[str] = []
    guard = threading.Lock()
    gate = threading.Barrier(writers)

    def writer(n: int) -> None:
        gate.wait()  # everyone swings at once
        for i in range(each):
            try:
                record = sink.append(
                    session_id,
                    AuditKind.TRANSITION_ACCEPTED,
                    detail=f"writer {n} event {i}",
                )
                with guard:
                    sequences.append(record.sequence)
            except Exception as exc:  # noqa: BLE001 - the point is to see any
                with guard:
                    failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"appends were dropped: {failures[:3]}"

    duplicates = [s for s, count in collections.Counter(sequences).items() if count > 1]
    assert not duplicates, f"the same sequence was handed out twice: {duplicates}"

    trail = sink.for_session(session_id)
    assert len(trail) == writers * each
    assert [e.sequence for e in trail] == list(range(writers * each)), "gap in the trail"


def test_the_database_refuses_to_truncate_the_audit_trail(sink, schema):
    """The hole the row-level trigger left open.

    `audit_events_no_update` is FOR EACH ROW, and TRUNCATE fires no row
    triggers -- so before `audit_events_no_truncate` existed, one ordinary
    statement emptied the whole table while the trigger that supposedly made
    it append-only watched. "You cannot delete these rows" and "you cannot
    empty this table" are different claims, and the docstring makes the second.

    Rolled back either way, so a regression fails the assertion instead of
    destroying the trail the rest of the suite is building.
    """
    session_id = fresh_id()
    sink.append(session_id, AuditKind.TRANSITION_ACCEPTED, detail="keep me")

    with psycopg.connect(schema) as connection:  # autocommit off: TRUNCATE is transactional
        with pytest.raises(psycopg.errors.RaiseException):
            connection.execute("TRUNCATE audit_events")
        connection.rollback()

    assert len(sink.for_session(session_id)) == 1
