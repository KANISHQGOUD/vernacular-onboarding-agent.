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
from datetime import date

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


@pytest.fixture(scope="module")
def pool(schema):
    """One pool for the module, closed at the end.

    Module-scoped because building a pool opens real connections, and a
    per-test pool would put back most of the setup cost this change exists to
    remove. Closed on teardown: `ConnectionPool` runs worker threads, and
    leaking one per test module is how a suite ends up hanging at exit.
    """
    from app.store.postgres import open_pool

    created = open_pool(schema, name="tests")
    yield created
    created.close()


@pytest.fixture
def store(pool):
    from app.store.postgres import PostgresSessionStore

    return PostgresSessionStore(pool)


@pytest.fixture
def sink(pool):
    from app.store.postgres import PostgresAuditSink

    return PostgresAuditSink(pool)


@pytest.fixture
def tokens(pool):
    from app.store.postgres import PostgresTokenStore

    return PostgresTokenStore(pool)


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
    from app.store.postgres import PostgresSessionStore, PostgresTokenStore, open_pool

    # A pool each, not a shared one. The point of the test is that nothing but
    # the database carries the session across, and a shared pool would be a
    # second thing both "processes" have in common.
    pools = []

    def new_process() -> SessionRegistry:
        created = open_pool(schema, min_size=1, max_size=2)
        pools.append(created)
        return SessionRegistry(
            store=PostgresSessionStore(created),
            tokens=PostgresTokenStore(created),
        )

    try:
        before = new_process()
        session, token = before.create()
        session.state = State.KYC_PENDING
        session.language = "hi-IN"
        session.kyc_reference = "vckyc-xyz"
        before.save(session)

        pools.pop().close()  # the process ends here, connections and all
        del before

        after = new_process()
        recovered = after.open(session.session_id, token)
    finally:
        for leftover in pools:
            leftover.close()

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


# ---------------------------------------------------------------------------
# Erasure. Delete the person, keep the trail.
#
# HOW FAR "DELETED" GOES, MEASURED
# --------------------------------
# These tests assert erasure at the SQL level -- the level the application, an
# auditor, and anyone with a psql prompt actually see. That is not the same as
# the bytes being gone, and the difference is worth stating rather than
# discovering during a regulatory conversation.
#
# UPDATE in Postgres writes a new row version and leaves the old one behind as
# a dead tuple. Checked with pageinspect against this schema: after erase and
# save, a marker planted in `applicant` was unreachable by SQL (0 rows) and
# still sitting in the heap page (1 page) -- and still there after a plain
# VACUUM, because VACUUM marks space reusable without zeroing it.
#
# What does remove it: VACUUM FULL or pg_repack, which rewrite the table; or
# eventual overwrite by new tuples. Separately, the WAL keeps the old value for
# its retention window, and so does every backup taken before the erasure.
#
# So the honest claim is "erased from the database" and not "erased from the
# disk". A real deployment answering a DPDP request needs the second, which is
# an operational procedure, not a method call.
# ---------------------------------------------------------------------------


def _stored_text(conninfo: str, session_id: str) -> str:
    """Everything this session has left in the database, as one searchable string.

    `data::text` on purpose: this reads what Postgres actually stored, not what
    `Session.from_dict` chooses to hand back. Asserting through the loader
    would only prove the loader agrees with the saver, which is the same
    circularity this module's docstring refuses mocks for.
    """
    with psycopg.connect(conninfo) as connection:
        row = connection.execute(
            "SELECT data::text FROM sessions WHERE session_id = %s", (session_id,)
        ).fetchone()
        trail = connection.execute(
            """
            SELECT kind, from_state, to_state, event, tool, detail
            FROM audit_events WHERE session_id = %s
            """,
            (session_id,),
        ).fetchall()

    parts = [row[0] if row else ""]
    parts.extend(str(value) for record in trail for value in record if value is not None)
    return " ".join(parts)


def _applicant_with_everything() -> dict[str, str]:
    """Every field REQUIRED_APPLICANT_FIELDS names, all of it personal data."""
    return {
        "full_name": "Asha Devi Rathore",
        "date_of_birth": "1990-01-01",
        "address": "14 Gandhi Marg, Jaipur 302001",
        "occupation": "weaver",
        "mobile": "9812345678",
    }


def test_erasure_removes_the_person_from_the_database(store, schema):
    """The applicant / notes split, verified where the data actually sits.

    `test_security.py` already checks this in memory, but an in-memory store
    holds the Session object itself -- there are no stored bytes to search, so
    "the person is gone" can only be asserted about the object. Here there is a
    row, and the question is whether the person is still inside it.
    """
    from app.flow.eligibility import evaluate
    from tests.conftest import eligible_applicant

    session = Session(session_id=fresh_id(), state=State.COMPLETED)
    session.language = "hi-IN"
    session.consent_recorded = True
    session.applicant = _applicant_with_everything()
    # Raw transcripts of the same personal data, which must go with it.
    session.heard = dict(_applicant_with_everything())
    session.notes = {"last_kyc_failure": "NETWORK_DROPPED", "attempts": "2"}
    session.eligibility_input = eligible_applicant()
    session.eligibility = evaluate(session.eligibility_input)
    session.account_number = "00012345678"
    store.save(session)

    # The test proves nothing unless the data was genuinely in there first.
    before = _stored_text(schema, session.session_id)
    for value in _applicant_with_everything().values():
        assert value in before, f"{value} was never stored, so erasing it proves nothing"

    session.erase_personal_data()
    store.save(session)

    after = _stored_text(schema, session.session_id)
    for name, value in _applicant_with_everything().items():
        assert value not in after, f"{name} survived erasure in the stored row"

    loaded = store.load(session.session_id)
    assert loaded is not None, "erasure deleted the session, not just the person"
    assert loaded.applicant == {}
    assert loaded.heard == {}, "the raw transcripts survived erasure"
    assert loaded.eligibility_input is None, "date of birth and residency survived"

    # What the bank must still be able to answer afterwards.
    assert loaded.eligibility == session.eligibility, "the decision did not survive"
    assert loaded.state is State.COMPLETED
    assert loaded.account_number == "00012345678"
    assert loaded.notes["last_kyc_failure"] == "NETWORK_DROPPED"


def test_the_surviving_decision_carries_no_personal_data(store, schema):
    """Erasure would be pointless if the verdict quietly kept its inputs.

    `EligibilityDecision` is ruleset_version, passed, and a rule_id / passed /
    decline_reason per rule -- structural all the way down, which is what makes
    keeping it compatible with erasing the person. This pins that shape down,
    because a later rule that helpfully recorded "declined: born 2015-06-30"
    would breach the erasure guarantee without touching a line of this file.
    """
    from app.flow.eligibility import evaluate
    from tests.conftest import eligible_applicant

    # Young enough to fail the age rule, so there are real decline reasons.
    applicant = eligible_applicant(date_of_birth=date(2015, 6, 30))

    session = Session(session_id=fresh_id())
    session.applicant = _applicant_with_everything()
    session.eligibility_input = applicant
    session.eligibility = evaluate(applicant)
    assert not session.eligibility.passed, "pick an input that actually declines"

    store.save(session)
    session.erase_personal_data()
    store.save(session)

    after = _stored_text(schema, session.session_id)
    assert "2015-06-30" not in after, "the date of birth survived inside the decision"

    loaded = store.load(session.session_id)
    assert loaded.eligibility.decline_reasons, "the reasons should outlive the person"


def test_erasure_leaves_the_audit_trail_untouched(store, sink, schema):
    """The point of the split: answer "was this handled correctly" after you
    can no longer answer "who was it".

    The trail refuses UPDATE, DELETE and TRUNCATE, so it could not be scrubbed
    even deliberately. That only stays compatible with an erasure request
    because nothing personal is written there in the first place.
    """
    session = Session(session_id=fresh_id(), state=State.COMPLETED)
    session.applicant = _applicant_with_everything()
    store.save(session)

    sink.append(
        session.session_id,
        AuditKind.TRANSITION_ACCEPTED,
        from_state=State.START,
        to_state=State.LANGUAGE_SELECT,
        event=Event.SESSION_STARTED,
        detail="unconditional",
    )
    sink.append(
        session.session_id,
        AuditKind.TOOL_ALLOWED,
        from_state=State.ACCOUNT_CREATE,
        tool="open_account",
    )
    before = sink.for_session(session.session_id)

    session.erase_personal_data()
    store.save(session)

    after = sink.for_session(session.session_id)
    assert after == before, "erasing the person changed the trail"
    assert [record.sequence for record in after] == [0, 1]
    assert after[0].detail == "unconditional"


def test_the_audit_trail_never_held_the_person_to_begin_with(store, sink):
    """Erasure cannot reach an append-only table, so the guarantee has to come
    from never writing personal data there -- not from cleaning it up later."""
    from app.flow.machine import Machine

    session = Session(session_id=fresh_id())
    session.applicant = _applicant_with_everything()
    store.save(session)

    Machine(sink).fire(session, Event.SESSION_STARTED)
    store.save(session)

    trail = sink.for_session(session.session_id)
    assert trail, "the transition should have been recorded"

    written = " ".join(
        str(value)
        for record in trail
        for value in (
            record.kind,
            record.from_state,
            record.to_state,
            record.event,
            record.tool,
            record.detail,
        )
        if value is not None
    )
    for name, value in _applicant_with_everything().items():
        assert value not in written, f"{name} reached the audit trail"


# ---------------------------------------------------------------------------
# Pooling. A borrowed connection has to go back in the state it came out in.
# ---------------------------------------------------------------------------


def test_a_pooled_connection_goes_back_without_locks_or_an_open_transaction(sink, pool, schema):
    """The failure pooling introduces, which no functional test would catch.

    Every assertion here would still pass on a broken implementation if it
    were checked through the pool, so it is checked from a connection outside
    it.

    Two ways this goes wrong, both silent:

      `pg_advisory_lock` instead of `pg_advisory_xact_lock` -- session-scoped,
      so the lock survives the return to the pool and poisons that connection
      for whoever borrows it next. The symptom is a deadlock that moves
      between unrelated sessions and vanishes on restart.

      A transaction left open -- the connection sits "idle in transaction",
      holding its locks and blocking VACUUM on the whole table for as long as
      the pool keeps it, which is forever.
    """
    session_id = fresh_id()
    for index in range(5):
        sink.append(session_id, AuditKind.TRANSITION_ACCEPTED, detail=f"event {index}")
    assert len(sink.for_session(session_id)) == 5

    with psycopg.connect(schema) as watcher:
        advisory = watcher.execute(
            """
            SELECT count(*) FROM pg_locks
            WHERE locktype = 'advisory'
              AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
            """
        ).fetchone()[0]
        idle_in_transaction = watcher.execute(
            """
            SELECT count(*) FROM pg_stat_activity
            WHERE datname = current_database()
              AND state = 'idle in transaction'
            """
        ).fetchone()[0]

    assert advisory == 0, "an advisory lock outlived its transaction"
    assert idle_in_transaction == 0, "a pooled connection was left mid-transaction"


def test_connections_are_returned_rather_than_accumulated(store, sink, pool):
    """Borrow far more times than the pool is big, and it must not grow."""
    from app.store.postgres import POOL_MAX_SIZE

    session_id = fresh_id()
    for index in range(40):
        store.save(Session(session_id=session_id, state=State.COLLECT_DATA))
        store.load(session_id)
        sink.append(session_id, AuditKind.TOOL_ALLOWED, detail=f"borrow {index}")

    stats = pool.get_stats()
    assert stats["pool_size"] <= POOL_MAX_SIZE
    assert stats.get("requests_waiting", 0) == 0
    # Everything handed out came back.
    assert stats["pool_available"] == stats["pool_size"]
