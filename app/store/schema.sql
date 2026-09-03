-- Durable storage for sessions, their access tokens, and the audit trail.
--
-- Three tables, and the third one is the interesting one.
--
-- APPEND-ONLY, ENFORCED BY THE DATABASE
-- -------------------------------------
-- audit.py claims the trail cannot be edited, and in Python that claim rests
-- on the sink exposing no update or delete method. That is a real guarantee
-- against accident and no guarantee at all against anyone holding a psql
-- prompt.
--
-- Here the claim is enforced where the data actually lives: a trigger rejects
-- every UPDATE and DELETE on audit_events, for every role, including the
-- table's owner. Revoking the grants would stop an application bug; the
-- trigger also stops a person who means it.
--
-- Row triggers alone are not enough. TRUNCATE removes every row without firing
-- a single row trigger, so the original BEFORE UPDATE OR DELETE trigger let
-- `TRUNCATE audit_events` empty the table unchallenged -- verified, not
-- assumed. A second statement-level trigger closes that.
--
-- The honest limit: DROP TABLE, and `ALTER TABLE ... DISABLE TRIGGER`, are
-- still open to the owner. No trigger can defend against them. What this buys
-- is that erasing the trail is no longer one ordinary statement -- it takes a
-- deliberate schema change, which is a different thing to do by accident and a
-- different thing to explain afterwards.
--
-- Written to be run more than once. Every statement is idempotent, so applying
-- it to an existing database is a no-op rather than an error.

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    state       TEXT        NOT NULL,
    -- The whole serialised Session. Deliberately one JSONB column rather than
    -- a column per field: `Session.to_dict()` is already the storage format,
    -- already round-trip tested, and already the thing the audit story depends
    -- on. Splitting it into columns would create a second schema to keep in
    -- step with the dataclass, and they would drift.
    data        JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Finding the sessions parked at the regulatory handoff is the one query the
-- product actually needs: those are people waiting on a video call.
CREATE INDEX IF NOT EXISTS sessions_state_idx ON sessions (state);

CREATE TABLE IF NOT EXISTS session_tokens (
    session_id  TEXT PRIMARY KEY REFERENCES sessions (session_id) ON DELETE CASCADE,
    -- HASHED, never the token itself. A database dump, a backup on a laptop,
    -- or a support engineer with read access must not come with the ability to
    -- attach to live sessions. The in-memory registry could hold plaintext
    -- because it dies with the process; a table cannot.
    token_sha256 TEXT       NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_events (
    session_id  TEXT        NOT NULL,
    sequence    INTEGER     NOT NULL,
    kind        TEXT        NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    from_state  TEXT,
    to_state    TEXT,
    event       TEXT,
    tool        TEXT,
    detail      TEXT        NOT NULL DEFAULT '',
    -- The composite key is what makes the sequence gapless under concurrency:
    -- two writers computing the same next number cannot both win, so one
    -- retries instead of silently overwriting.
    PRIMARY KEY (session_id, sequence)
);

CREATE INDEX IF NOT EXISTS audit_events_session_idx
    ON audit_events (session_id, sequence);

-- The append-only guarantee.
CREATE OR REPLACE FUNCTION audit_events_are_append_only()
RETURNS trigger AS $$
BEGIN
    -- A statement-level trigger has no OLD row to name, and touching OLD there
    -- is itself an error -- which would replace this clear refusal with a
    -- confusing one.
    IF TG_LEVEL = 'STATEMENT' THEN
        RAISE EXCEPTION
            'audit_events is append-only: % rejected on the whole table', TG_OP;
    END IF;

    RAISE EXCEPTION
        'audit_events is append-only: % rejected on session %',
        TG_OP, COALESCE(OLD.session_id, '(unknown)');
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_events_no_update ON audit_events;
CREATE TRIGGER audit_events_no_update
    BEFORE UPDATE OR DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION audit_events_are_append_only();

-- TRUNCATE fires no row triggers, so the one above never sees it. This is the
-- difference between "you cannot delete these rows" and "you cannot empty this
-- table", and only the second is the claim the module docstring makes.
DROP TRIGGER IF EXISTS audit_events_no_truncate ON audit_events;
CREATE TRIGGER audit_events_no_truncate
    BEFORE TRUNCATE ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION audit_events_are_append_only();

-- Keep updated_at honest without the application having to remember.
CREATE OR REPLACE FUNCTION sessions_touch_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sessions_touch ON sessions;
CREATE TRIGGER sessions_touch
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION sessions_touch_updated_at();
