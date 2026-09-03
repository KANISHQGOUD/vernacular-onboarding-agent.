"""Environment configuration that would otherwise fail quietly.

This file exists because of one bug. `DATABASE_URL` sat in `.env` as an empty
value, `skipif` could not tell that apart from "not configured", and the whole
Postgres layer reported as 14 skipped tests for an entire commit while the
summary read "822 passed" and looked healthy.

Nothing here touches a database. These are assertions about whether the suite
is testing what it claims to be testing, which is why they must run even --
especially -- when the database tests do not.
"""

from __future__ import annotations

import os


def test_a_configured_database_url_is_not_blank():
    """Absent is a choice. Present-but-empty is a mistake.

    `os.getenv(name)` and `os.getenv(name, "")` differ in exactly the way that
    matters: absent gives None, present-but-empty gives "". The Postgres suite
    passes a "" default, which collapses the two -- so this is the one place
    that looks at the difference.

    A failing test rather than an import-time raise, on purpose: a collection
    error halts the entire pytest session, and a misconfigured optional
    database should not be able to take away an offline suite that needs no
    keys and no network to run.
    """
    configured = os.getenv("DATABASE_URL")

    if configured is None:
        return  # not configured at all: running offline, on purpose

    assert configured.strip(), (
        "DATABASE_URL is present but empty, so tests/test_postgres.py skips\n"
        "silently and the suite still reports all green.\n\n"
        "Give it a value:\n"
        "  DATABASE_URL=postgresql://postgres:dev@localhost:5432/onboarding\n\n"
        "or delete the line entirely to run offline on purpose."
    )
