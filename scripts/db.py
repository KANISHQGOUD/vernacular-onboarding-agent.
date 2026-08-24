"""Set up and inspect the database.

    .venv\\Scripts\\python.exe scripts/db.py apply     create tables and triggers
    .venv\\Scripts\\python.exe scripts/db.py check     is it there, what is in it
    .venv\\Scripts\\python.exe scripts/db.py waiting   sessions parked at KYC

Reads DATABASE_URL from .env. There is no `drop` command on purpose: a script
that can delete an append-only audit trail is not much of an append-only audit
trail. Dropping is a deliberate act for a psql prompt, not a flag away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os

from dotenv import load_dotenv


def conninfo() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        print(
            "DATABASE_URL is empty.\n\n"
            "Start one with Docker:\n"
            "  docker run -d --name onboarding-db -p 5432:5432 \\\n"
            "    -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=onboarding postgres:17\n\n"
            "then put this in .env:\n"
            "  DATABASE_URL=postgresql://postgres:dev@localhost:5432/onboarding"
        )
        raise SystemExit(1)
    return url


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["apply", "check", "waiting"])
    args = parser.parse_args()

    from app.store.postgres import (
        PostgresSessionStore,
        apply_schema,
        describe,
        open_pool,
    )

    url = conninfo()

    if args.command == "apply":
        apply_schema(url)
        print("schema applied")
        state = describe(url)
        print(f"append-only trigger present: {state['append_only_trigger']}")
        return

    if args.command == "check":
        state = describe(url)
        for table, count in state["counts"].items():
            print(f"  {table:16} {count}")
        print(f"  {'append-only':16} {state['append_only_trigger']}")
        return

    if args.command == "waiting":
        # A one-shot script wants one connection, not a warm pool, and it has
        # to hand it back -- an unclosed pool keeps non-daemon worker threads
        # alive and the script never exits.
        pool = open_pool(url, min_size=1, max_size=2)
        try:
            waiting = PostgresSessionStore(pool).suspended_at_kyc()
        finally:
            pool.close()
        if not waiting:
            print("nobody is waiting on a video call")
            return
        print(f"{len(waiting)} session(s) parked at KYC_PENDING:")
        for session_id in waiting:
            print(f"  {session_id}")


if __name__ == "__main__":
    main()
