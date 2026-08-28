"""Where people stop.

    scripts/funnel.py                 the report
    scripts/funnel.py --since 2026-08-01
    scripts/funnel.py --json out.json

Reads the audit trail from Postgres. Needs DATABASE_URL; there is nothing to
report from an in-memory registry, because it died with the process.

This is the instrument for the user study. Ten people without it produce ten
anecdotes; with it they produce "six of ten stopped at the KYC handoff", which
is a finding you can act on.

It reads the TRAIL rather than `sessions.state`, because where somebody is now
says nothing about the journey: a session sitting in ABANDONED does not tell
you whether they gave up before consent or after their account was open, and
those are opposite problems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from app.audit import AuditEvent, AuditKind
from app.funnel import build


def fetch(conninfo: str, since: datetime | None) -> list[list[AuditEvent]]:
    """Every session's trail, in order.

    Queried here rather than through `PostgresAuditSink`, which reads one
    session at a time by design -- a reporting job that made one round trip
    per applicant would be the connection-per-call mistake again, in a
    different shape.
    """
    import psycopg

    sql = """
        SELECT session_id, sequence, kind, recorded_at,
               from_state, to_state, event, tool, detail
        FROM audit_events
        {where}
        ORDER BY session_id, sequence
    """.format(where="WHERE recorded_at >= %s" if since else "")

    with psycopg.connect(conninfo) as connection:
        rows = connection.execute(sql, (since,) if since else ()).fetchall()

    trails: dict[str, list[AuditEvent]] = defaultdict(list)
    for row in rows:
        trails[row[0]].append(
            AuditEvent(
                session_id=row[0],
                sequence=row[1],
                kind=AuditKind(row[2]),
                recorded_at=row[3],
                from_state=row[4],
                to_state=row[5],
                event=row[6],
                tool=row[7],
                detail=row[8] or "",
            )
        )
    return list(trails.values())


def report(funnel) -> None:
    if not funnel.sessions:
        print("No sessions in the trail yet.")
        print()
        print("Run one: start the server, open the page, and have a conversation.")
        return

    print(f"{funnel.sessions} session(s)")
    print()
    print(f"{'reached':<20}{'n':>5}{'of all':>9}{'lost here':>11}{'median':>9}")
    print("-" * 54)

    worst = funnel.worst_step
    for step in funnel.steps:
        took = f"{step.median_seconds:.0f}s" if step.median_seconds is not None else "-"
        lost = f"{step.lost_here}" if step.lost_here else ""
        mark = "  <- worst" if worst and step.state is worst.state else ""
        print(
            f"{str(step.state):<20}{step.reached:>5}{step.share:>8.0%}"
            f"{lost:>11}{took:>9}{mark}"
        )

    print()
    print("THE METRIC")
    print("-" * 54)
    print(f"first-transaction completion   {funnel.completion_rate:>6.0%}"
          f"   ({funnel.completed}/{funnel.sessions})")
    print(f"accounts opened                {funnel.opened_an_account:>6}")
    print(f"opened and never used          {funnel.dormant:>6}"
          "   <- the problem being solved")

    if funnel.exits:
        print()
        print("HOW THEY ENDED")
        print("-" * 54)
        for name, count in funnel.exits.most_common():
            print(f"  {name:<22}{count:>4}")

    if funnel.lapses:
        print()
        print("WHERE THEY WENT QUIET (not an ending -- the session waits)")
        print("-" * 54)
        for name, count in funnel.lapses.most_common():
            print(f"  {name:<22}{count:>4}")

    if funnel.interruptions:
        print()
        print("WHERE THEY TALKED OVER THE AGENT")
        print("-" * 54)
        for name, count in funnel.interruptions.most_common():
            print(f"  {name:<22}{count:>4}")

    if funnel.refusals:
        print()
        print("WHERE THE MACHINE SAID NO (a refused transition)")
        print("-" * 54)
        for name, count in funnel.refusals.most_common():
            print(f"  {name:<22}{count:>4}")
        print()
        print("  Refusals are a metric, not an error log. A count that climbs")
        print("  after a prompt change is the model starting to wander.")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="ISO date, e.g. 2026-08-01")
    parser.add_argument("--json", help="write the numbers here as well")
    args = parser.parse_args()

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(
            "DATABASE_URL is empty. There is nothing to report from an\n"
            "in-memory registry -- it died with the process that held it."
        )

    since = datetime.fromisoformat(args.since) if args.since else None
    funnel = build(fetch(url, since))
    report(funnel)

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "sessions": funnel.sessions,
                    "completion_rate": funnel.completion_rate,
                    "accounts_opened": funnel.opened_an_account,
                    "dormant": funnel.dormant,
                    "steps": [
                        {
                            "state": str(s.state),
                            "reached": s.reached,
                            "share": s.share,
                            "lost_here": s.lost_here,
                            "median_seconds": s.median_seconds,
                        }
                        for s in funnel.steps
                    ],
                    "exits": dict(funnel.exits),
                    "lapses": dict(funnel.lapses),
                    "interruptions": dict(funnel.interruptions),
                    "refusals": dict(funnel.refusals),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
