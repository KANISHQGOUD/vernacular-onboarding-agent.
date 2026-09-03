"""Talk to the agent from a terminal.

    .venv\\Scripts\\python.exe scripts/demo.py            interactive
    .venv\\Scripts\\python.exe scripts/demo.py --scripted  no API key needed

Interactive mode needs GOOGLE_API_KEY in .env. Scripted mode runs a fixed
conversation through the same loop with a fake classifier, so the state machine
can be demonstrated with no key, no network, and no quota.

Every turn prints the state before and after, so the machine is visible rather
than implied -- the point of the project is that the path is deterministic, and
a demo that hides the path demonstrates the wrong thing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from anywhere: `python scripts/demo.py` puts scripts/ on the path, not
# the project root, so `app` would not be importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from app.audit import AuditKind, InMemoryAuditSink
from app.conversation import Conversation
from app.flow.actions import Actions
from app.flow.machine import Machine
from app.flow.states import Event, State
from app.languages import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from app.llm.classifier import Classification, GeminiIntentClassifier
from app.llm.client import LlmClient, api_key_available
from app.llm.phrasing import Phraser
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import KycStatus, MockKycProvider
from app.providers.payments import MockPaymentRail
from app.session import InMemorySessionStore, Session

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


class ScriptedReader:
    """A classifier with no model behind it, for the no-key demo."""

    SCRIPT: list[tuple[str, Classification]] = [
        ("hindi mein baat kijiye", Classification(Event.LANGUAGE_CHOSEN, language="hi-IN")),
        ("haan ji theek hai", Classification(Event.CONSENT_GIVEN)),
        ("mera naam Asha Devi hai", Classification(Event.DATA_PROVIDED, "full_name", "Asha Devi")),
        ("1990-01-01", Classification(Event.DATA_PROVIDED, "date_of_birth", "1990-01-01")),
        ("14 Gandhi Road, Nashik", Classification(Event.DATA_PROVIDED, "address", "14 Gandhi Road, Nashik")),
        ("main darzi hoon", Classification(Event.DATA_PROVIDED, "occupation", "tailor")),
    ]
    # No mobile entry: screening already captured it, so COLLECT_DATA never
    # asks for it. Re-asking would be the bug the actions tests pin down.

    def __init__(self) -> None:
        self._index = 0

    def read(self, utterance: str, state: State) -> Classification | None:
        if self._index >= len(self.SCRIPT):
            return None
        _, classification = self.SCRIPT[self._index]
        self._index += 1
        return classification

    def classify(self, utterance: str, state: State):
        result = self.read(utterance, state)
        return result.intent if result else None


def build(scripted: bool) -> tuple[Conversation, InMemoryAuditSink]:
    audit = InMemoryAuditSink()
    machine = Machine(audit)

    # First poll returns PENDING, so the demo actually shows the suspension
    # rather than blowing through it. That pause is the whole regulatory
    # story: the agent legally cannot do the identity check, so it stops.
    kyc = MockKycProvider(outcomes=[(KycStatus.PENDING, None)])

    actions = Actions(
        machine=machine,
        kyc=kyc,
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )

    if scripted:
        return Conversation(actions, ScriptedReader(), Phraser(client=None)), audit

    client = LlmClient()
    print(f"{DIM}model: {client.model}{RESET}")
    return (
        Conversation(actions, GeminiIntentClassifier(client), Phraser(client=client)),
        audit,
    )


def show(reply, before: State, elapsed_ms: float | None = None) -> None:
    arrow = f"{before} -> {reply.state}" if reply.moved else f"{before} (unchanged)"
    timing = f"  {elapsed_ms:.0f}ms" if elapsed_ms is not None else ""
    print(f"{DIM}  [{arrow}]{timing}{RESET}")
    print(f"{BOLD}  agent:{RESET} {reply.text}\n")


def run_scripted() -> None:
    conversation, audit = build(scripted=True)
    session = Session(session_id="demo")

    print(f"\n{BOLD}Scripted demo -- no API key, no network.{RESET}\n")
    reply = conversation.open(session)
    show(reply, State.START)

    for utterance, _ in ScriptedReader.SCRIPT:
        before = session.state
        print(f"  {DIM}caller:{RESET} {utterance}")
        show(conversation.respond(session, utterance), before)
        if session.state in {State.COMPLETED, State.DECLINED, State.HUMAN_HANDOFF}:
            break

    # The handoff, made visible. The session is suspended, gets written to
    # storage, and is picked back up as a different object entirely -- which
    # is what happens when someone finishes video KYC hours later.
    if session.state is State.KYC_PENDING:
        store = InMemorySessionStore()
        store.save(session)
        print(f"{DIM}  --- applicant leaves for the video call ---{RESET}")
        print(f"{DIM}  --- session written to storage, process could end here ---{RESET}")

        session = store.load(session.session_id)
        print(f"{DIM}  --- hours later: reloaded from storage, resuming ---{RESET}\n")

        before = session.state
        show(conversation.resume(session), before)

    summarise(session, audit)


# A fixed conversation, replayed against the real model. Hindi and Hinglish
# input, but the agent answers in English on purpose: phrasing in English costs
# no API calls, so the whole per-minute budget goes to real classification --
# which is the part that cannot be cached and the part worth verifying.
AUTO_SCRIPT: list[str] = [
    "english theek rahega",
    "haan ji, mujhe koi dikkat nahi hai",
    "mera naam Asha Devi hai",
    "main ek janvari unnees sau nabbe ko paida hui thi",
    "main Nashik mein Gandhi Road par rehti hoon, makaan number chaudah",
    "main darzi ka kaam karti hoon",
]


def run_auto(pace: float) -> None:
    """Replay AUTO_SCRIPT against the live model, paced under the rate limit."""
    if not api_key_available():
        print("GOOGLE_API_KEY is empty. Use --scripted, or add a key to .env.")
        raise SystemExit(1)

    conversation, audit = build(scripted=False)
    session = Session(session_id="demo")

    print(f"\n{BOLD}Live run -- real classification on every turn.{RESET}")
    print(f"{DIM}pacing {pace:.0f}s between turns to stay under the free-tier limit{RESET}\n")

    show(conversation.open(session), State.START)

    timings: list[float] = []
    for index, utterance in enumerate(AUTO_SCRIPT):
        before = session.state
        print(f"  {DIM}caller:{RESET} {utterance}")

        started = time.perf_counter()
        reply = conversation.respond(session, utterance)
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        show(reply, before, elapsed)

        if session.state in {State.COMPLETED, State.DECLINED, State.HUMAN_HANDOFF}:
            break
        if index < len(AUTO_SCRIPT) - 1:
            time.sleep(pace)

    if session.state is State.KYC_PENDING:
        print(f"{DIM}  --- applicant leaves for the video call, session stored ---{RESET}")
        store = InMemorySessionStore()
        store.save(session)
        session = store.load(session.session_id)
        print(f"{DIM}  --- hours later: reloaded, resuming (no model call) ---{RESET}\n")
        before = session.state
        show(conversation.resume(session), before)

    if timings:
        ordered = sorted(timings)
        print(f"{BOLD}turn latency:{RESET} median {ordered[len(ordered) // 2]:.0f}ms, "
              f"min {ordered[0]:.0f}ms, max {ordered[-1]:.0f}ms")
    print(f"{BOLD}phrasing:{RESET}    {conversation.phraser.stats()}")
    summarise(session, audit)


def run_interactive() -> None:
    if not api_key_available():
        print("GOOGLE_API_KEY is empty. Use --scripted, or add a key to .env.")
        raise SystemExit(1)

    conversation, audit = build(scripted=False)
    session = Session(session_id="demo")

    print(f"\n{BOLD}Speak to the agent. Ctrl-C to stop.{RESET}")
    print(f"{DIM}languages: {', '.join(SUPPORTED_LANGUAGES)}{RESET}\n")

    reply = conversation.open(session)
    show(reply, State.START)

    while session.state not in {State.COMPLETED, State.DECLINED, State.ABANDONED, State.HUMAN_HANDOFF}:
        try:
            utterance = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not utterance:
            continue

        before = session.state
        started = time.perf_counter()
        reply = conversation.respond(session, utterance)
        show(reply, before, (time.perf_counter() - started) * 1000)

    summarise(session, audit)


def summarise(session: Session, audit: InMemoryAuditSink) -> None:
    trail = audit.for_session(session.session_id)
    moves = [e for e in trail if e.kind is AuditKind.TRANSITION_ACCEPTED]
    refusals = [e for e in trail if e.kind is AuditKind.TRANSITION_REFUSED]

    print(f"{BOLD}final state:{RESET} {session.state}")
    print(f"{BOLD}language:{RESET}    {session.language or DEFAULT_LANGUAGE}")
    if session.applicant:
        print(f"{BOLD}collected:{RESET}")
        for name in sorted(session.applicant):
            print(f"    {name:16} {session.applicant[name]}")
    if session.account_number:
        print(f"{BOLD}account:{RESET}     {session.account_number}")
    print(f"{BOLD}audit:{RESET}       {len(trail)} records, {len(moves)} moves, {len(refusals)} refused")
    print(f"\n{DIM}path taken:{RESET}")
    for record in moves:
        print(f"{DIM}  {record.from_state} -> {record.to_state}  ({record.event}){RESET}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted", action="store_true", help="run without an API key")
    parser.add_argument("--auto", action="store_true", help="replay a fixed conversation live")
    parser.add_argument("--pace", type=float, default=13.0, help="seconds between live turns")
    args = parser.parse_args()

    if args.scripted:
        run_scripted()
    elif args.auto:
        run_auto(args.pace)
    else:
        run_interactive()
