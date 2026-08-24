"""The web layer. Wraps the conversation; never bypasses it.

Every socket message ends up in exactly one call to `Conversation.respond()`.
There is no path here that fires an event, touches a provider, or decides what
to say -- if there were, the guarantees the state machine provides would stop
being guarantees, because the web layer would be a second way in.

    POST /api/session          mint a session and its bearer token
    WS   /ws                   the conversation
    GET  /api/session/{id}     state + audit trail (token required)
    GET  /                     the page

RECONNECT IS RESUME
-------------------
A dropped socket is not a lost application. The session lives in the registry,
not in the connection, so reconnecting reloads it and carries on from whatever
state it is actually in. That is the same mechanism as the KYC handoff -- an
applicant leaving for a video call and coming back hours later is just a very
long disconnection -- which is why `Conversation.resume()` needed to be a real
method rather than demo scaffolding.

SESSIONS DO NOT SURVIVE A RESTART
---------------------------------
The registry is in memory, so restarting the server drops every live session.
A browser holding an old id and token then fails authorisation and gets a 1008
close -- correctly, since that session genuinely no longer exists.

Worth stating plainly because the symptom is confusing: the page says the
session expired when the real cause is that the process did. The frontend
handles it by offering a new session rather than sitting on a dead one, and
the durable fix is the Postgres store the SessionStore protocol already exists
for. Until then, a restart means starting over.

WHAT THE SOCKET REFUSES
-----------------------
Utterances are bounded and non-empty. The text goes into a model prompt, and
an unbounded field reaching a prompt is both a cost problem and an injection
surface. The fence in classifier.py handles the second; this handles the
first, before the call is billed rather than after.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.audit import AuditKind, InMemoryAuditSink
from app.conversation import Conversation
from app.flow.actions import Actions
from app.flow.machine import Machine
from app.flow.hearing import is_confident_enough
from app.flow.states import TERMINAL_STATES, State
from app.llm.classifier import GeminiIntentClassifier, OfflineClassifier
from app.llm.client import LlmClient, api_key_available
from app.languages import DEFAULT_LANGUAGE
from app.llm.phrasing import Phraser
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import MockKycProvider
from app.providers.payments import MockPaymentRail
from app.web.registry import SessionRegistry

# The CLI loaded this and the server did not, so uvicorn came up with no key
# and every socket died on the first utterance. Loaded at import, before
# anything checks whether a key exists.
load_dotenv()

log = logging.getLogger(__name__)

# Longer than any real spoken turn, short enough that it cannot be used to
# inflate a prompt. A caller who needs more than this is not talking.
MAX_UTTERANCE_LENGTH = 500

# 1008 is "policy violation" -- the right answer to a socket that failed
# authorisation, and deliberately the same answer for every reason it failed.
CLOSE_UNAUTHORISED = 1008

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"


def build_conversation(audit: InMemoryAuditSink) -> Conversation:
    """One conversation engine, shared across sockets.

    Providers are mocked. The classifier and phraser both degrade to offline
    behaviour without a key, so the server starts either way.
    """
    machine = Machine(audit)
    actions = Actions(
        machine=machine,
        kyc=MockKycProvider(),
        core_banking=MockCoreBanking(),
        payments=MockPaymentRail(),
    )

    if api_key_available():
        client = LlmClient()
        return Conversation(actions, GeminiIntentClassifier(client), Phraser(client=client))

    # A real classifier holding None builds fine and raises on the first
    # utterance -- a missing key would become a dropped socket mid-application
    # rather than a warning at startup. The null object degrades honestly.
    log.warning("GOOGLE_API_KEY is empty -- the agent will understand nothing")
    return Conversation(actions, OfflineClassifier(), Phraser(client=None))


def build_storage() -> tuple[SessionRegistry, object]:
    """Postgres when DATABASE_URL is set, memory otherwise.

    The choice is made here and nowhere else. `SessionStore`, `TokenStore` and
    `AuditSink` have been protocols since the first commit, so everything above
    this function is identical either way -- including the state machine, which
    does not import either implementation.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        log.warning(
            "DATABASE_URL is empty -- sessions live in memory and a restart "
            "will drop every one of them"
        )
        return SessionRegistry(), InMemoryAuditSink()

    # Imported lazily so the app still starts with no psycopg installed.
    from app.store.postgres import (
        PostgresAuditSink,
        PostgresSessionStore,
        PostgresTokenStore,
    )

    log.info("using Postgres for sessions and the audit trail")
    registry = SessionRegistry(
        store=PostgresSessionStore(url),
        tokens=PostgresTokenStore(url),
    )
    return registry, PostgresAuditSink(url)


def create_app(
    registry: SessionRegistry | None = None,
    conversation: Conversation | None = None,
    audit: InMemoryAuditSink | None = None,
) -> FastAPI:
    """Everything injectable, so tests never touch the network."""
    if registry is None and audit is None:
        registry, audit = build_storage()
    audit = audit or InMemoryAuditSink()
    registry = registry or SessionRegistry()
    conversation = conversation or build_conversation(audit)

    app = FastAPI(title="Vernacular onboarding agent")
    app.state.registry = registry
    app.state.conversation = conversation
    app.state.audit = audit

    @app.post("/api/session")
    async def create_session() -> JSONResponse:
        session, token = registry.create()
        return JSONResponse(
            {
                "session_id": session.session_id,
                "token": token,
                "state": str(session.state),
            }
        )

    @app.get("/api/session/{session_id}")
    async def read_session(session_id: str, token: str = Query(default="")) -> JSONResponse:
        session = registry.open(session_id, token)
        if session is None:
            # Same answer for unknown and unauthorised, so this cannot be used
            # to discover which session ids exist.
            return JSONResponse({"error": "not found"}, status_code=404)

        interesting = (AuditKind.TRANSITION_ACCEPTED, AuditKind.TRANSITION_REFUSED)
        return JSONResponse(
            {
                "state": str(session.state),
                "language": session.language,
                "account_number": session.account_number,
                # Field NAMES only. The values are the regulated personal data
                # and have no business being served back over a socket.
                "collected": sorted(session.applicant),
                "audit": [
                    {
                        "sequence": record.sequence,
                        "kind": str(record.kind),
                        "from_state": record.from_state,
                        "to_state": record.to_state,
                        "event": record.event,
                    }
                    for record in audit.for_session(session_id)
                    if record.kind in interesting
                ],
            }
        )

    @app.websocket("/ws")
    async def talk(
        socket: WebSocket,
        session: str = Query(default=""),
        token: str = Query(default=""),
    ) -> None:
        loaded = registry.open(session, token)
        if loaded is None:
            # Accept then close, so the browser sees a close code it can report
            # rather than an opaque handshake failure.
            await socket.accept()
            await socket.close(code=CLOSE_UNAUTHORISED)
            return

        await socket.accept()

        # Opening a fresh session and resuming a suspended one are the same
        # act: run whatever the current state requires. A reconnecting client
        # is caught up without ever knowing it was away.
        reply = (
            conversation.open(loaded)
            if loaded.state is State.START
            else conversation.resume(loaded)
        )
        registry.save(loaded)
        await socket.send_json(_payload(reply, loaded))

        try:
            while True:
                message = await socket.receive_json()
                utterance = (message or {}).get("utterance", "")

                if not isinstance(utterance, str) or not utterance.strip():
                    await socket.send_json({"error": "empty utterance"})
                    continue

                if len(utterance) > MAX_UTTERANCE_LENGTH:
                    await socket.send_json({"error": "utterance too long"})
                    continue

                if loaded.state in TERMINAL_STATES:
                    await socket.send_json({"error": "session is finished"})
                    continue

                # How sure the recogniser was, when there was one. Typed input
                # sends nothing and is treated as certain, which is correct --
                # nobody mishears a keyboard.
                #
                # Client-reported, and therefore advisory rather than a
                # security control: this page could always claim 1.0, exactly
                # as it could always type the same words. It is a quality
                # signal from our own front end, not a check on an adversary.
                # Under a server-side topology the number would come from our
                # provider instead and could be trusted properly.
                if not is_confident_enough(_confidence(message), loaded.state):
                    reply = conversation.misheard(loaded)
                    registry.save(loaded)
                    await socket.send_json(_payload(reply, loaded))
                    continue

                try:
                    reply = conversation.respond(loaded, utterance)
                except Exception:
                    # Same argument as _advance in conversation.py: a caller
                    # mid-application must not lose the connection because
                    # something downstream raised. Logged with a traceback so
                    # the failure is visible; the socket stays open.
                    log.error("turn failed in %s", loaded.state, exc_info=True)
                    await socket.send_json({"error": "something went wrong, please try again"})
                    continue

                registry.save(loaded)
                await socket.send_json(_payload(reply, loaded))

        except WebSocketDisconnect:
            # Not an error. The session is already saved, and the applicant can
            # come back to it whenever they like.
            log.info("socket closed for %s in %s", session, loaded.state)

    if WEB_ROOT.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(WEB_ROOT / "index.html")

    return app


def _confidence(message) -> float | None:
    """Pull a usable confidence out of whatever the client sent.

    Anything that is not a real number in 0..1 becomes None, which
    `is_confident_enough` treats as "not reported" and lets through. A
    malformed field must not silently become a low score and lock somebody out
    of their own application -- the failure has to fall towards letting the
    turn proceed, because the alternative is an applicant who cannot speak.
    """
    raw = (message or {}).get("confidence")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value != value or not 0.0 <= value <= 1.0:  # NaN or out of range
        return None
    return value


def _payload(reply, session) -> dict:
    return {
        "text": reply.text,
        "state": str(reply.state),
        "moved": reply.moved,
        "understood": reply.understood,
        "finished": session.state in TERMINAL_STATES,
        # So the page knows which voice to speak the reply in. The machine
        # picks the language; the browser does not get to guess it.
        "language": session.language or DEFAULT_LANGUAGE,
    }


app = create_app()
