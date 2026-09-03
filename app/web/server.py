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

`audio` is bounded by the same argument and needs it more, because it is
billed per second against a prepaid balance rather than per token. Both caps,
and the terminal-state check, run *before* anything is sent to a vendor: the
order of the checks in the loop is load-bearing, not incidental.

WHAT THE SOCKET DOES NOT WRITE DOWN
-----------------------------------
Transcripts never reach a log line. A caller in COLLECT_DATA is saying their
name, address and date of birth, and application logs are outside both
`erase_personal_data()` and the deliberately PII-free audit trail -- so a
debug line echoing the transcript would put regulated data in the one place
this project has no story for erasing. Byte counts and lengths, never text.
"""

from __future__ import annotations

import base64
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
from app.flow.fields import next_field
from app.flow.hearing import expectation_for, is_confident_enough
from app.flow.turn_taking import (
    DEFAULT_PATIENCE_MS,
    interruption_matters,
    policy_for,
)
from app.flow.states import TERMINAL_STATES, State
from app.llm.classifier import GeminiIntentClassifier, OfflineClassifier
from app.llm.client import LlmClient, api_key_available
from app.languages import DEFAULT_LANGUAGE
from app.llm.phrasing import Phraser
from app.providers.core_banking import MockCoreBanking
from app.providers.kyc import MockKycProvider
from app.providers.payments import MockPaymentRail
from app.speech import Audio, CachedSpeech, SarvamSpeech, SpeechProvider
from app.web.registry import SessionRegistry

# The CLI loaded this and the server did not, so uvicorn came up with no key
# and every socket died on the first utterance. Loaded at import, before
# anything checks whether a key exists.
load_dotenv()

log = logging.getLogger(__name__)

# Longer than any real spoken turn, short enough that it cannot be used to
# inflate a prompt. A caller who needs more than this is not talking.
MAX_UTTERANCE_LENGTH = 500

# The same bound for audio, derived rather than guessed. The page records for
# at most DEFAULT_PATIENCE_MS before its hard stop, and Opus in a WebM
# container tops out around 128 kbps in a browser -- so the longest turn our
# own front end can produce is about 192 KB. The doubling is headroom for a
# browser that encodes fatter than Chrome does.
#
# Deliberately loose. This is a ceiling on abuse, not a fit to real speech: a
# cap tuned to the exact bitrate would start refusing honest callers on some
# browser nobody tested, which is the failure this product can least afford.
MAX_AUDIO_BYTES = 2 * (DEFAULT_PATIENCE_MS // 1000) * 128_000 // 8

# Checked against the field as it arrives. Base64 is four characters per three
# bytes, and decoding first would mean allocating the thing being refused.
MAX_AUDIO_BASE64 = MAX_AUDIO_BYTES * 4 // 3 + 4

# 1008 is "policy violation" -- the right answer to a socket that failed
# authorisation, and deliberately the same answer for every reason it failed.
CLOSE_UNAUTHORISED = 1008

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"

# Read once at import: a flag that changed under a running server would
# put the page and the socket into different modes mid-conversation.
SERVER_STT = os.getenv("SARVAM_STT", "").strip() == "1"


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


def build_speech() -> SpeechProvider | None:
    """Server-side synthesis, if it has been asked for and paid for.

    OFF unless SARVAM_TTS=1, and that default is about money rather than
    taste. Sarvam credit is a one-time prepaid balance, so a server that
    synthesised every reply the moment a key existed would spend it on
    development runs nobody listened to. Turning it on is a decision.

    THE SPLIT, AND WHY IT IS NOT ARBITRARY
    --------------------------------------
    Recognition stays in the browser. Synthesis comes from here. Each half
    goes where it is actually better:

        STT   the browser is free, and it REPORTS CONFIDENCE. Sarvam does not
              -- see app/speech/sarvam.py -- and hearing.py's consent floor is
              worthless against a provider that returns nothing to compare.
        TTS   the browser cannot do it at all. Windows ships no Indic voices,
              so `speechSynthesis` reads Devanagari in an English one, which
              is unusable for the person this product exists for.

    So the arrangement that is cheapest is also the one that keeps the consent
    gate working. That is a coincidence, but a welcome one.
    """
    if os.getenv("SARVAM_TTS", "").strip() != "1" and os.getenv("SARVAM_STT", "").strip() != "1":
        return None

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        log.warning("SARVAM_TTS=1 but SARVAM_API_KEY is empty -- no server-side voice")
        return None

    # Cached, always. Every fallback is a fixed literal and the consent notice
    # is spoken in every call; paying twice for identical audio is a choice.
    # (Only synthesis is cached -- see app/speech/cache.py on why transcription
    # never can be.)
    log.info("server-side speech: Sarvam, cached")
    return CachedSpeech(inner=SarvamSpeech(api_key=key))


def server_transcription_enabled() -> bool:
    """Whether the page should send audio instead of its own transcript.

    Chrome's SpeechRecognition is free and reports confidence, which is why it
    is the default. But it depends on Google's speech service being reachable
    from the browser, and when that fails it fails SILENTLY -- no error, no
    results, a microphone that is plainly working and a page that receives
    nothing. Observed on a real machine, which is why this exists.

    Sarvam is the fallback and, for this product, arguably the better path
    anyway: it is trained on Indian accents and code-mixing, where Chrome's
    generic hi-IN is weakest. The cost is real -- roughly Rs 0.008 per second
    of audio -- so it is opt-in rather than automatic.
    """
    return os.getenv("SARVAM_STT", "").strip() == "1"


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
        open_pool,
    )

    # Reachable? A pool opens lazily enough that an unreachable database
    # becomes a 500 on the first session instead of a problem at startup --
    # and a 500 on session creation means no socket, which means the caller
    # cannot say anything at all. Everywhere else here a failing dependency
    # degrades rather than removing the capability; storage should too.
    try:
        pool = open_pool(url)
        with pool.connection() as probe:
            probe.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - any failure to reach it means the same
        log.warning(
            "DATABASE_URL is set but the database is unreachable -- falling "
            "back to memory. Sessions will not survive a restart. Start it "
            "with: docker start onboarding-db",
            exc_info=True,
        )
        return SessionRegistry(), InMemoryAuditSink()

    log.info("using Postgres for sessions and the audit trail")
    # One pool, shared by all three stores. Three pools would triple the idle
    # connections for no benefit -- they are backed by the same database and
    # a connection does not care which store borrowed it.
    registry = SessionRegistry(
        store=PostgresSessionStore(pool),
        tokens=PostgresTokenStore(pool),
    )
    return registry, PostgresAuditSink(pool)


def create_app(
    registry: SessionRegistry | None = None,
    conversation: Conversation | None = None,
    audit: InMemoryAuditSink | None = None,
    speech: SpeechProvider | None = None,
) -> FastAPI:
    """Everything injectable, so tests never touch the network."""
    if registry is None and audit is None:
        registry, audit = build_storage()
    audit = audit or InMemoryAuditSink()
    registry = registry or SessionRegistry()
    conversation = conversation or build_conversation(audit)
    speech = speech if speech is not None else build_speech()

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
        await socket.send_json(_payload(reply, loaded, speech))

        try:
            while True:
                message = await socket.receive_json()

                # First, because everything below it spends money. A finished
                # session has nothing to say, and a client replaying its last
                # turn into one would otherwise pay a vendor to transcribe
                # audio we then throw away with "session is finished".
                if loaded.state in TERMINAL_STATES:
                    await socket.send_json({"error": "session is finished"})
                    continue

                utterance = (message or {}).get("utterance", "")

                # Audio instead of a transcript, when the page could not get
                # one itself. Transcribed here rather than in the browser, so
                # a machine whose Chrome cannot reach Google's speech service
                # is still able to hold a conversation.
                clip = (message or {}).get("audio")

                if clip is not None and not isinstance(clip, str):
                    await socket.send_json({"error": "audio not usable"})
                    continue

                if clip and len(clip) > MAX_AUDIO_BASE64:
                    await socket.send_json({"error": "audio too long"})
                    continue

                if clip and speech is not None and not utterance.strip():
                    try:
                        heard = speech.transcribe(
                            Audio(
                                data=base64.b64decode(clip),
                                media_type=(message or {}).get(
                                    "audio_type", "audio/webm"
                                ),
                            ),
                            loaded.language or DEFAULT_LANGUAGE,
                            # The machine knows it is asking for a name rather
                            # than for a sentence, so the recogniser can be
                            # told. See app/flow/hearing.py.
                            expectation_for(
                                loaded.state, next_field(loaded.missing_fields)
                            ),
                        )
                        utterance = heard.text
                        # Sizes, not words. See the module docstring: this is
                        # a name or an address, and a log file is the one
                        # store erase_personal_data() cannot reach.
                        log.info(
                            "transcribed %s bytes -> %s chars",
                            len(clip),
                            len(utterance),
                        )
                    except Exception:  # noqa: BLE001 - a vendor failing is not fatal
                        log.warning("server-side transcription failed", exc_info=True)
                        await socket.send_json(
                            {"error": "could not hear that, please try again"}
                        )
                        continue

                if not isinstance(utterance, str) or not utterance.strip():
                    await socket.send_json({"error": "empty utterance"})
                    continue

                if len(utterance) > MAX_UTTERANCE_LENGTH:
                    await socket.send_json({"error": "utterance too long"})
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
                # Barge-in, where it is material. Recorded before the turn
                # is processed, so the note lands against the state they
                # interrupted rather than the one they end up in.
                if message.get("interrupted") and interruption_matters(loaded.state):
                    conversation.actions.record_interruption(loaded)

                if not is_confident_enough(_confidence(message), loaded.state):
                    reply = conversation.misheard(loaded)
                    registry.save(loaded)
                    await socket.send_json(_payload(reply, loaded, speech))
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
                await socket.send_json(_payload(reply, loaded, speech))

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


def _spoken(reply, session, speech: SpeechProvider | None) -> str | None:
    """The reply as audio, base64, when there is a provider to make it.

    Failure here is never fatal. If synthesis raises -- quota gone, network
    dropped, vendor having a bad afternoon -- the turn still returns text and
    the page falls back to the browser's own voice. An agent that goes silent
    because a billing balance ran out has hung up on somebody halfway through
    opening a bank account.
    """
    if speech is None or not reply.text:
        return None

    try:
        audio = speech.synthesize(reply.text, session.language or DEFAULT_LANGUAGE)
    except Exception:  # noqa: BLE001 - a vendor failure must not end the call
        log.warning("synthesis failed; falling back to the browser voice", exc_info=True)
        return None

    return base64.b64encode(audio.data).decode("ascii")


def _payload(reply, session, speech: SpeechProvider | None = None) -> dict:
    return {
        "text": reply.text,
        "state": str(reply.state),
        "moved": reply.moved,
        "understood": reply.understood,
        "finished": session.state in TERMINAL_STATES,
        # So the page knows which voice to speak the reply in. The machine
        # picks the language; the browser does not get to guess it.
        "language": session.language or DEFAULT_LANGUAGE,
        # Present only when server-side synthesis is on. The page prefers it
        # and falls back to speechSynthesis when it is absent.
        "audio": _spoken(reply, session, speech),
        # How patient to be before deciding they have stopped talking. Sent
        # per turn because it changes with the state, and the page should not
        # have to know why -- see app/flow/turn_taking.py.
        "turn": policy_for(session.state),
        # Whether the page should send audio for us to transcribe, rather than
        # relying on Chrome's recogniser. The server knows; the page should
        # not have to be configured separately and drift.
        "server_stt": SERVER_STT,
    }


app = create_app()
