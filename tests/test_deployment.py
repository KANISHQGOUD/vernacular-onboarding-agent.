"""What changes when strangers can reach the server.

Pinned here: the access code and daily cap on session creation, the strict
mode that refuses to start a public server misconfigured, and the frame limit
that bounds memory per WebSocket message. None of it is visible on a laptop,
which is exactly why it needs tests rather than a checklist.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.web import __main__ as entrypoint
from app.web import server
from app.web.gate import (
    MIN_ACCESS_CODE_LENGTH,
    Gate,
    Refused,
    public_deployment_problems,
)
from app.web.registry import SessionRegistry
from app.audit import InMemoryAuditSink
from app.web.server import (
    MAX_AUDIO_BASE64,
    WS_MAX_SIZE,
    build_storage,
    create_app,
)
from tests.test_web import build_client

CODE = "a-long-shared-code"

COMPLETE_ENV = {
    "ACCESS_CODE": CODE,
    "MAX_SESSIONS_PER_DAY": "5",
    "DATABASE_URL": "postgresql://example/onboarding",
    "GOOGLE_API_KEY": "present",
}


class Clock:
    def __init__(self, day: date) -> None:
        self.day = day

    def __call__(self) -> date:
        return self.day


# ---------------------------------------------------------------------------
# The gate on POST /api/session
# ---------------------------------------------------------------------------


def test_no_code_and_no_cap_admits_everyone():
    """Local development is unchanged: nothing configured, nothing refused."""
    gate = Gate()
    for _ in range(100):
        gate.admit("")


def test_a_missing_or_wrong_code_is_refused_over_http():
    client, _ = build_client(gate=Gate(access_code=CODE))

    missing = client.post("/api/session")
    wrong = client.post("/api/session", headers={"X-Access-Code": "nope"})
    right = client.post("/api/session", headers={"X-Access-Code": CODE})

    assert missing.status_code == wrong.status_code == 401
    assert "token" not in missing.json()
    assert right.status_code == 200
    assert right.json()["token"]


def test_a_refused_request_creates_no_session():
    client, registry = build_client(gate=Gate(access_code=CODE))
    before = len(registry.store._rows)

    client.post("/api/session", headers={"X-Access-Code": "nope"})

    assert len(registry.store._rows) == before


def test_the_cap_refuses_once_the_day_is_used_up():
    client, _ = build_client(gate=Gate(daily_cap=2))

    statuses = [client.post("/api/session").status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_wrong_codes_do_not_use_up_the_day():
    """Somebody guessing codes must not spend the allowance of the people who
    were actually given one. The code is checked first, and a refusal is
    never counted."""
    gate = Gate(access_code=CODE, daily_cap=1)

    for _ in range(50):
        with pytest.raises(Refused) as refused:
            gate.admit("guess")
        assert refused.value.status == 401

    gate.admit(CODE)  # the one real session of the day still gets in


def test_the_cap_resets_when_the_day_changes():
    """The date is read per request. A gate that read it once at startup
    would stop resetting after the first night."""
    clock = Clock(date(2026, 11, 19))
    gate = Gate(daily_cap=1, today=clock)

    gate.admit("")
    with pytest.raises(Refused):
        gate.admit("")

    clock.day = date(2026, 11, 20)
    gate.admit("")


def test_the_gate_reads_its_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", f"  {CODE}  ")
    monkeypatch.setenv("MAX_SESSIONS_PER_DAY", "7")

    gate = Gate.from_env()

    assert gate.access_code == CODE
    assert gate.daily_cap == 7


# ---------------------------------------------------------------------------
# Public mode refuses to start misconfigured
# ---------------------------------------------------------------------------


def test_a_complete_environment_has_no_problems():
    assert public_deployment_problems(COMPLETE_ENV) == []


@pytest.mark.parametrize("missing", sorted(COMPLETE_ENV))
def test_every_required_setting_is_named_when_missing(missing):
    env = {k: v for k, v in COMPLETE_ENV.items() if k != missing}

    problems = public_deployment_problems(env)

    assert len(problems) == 1
    assert missing in problems[0]


def test_every_problem_is_listed_at_once():
    """One failed start should show the whole fix, not the first item of it."""
    assert len(public_deployment_problems({})) == len(COMPLETE_ENV)


def test_a_short_access_code_is_a_problem():
    env = dict(COMPLETE_ENV, ACCESS_CODE="x" * (MIN_ACCESS_CODE_LENGTH - 1))
    assert public_deployment_problems(env)


@pytest.mark.parametrize("cap", ["0", "-1", "lots", "2.5"])
def test_a_nonsense_cap_is_a_problem(cap):
    env = dict(COMPLETE_ENV, MAX_SESSIONS_PER_DAY=cap)
    assert public_deployment_problems(env)


def test_public_mode_refuses_to_build_the_app(monkeypatch):
    for name in COMPLETE_ENV:
        monkeypatch.setenv(name, "")

    with pytest.raises(RuntimeError, match="ACCESS_CODE"):
        create_app(public=True)


def test_public_mode_is_switched_on_by_the_environment(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "1")
    for name in COMPLETE_ENV:
        monkeypatch.setenv(name, "")

    with pytest.raises(RuntimeError):
        create_app()


def test_strict_storage_will_not_fall_back_to_memory(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")

    registry, _ = build_storage(strict=False)
    assert isinstance(registry, SessionRegistry)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        build_storage(strict=True)


def test_strict_storage_refuses_an_unreachable_database(monkeypatch):
    def unreachable(url):
        raise OSError("connection refused")

    monkeypatch.setenv("DATABASE_URL", "postgresql://nowhere/onboarding")
    monkeypatch.setattr("app.store.postgres.open_pool", unreachable)

    registry, audit = build_storage(strict=False)
    assert isinstance(audit, InMemoryAuditSink)  # the laptop behaviour

    with pytest.raises(RuntimeError, match="unreachable"):
        build_storage(strict=True)


# ---------------------------------------------------------------------------
# Memory per message, and how the server is started
# ---------------------------------------------------------------------------


def test_the_frame_limit_admits_the_largest_legitimate_message():
    """The audio cap is checked AFTER the frame is read, so the frame limit
    must be above it or honest callers get a dropped socket instead of the
    polite 'audio too long'."""
    assert WS_MAX_SIZE > MAX_AUDIO_BASE64


def test_the_frame_limit_is_far_below_uvicorns_default():
    # uvicorn's own default is 16 MB per message.
    assert WS_MAX_SIZE < 1024 * 1024


def test_the_entrypoint_passes_the_derived_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    monkeypatch.setenv("PORT", "9123")

    entrypoint.main()

    assert seen["app"] == "app.web.server:app"
    assert seen["ws_max_size"] == WS_MAX_SIZE
    assert seen["workers"] == 1
    assert seen["port"] == 9123
    assert seen["proxy_headers"] is True


def test_the_health_check_answers():
    client, _ = build_client()
    assert client.get("/healthz").json() == {"ok": True}


def test_the_module_level_app_built_in_local_mode():
    """Importing the server must not need a public environment."""
    assert server.app is not None

