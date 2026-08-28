# Vernacular onboarding agent

A voice agent that helps people open and actually start using a bank account,
speaking to them in their own Indian language.

SBI has 50+ crore customers and roughly 9.6 crore app users. The rest hold an
account they never touch. Onboarding is form-filling in English; many customers
are rural, older, or do not read English, and SBI's answer is hiring 10,000
staff to sit with people. Acquisition is not the bottleneck. **Activation is.**

**Success metric: first-transaction completion rate, not accounts opened.**
An applicant who reaches ACCOUNT_OPENED and stops has produced exactly the
dormant account this exists to prevent, and the state machine says so.

> These notes live in the repo on purpose. They used to live outside it and
> drifted — state counts, test counts, and a section about an SDK the project
> had already migrated off. Anything here that a program can check should be
> re-checked rather than trusted; the commands to do so are below.

---

## The architecture — do not undo these

**A state machine that uses an LLM, not an LLM agent that improvises.**

- `app/flow/states.py` — states, events, tools and transitions as **data, not
  if-statements**. Illegal is *absent from the table*, so the tests enumerate
  the complement rather than hand-writing refusals. Adding one state generates
  its own refusal cases.
- **Per-state tool binding.** `open_account` exists in exactly one state,
  reachable only via `KYC_VERIFIED`. No cross-sell tool exists anywhere in the
  registry — the risk is closed by absence, not by asking a model to behave.
- `app/flow/intents.py` — events split into **USER-INTENT** (what they meant)
  and **FACT** (what happened). A model may emit only intents; `actions.py`
  emits only facts, minted from real provider responses. Enforced by a test
  that parses the AST.
- **The regulatory boundary, three independent walls.** The map says
  ACCOUNT_CREATE is reachable only via `KYC_VERIFIED`; `intents.py` says a
  model cannot emit `KYC_VERIFIED`; `providers/kyc.py` has no verify method at
  all. You cannot write the bypass, because the method does not exist.
- **`KYC_PENDING` has no timeout.** It can last hours. Nothing times any state
  out any more — see *lapsing* below.
- **Eligibility rules are pure and versioned** (`2026-08-13.v1`), evaluated
  without short-circuit, with an explicit `as_of` and never a clock, so
  decisions replay years later.
- **Money is integer paise, never floats.**
- **`session.applicant` is regulated PII; `session.notes` is operational
  metadata.** Split so `erase_personal_data()` deletes the person and keeps
  the trail.
- **The audit trail is append-only, enforced by the database** — a trigger
  rejects UPDATE, DELETE *and* TRUNCATE for every role including the owner.
  `Machine.record()` writes non-transition facts and refuses the transition
  kinds outright: only `fire()` may say a session moved.
- **`fire()` audits before it mutates.** A failed audit write leaves the
  session where it was. Fail closed: never occupy a state the trail cannot
  explain.

### Lapsing is not a transition

Going quiet does **not** move the session. Someone who stops answering in
COLLECT_DATA is still in COLLECT_DATA, so resuming needs no edge back in.

This was tried the other way first. Making a `TIMED_OUT` state resumable means
one guarded edge per state a timeout can occur in — nine new entrances into
ELIGIBILITY, COLLECT_DATA and ACCOUNT_CREATE, each asserted to have exactly
one. That turns "illegal is absent from the table" into "illegal unless a
guard says otherwise", on the entrance that opens a bank account. Not worth it.

`Actions.record_lapse()` writes the fact to the trail and counts it in
`session.lapses`, which survives both the resume and `erase_personal_data()`.

### How sure we have to be before acting on speech

`app/flow/hearing.py`. The floor tracks **the cost of acting on a mishearing,
not the importance of the field**:

| state | floor | why |
|---|---|---|
| `CONSENT` | 0.85 | caught by nobody; everything after is unlawful processing |
| `FUNDING`, `FIRST_TRANSACTION` | 0.80 | money, spoken as numbers |
| `LANGUAGE_SELECT` | 0.60 | wrong language is loud and recoverable |
| `COLLECT_DATA` | 0.35 | read back and corrected anyway |
| everything else | 0.50 | |

`COLLECT_DATA` is low deliberately: a strict floor on names punishes unusual
names, regional pronunciations and heavy accents — precisely the people this
product is for. A low-confidence turn never reaches the classifier; asking a
model about a transcript you already doubt buys a confident reading of a guess
and spends one request from a 20/day quota.

---

## Current state

Branch `master`, 24 commits, tree clean.

**944 tests pass, ~6s.** 922 need no network and no keys; 22 need
Postgres. One more is skipped by default: the live Sarvam round trip, which
runs only under `SARVAM_LIVE=1` because it spends prepaid credit.

| | count |
|---|---|
| states / events / tools | 15 / 17 / 10 |
| transition rows / terminal states | 43 / 5 |
| speech acts / audit kinds / languages | 15 / 7 / 11 |

Terminal states: `COMPLETED`, `DECLINED`, `CONSENT_REFUSED`, `ABANDONED`,
`HUMAN_HANDOFF`. Each is a different reason, because a funnel is only readable
if the exits are distinguishable.

Working end to end, verified live in a browser in Hindi:

- State machine, eligibility, mocked providers (KYC, core banking, payments)
- Gemini classifier — intent and extraction folded into one constrained call
- Phrasing layer — the machine picks the subject, the model picks the wording
- Conversation loop (`app/conversation.py`), CLI demo (`scripts/demo.py`)
- FastAPI + WebSocket + plain HTML/JS (`app/web/`, `web/index.html`)
- Postgres, verified and pooled
- Speech seam (`app/speech/`) with browser Web Speech as the free backend

---

## Running it

Always `.venv\Scripts\python.exe`. Never bare `python`.

```bash
.venv\Scripts\python.exe -m pytest -q
```

```bash
.venv\Scripts\python.exe -m uvicorn app.web.server:app --port 8000
```

Postgres (start Docker Desktop **from the Start menu** — it will not start from
a background process):

```bash
docker start onboarding-db
```

First time only:

```bash
docker run -d --name onboarding-db -p 5432:5432 -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=onboarding postgres:17
```

```bash
.venv\Scripts\python.exe scripts/db.py apply
```

`scripts/db.py` also takes `check` and `waiting`. There is no `drop`, on
purpose.

---

## This machine — these cost time to find

- Windows 11, PowerShell 5.1. No `&&` chaining; use `;` or `if ($?)`.
- **Two Pythons.** Always `.venv\Scripts\python.exe`.
- Do **not** use `activate` — execution policy is undefined for CurrentUser.
- The console mangles non-ASCII. Any script printing Indic text needs
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- **`pkill` from Git Bash does not kill Windows processes.** To stop the dev
  server:
  ```bash
  Get-NetTCPConnection -LocalPort 8000 -State Listen | %{ Stop-Process -Id $_.OwningProcess -Force }
  ```
  A "restart" that silently did not take once cost a debugging cycle —
  mismatched traceback line numbers were the tell.
- Large Python/HTML edits: bash heredocs choke on the quoting. Use the Write
  tool or a Python script.
- **`git checkout -- <file>` discards uncommitted work.** Copy the file aside
  before mutation-testing; this bit once in this repo.
- A dead database reports in ~11s, not 4.5 minutes: `CONNECT_TIMEOUT_SECONDS`
  in `app/store/postgres.py`. libpq waits forever by default.
- Naive `ch.isalnum()` filtering **destroys Indic text** — Devanagari vowel
  signs are combining marks (category Mn) and isalnum() is False for them,
  so `आशा` becomes `आश`. Strip by punctuation category instead. This looked
  exactly like a vendor accuracy problem for one debugging round.
- Port 5432 is this project. 54321–54327 is the `barkeep` Supabase stack from
  another project — do not disturb it.

---

## Gemini — measured against the live API, not from docs

`google-genai` **2.18.1**. Run `scripts/introspect_sdk.py` after any upgrade;
it re-measures the surface. Re-verify anything here that matters — the API
moves faster than this file.

- Model **`gemini-3.5-flash`** (`DEFAULT_MODEL` in `app/llm/client.py`).
  `gemini-2.5-flash` is retired; `gemini-3.7-flash` exists but the free tier
  rejects it.
- `thinking_level=MINIMAL` works on 3.5-flash, 400s on 3.7-flash.
- **Quota: 5 requests/minute and 20 per DAY**, both per project **per model**.
  Twenty a day is about three conversations. Switching `GEMINI_MODEL` moves to
  a fresh bucket — that is why it is an env var.
- Latency varies with congestion: 1.2s median on a good run, 3.4s on a bad
  one. Do not redesign on one measurement.
- Structured output is strong: a date spoken entirely in Hindi words
  ("ek janvari unnees sau nabbe") extracted as `1990-01-01`, no date parser.

## Speech

`app/speech/provider.py` is the seam; `MockSpeech` keeps the suite offline.
The protocol centres on `Transcript` (text + language + **confidence**) rather
than `transcribe(audio) -> str`, because with browser speech the server never
receives audio — and confidence is the one signal a text-only channel throws
away. Recognition does not fail loudly; it returns a plausible sentence.

**Browser Web Speech is the current backend.** Free, no key, no install.
Measured on this machine: 8 voices installed, **zero Indic**. `en-IN` exists
(Heera, Ravi); Hindi, Tamil, Bengali and the rest do not, and Windows does not
ship them. So:

- **STT can stay free** — Chrome sends audio to Google's servers, which handle
  `hi-IN`.
- **TTS cannot** — it needs local OS voices no caller can be assumed to have.

Awkward, because TTS is roughly 85% of Sarvam's per-conversation cost: the
free option covers the cheap half and fails on the expensive one. That is the
argument for self-hosting IndicF5.

### Sarvam

`app/speech/sarvam.py`, SDK `sarvamai` **0.1.31**. Working — a Hindi sentence
round-trips (synthesise, then transcribe) with only an added danda.

**The SDK's `Literal[...]` lists disagree with the live API in both
directions.** All three found by calling it, none by reading:

| | |
|---|---|
| `bulbul:v2` | SDK offers it; API returns 400 "has been deprecated" |
| `anushka` | SDK lists it first; API rejects it for `bulbul:v3` |
| `niharika` | API offers it for v3; SDK has never heard of it |

Verified defaults: **`bulbul:v3`**, **`saaras:v4`**, speaker **`ritu`**. Since
v2 is gone, TTS really costs ₹30/10K chars, not the ₹15 the pricing page lists
for v2 — so ₹100 is roughly 33K characters, call it 6–10 conversations rather
than 12–20.

**Sarvam spells Odia `od-IN`.** ISO 639-1, and this project, say `or-IN`. The
mapping lives in the adapter, not in `languages.py`: one vendor's spelling is
not a reason to make the shared vocabulary wrong.

**Sarvam STT returns no transcription confidence.** It returns
`language_probability`, which answers "was this Hindi" — a different question,
and feeding it to `hearing.py` would compare 0.85 against an answer to
something else. So `Transcript.confidence` is None and `reports_confidence` is
False, which means **the consent gate's protection does not exist on this
provider**. Two ways out, neither built: read the answer back at the regulated
gates, or keep browser STT (which does report confidence) and use Sarvam only
for synthesis — also the cheaper arrangement, since TTS is most of the bill.

### Synthesis is cached, because it is 92% of the bill

Measured: a 20-turn conversation is about **Rs 9.83**, of which **Rs 9.00 is
TTS**. On a one-time Rs 100 credit that is ten conversations, and this needs
demonstrating more than ten times.

Most of what the agent says is not new. Every `fallback` in `speech_acts.py`
is a fixed literal, and the consent explanation and KYC handoff are spoken in
every single call. `app/speech/cache.py` wraps any provider and keeps
synthesis on disk, keyed on provider + model + speaker + language + text.

Measured against the live vendor, the same sentence twice:

    1st call (paid)     1570.3 ms
    2nd call (cached)      9.2 ms     170x, and free

The latency matters as much as the money for a voice agent. `CachedSpeech.warm()`
pays for the fixed fallbacks once, deliberately, rather than discovering the
bill one live demo at a time.

**Transcription is never cached.** Two recordings of the same words are never
byte-identical, so it would never hit -- and a key loose enough to hit would
hand one caller's words back for another caller's audio. Silent, wrong, and in
the one direction this project cannot afford.

`.speech-cache/` is gitignored: regenerable, but only by paying again.

Live tests are **opt in**, not opt out:

```bash
SARVAM_LIVE=1 .venv\Scripts\python.exe -m pytest tests/test_speech.py
```

Gating on the key being present, the way the Postgres tests gate on
`DATABASE_URL`, would mean having a key makes every suite run cost money.

---

## Still open

- **Nothing fires a lapse.** `record_lapse()` exists and is tested, but no
  timer calls it. Needs a clock-injection decision consistent with how
  `eligibility.py` takes an explicit `as_of` rather than reading a clock.
- **Sarvam STT/TTS**, then Silero VAD for turn-taking. Both need the key.
- **The WER comparison needs recordings.** `scripts/wer.py` is built and
  tested; it has only ever run against synthesised audio, which proves the
  harness and nothing about accuracy. It needs real speakers.

  Read the transcripts, not the aggregate. Sarvam normalises as it
  transcribes — `एक जनवरी उन्नीस सौ नब्बे` comes back as `1 जनवरी 1990` — so
  the first run scored 150.6% WER for an engine that misheard nothing. An
  engine that normalises is penalised against one that does not, which means
  a naive table recommends the worse recogniser.
- **Caller utterances and account numbers go to Google.** Fine for a portfolio
  project; a real deployment needs a DPA and probably in-region hosting.
- **Erasure is SQL-level, not disk-level.** Measured with pageinspect: after
  erase and save, a marker in `applicant` was unreachable by SQL and still
  present in the heap page — and still there after a plain `VACUUM`, which
  marks space reusable without zeroing it. Only `VACUUM FULL`/`pg_repack`
  rewrite it away, and the WAL and older backups keep it longer. A real DPDP
  response needs an operational procedure, not a method call.
- **No README.** For a portfolio project that is the gap a reader hits first.

---

## How to work on this

- **Teach as you build.** One step at a time with the reasoning, not finished
  modules dumped in.
- **Verify against the real thing** rather than trusting memory — introspect
  the installed SDK, run the live API, open the browser. That has caught a
  retired model, an unknown daily quota, and two bugs that only appeared in a
  browser.
- **Mutation-test new tests.** A test that passes on the buggy code is
  worthless. Break the thing deliberately and check the right test fails.
- **Say plainly when something is unverified.** Do not let a green test count
  imply coverage that is not there — 14 Postgres tests once reported as
  *skipped* for a whole commit because `DATABASE_URL` was present but empty,
  while the summary read "822 passed". `tests/test_env_config.py` exists so
  that cannot happen again.
- **Commit only when asked.** Branch first if on master.
- **Never write the API key anywhere it was not already, and never echo it.**
