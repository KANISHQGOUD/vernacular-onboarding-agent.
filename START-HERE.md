# Start here

This is a working copy of the vernacular onboarding agent, taken on
2026-09-03. It has the full git history and a live `.env`.

**[README.md](README.md) is the project.** What it is, why it is built the way
it is, what the results are, and what it does not do. Read that one.
**[CLAUDE.md](CLAUDE.md)** is the working notebook: measured API facts, traps
specific to this machine, and the current open list.

This file only covers getting it running from *this folder*, because two
things do not survive a copy and two more are easy to get wrong.

---

## 1. The virtualenv did not come with it

Virtualenvs bake absolute paths into their scripts, so a copied one points at
the folder it was created in and breaks quietly. It was left behind
deliberately. Recreate it:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Always `.venv\Scripts\python.exe`. Never bare `python` — this machine has two
Pythons and the wrong one has none of these packages. Do not use `activate`;
the execution policy for CurrentUser is undefined.

## 2. Check it works before changing anything

```bash
.venv\Scripts\python.exe -m pytest -q
```

**1125 passed, 1 skipped** is the expected result, in about seven seconds.

Twenty-two of those need Postgres and will fail rather than skip if
`DATABASE_URL` is set but the database is down — that is deliberate, see
`tests/test_env_config.py`. Start it with:

```bash
docker start onboarding-db
```

The one skip is the live Sarvam round trip. It runs only under
`SARVAM_LIVE=1`, because it spends prepaid credit.

## 3. Running it — the two flags that matter

The plain command in the README works, but on this machine it produces a page
that cannot hear you and a model that has run out of budget. Both are known,
neither is a bug in the code, and both are one word to fix.

```bash
GEMINI_MODEL=gemini-3.5-flash-lite SARVAM_STT=1 .venv\Scripts\python.exe -m uvicorn app.web.server:app --port 8000
```

Then <http://localhost:8000> in **real Chrome** — an embedded or automation
browser cannot open a microphone at all, and the page will now say so.

**`SARVAM_STT=1`** — Chrome's own recogniser sends audio to Google's speech
service, and when that is unreachable it fails *silently*: no error, no
results, a microphone that is plainly working and a page that receives
nothing. That happens on this machine. This flag switches to recording in the
browser and transcribing on the server with Sarvam, which is built for exactly
this and handles the Indic languages properly. Transcription is the cheap half
of the Sarvam bill; synthesis is roughly 90% of it and stays off.

**`GEMINI_MODEL=gemini-3.5-flash-lite`** — the free tier allows **20 requests
per day per model**, and a single conversational turn costs **two** of them:
one to classify what the caller meant, one to phrase the reply. So the default
model is worth about ten turns a day, which is less than one complete journey.
Switching model moves to a fresh quota bucket; that is the entire reason the
model is an environment variable. `gemini-3.6-flash`, `gemini-3.8-flash` and
`gemini-flash-lite-latest` are also available on this key.

When the quota does run out the agent says so honestly — "something went wrong
on our side" rather than asking you to repeat yourself — and the server log
carries a `429 RESOURCE_EXHAUSTED`. Check the log before assuming the
microphone is at fault; on this machine both failed at once and each looked
like the other.

## 4. Without a microphone

The text box is not a lesser path. It goes through the identical conversation
loop and costs the same quota, so everything is testable by typing:

```
hindi → haan → Asha Devi → haan → 1990 → haan
→ 14 Gandhi Marg, Jaipur → tailor
→ haan, ek hazaar daal do → haan bhej do, pachaas rupaye
```

Two things worth watching. After `tailor` the agent should announce the
account **and stop** — the deposit and the payment are the caller's turns, not
the machine's, and that is the whole point of the product. And the final line
should say **Rs 50.00**, the amount actually asked for, not the suggested
Rs 125.50.

Then, to see where people stop:

```bash
.venv\Scripts\python.exe scripts/funnel.py
```

`dormant` is the headline — accounts opened that never made a first
transaction. Try quitting right after the account opens and watch it move.

---

## What is in this copy that is not in the repository

`.gitignore` keeps three things out of version control, and all three are
here:

| | why it is ignored | why it is here |
|---|---|---|
| `.env` | live Gemini and Sarvam API keys | so this copy runs without re-adding them |
| `corpus/*.wav` | recordings of speech; a project arguing about DPDP compliance should not ship voices | the WER harness needs them |
| `.speech-cache/` | regenerable — but only by paying for it again | 5.5 MB of already-purchased synthesis |

**Do not share this folder as-is.** `.env` contains working API keys. If you
need to hand it to someone, delete `.env`, `corpus/*.wav` and `.speech-cache/`
first — everything else is already public in the git history.

If the keys ever do escape, rotate them at
<https://aistudio.google.com/apikey> and <https://dashboard.sarvam.ai>.
