# Vernacular onboarding agent

A voice agent that helps someone open **and actually start using** a bank
account, speaking to them in their own Indian language.

---

## The problem it is aimed at

SBI has over 50 crore customers and roughly 9.6 crore app users. The gap is not
people who were never signed up — it is people holding an account they have
never touched. Onboarding is form-filling in English; a large share of those
customers are rural, older, or do not read English. SBI's answer is to hire
10,000 staff to sit beside people and fill the forms in.

Acquisition is not the bottleneck. **Activation is.**

So the metric this project optimises is **first-transaction completion rate,
not accounts opened**. An applicant who reaches "account created" and stops has
produced exactly the dormant account the exercise was meant to prevent — and
the state machine says so out loud: reaching `ACCOUNT_CREATE` is not a terminal
state, and quitting there is recorded as an abandonment, not a success.

## The one design claim

**It is a state machine that uses an LLM, not an LLM agent that improvises.**

The machine owns every transition. The model does two narrow jobs: reading what
someone meant, and choosing wording. It cannot decide what happens next.

That is not a stylistic preference. In a regulated flow, a model that can
choose its own next action can choose to mention a loan, and that is a
mis-selling incident. So the capability is removed rather than discouraged:

- **Transitions are data, not `if` statements.** Illegal is *absent from the
  table*, which means the tests enumerate the complement instead of
  hand-writing refusals.
- **Tools are bound per state.** `open_account` exists in exactly one state.
  There is no cross-sell tool anywhere in the registry — you cannot prompt your
  way to one, because there is nothing to call.
- **Events are split into intents and facts.** A model may say "they seem to
  have agreed". Only a provider response can say "KYC passed". A test parses
  the AST to enforce it.
- **The regulatory boundary has three independent walls.** The map says
  `ACCOUNT_CREATE` is reachable only via `KYC_VERIFIED`; the intent layer says
  no model may emit `KYC_VERIFIED`; and the KYC provider has no verify method
  at all. You cannot write the bypass, because the method does not exist.

## Things that only showed up by running it

The interesting part of this project is not the architecture diagram. It is
what happened when each layer met reality.

**The audit trail dropped 2.25% of events under load.** Sequence numbers were
allocated optimistically with a capped retry. Single-threaded tests passed
forever. Twelve concurrent writers lost 27 of 1200 events — and the loss rate
rose exactly when the system was busiest. Replaced with a transaction-scoped
advisory lock: zero lost, and ~30% faster because losing writers stopped
burning round-trips on doomed inserts.

**A transaction-scoped lock turned out to be load-bearing.** Once connections
were pooled, switching it to a session-scoped lock made the suite hang
indefinitely — a leaked lock poisons a pooled connection for every later
borrower. It was the right choice for a reason that only became visible two
commits later.

**"Append-only" was one statement away from false.** The trigger rejected
`UPDATE` and `DELETE`. `TRUNCATE` fires neither, so one ordinary statement
emptied the table while the trigger watched.

**Erasure is SQL-level, not disk-level.** Verified with `pageinspect`: after
erasing an applicant, a marker planted in their record was unreachable by SQL
and still sitting in the heap page — and still there after a plain `VACUUM`,
which marks space reusable without zeroing it. The honest claim is "erased from
the database", and the README says so rather than the code implying more.

**Two SDKs disagreed with their own live APIs.** A Gemini model the docs
offered was retired; a Sarvam TTS model the SDK's type hints still advertise
returns 400 "deprecated", and the speaker it lists *first* is invalid for the
current model. Everything version-shaped in this project was measured by
calling it.

**Speech recognition does not fail loudly.** It returns a plausible sentence
with no error. "नहीं" heard as "हाँ" at the consent gate produces a
lawful-looking record of a consent nobody gave. So confidence travels with
every transcript, and the acceptable floor is **per state** — 0.85 at consent,
0.35 while collecting a name. That looks backwards until you notice it tracks
the *cost of acting on a mishearing*: a wrong name is read back and corrected,
a wrong consent is caught by nobody. A strict floor on names would punish
unusual names and heavy accents — precisely the people this is for.

## Try it

Requires Python 3.13 and, optionally, Docker.

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
.venv\Scripts\python.exe -m pytest -q
```

**931 tests, no API key and no network required.** Everything external is
behind a seam with an offline implementation, which is the point.

To talk to it, copy `.env.example` to `.env`, add a free
[Gemini key](https://aistudio.google.com/apikey), and:

```bash
.venv\Scripts\python.exe -m uvicorn app.web.server:app --port 8000
```

Then open <http://localhost:8000> in Chrome and press **Speak**.

For a real Indic voice rather than the browser reading Devanagari in an English
one, add a [Sarvam key](https://dashboard.sarvam.ai) and start it with
`SARVAM_TTS=1`. Synthesis is cached on disk, so repeated phrases cost nothing
after the first time.

Durable sessions need Postgres; see [CLAUDE.md](CLAUDE.md) for the container.
Without it the app runs on in-memory storage and a restart drops live sessions.

## What is real and what is not

**Real:** the state machine, the versioned eligibility rules, the append-only
audit trail in Postgres, the intent/fact boundary, the confidence policy, the
web layer, and Indic speech in and out.

**Mocked:** the video-KYC vendor, core banking, and the payment rail. Not
oversight — these are unobtainable as an individual developer. They sit behind
protocols with scripted implementations, which is why the KYC retry path and
the suspend/resume handoff are testable at all.

**Not done:** nobody who is not me has used it yet. The design argues that
activation is the bottleneck; proving that needs a drop-off study with real
non-English-reading users, and until that exists this is a well-tested
argument rather than a validated one. Turn-taking (voice activity detection)
is also unbuilt, so the demo is push-to-talk.

## Where to read next

[CLAUDE.md](CLAUDE.md) has the working notes: architecture rationale, measured
API facts, the machine-specific traps, and the current open list. Every count
in it was read out of the code rather than remembered.

The commit messages are written to be read. Most of them explain a decision and
what it cost.
