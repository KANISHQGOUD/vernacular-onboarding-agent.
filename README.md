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
the state machine says so out loud: an open account is not a terminal state.
The caller makes their own first deposit and first payment, and stopping
before that is counted as a dormant account, not a success.

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

**The benchmark said 150% and the engine was right.** A harness was built to
answer "which recogniser" with numbers instead of opinion. Its first run
scored Sarvam at 150.6% word error rate — on utterances where it had misheard
nothing at all. It normalises as it transcribes: *एक जनवरी उन्नीस सौ नब्बे*
comes back as `1 जनवरी 1990`, *पाँच सौ रुपये* as `₹500`. Correct, useful, and
catastrophic against a reference written the way people speak.

That is worse than a confusing number. An engine that normalises is penalised
against one that does not, so the naive table recommends the **worse**
recogniser — while for this flow the normalising one is better, because the
extraction step downstream wants `1990`. The report now prints every
disagreeing transcript rather than a percentage, so nobody can quote the
aggregate without having read them.

**The turn detector answered a cough.** Endpointing measured "how long have
they been speaking" from the first voiced frame to *now* rather than to the
last voiced frame — so a 100ms cough followed by eight seconds of silence
looked like eight seconds of speech, and the agent replied to a chair scrape.
Behind it sat a second bug: the give-up timer was unreachable once any noise
had been heard, so the same cough left the detector listening forever. Both
found by pulling the decision out of the animation loop into a pure function
and feeding it made-up numbers — no microphone involved.

**"Say that again" was the wrong answer to a vendor outage.** When Gemini
returned `503 UNAVAILABLE` mid-conversation, the classifier returned `None` —
the same value it returns when it read the caller and understood nothing — so
the agent replied *"Sorry, I didn't catch that."* A comment in the code said
those cases were "all the same to the caller". They are not. Telling somebody
to speak more clearly when the fault is entirely ours is unanswerable: no
amount of speaking clearly fixes a 503, and for an applicant who already
expects machines not to understand them it is the exact humiliation this
product exists to remove. Unreachability now raises rather than returning
`None`, gets its own speech act — *"That's not you"* — and is counted in the
funnel apart from refusals, so a vendor's bad afternoon cannot be read as
people failing.

**The machine was finishing the job for the caller.** The first deposit and
the first payment were steps the machine took automatically, so the moment an
account existed it deposited Rs 500, paid a shop Rs 125.50 and marked the
session complete — in the same turn, without asking anyone. The caller was
never even told their account had opened. Worse, "opened but never used" was
structurally zero for every real session, so the one number this project
exists to move could not move. It only ever looked right against the
synthetic journeys in the funnel tests. Those two states now wait for the
caller, and a test pins exactly which states wait for a person.

**The age gate had never seen a real age.** Eligibility runs before data
collection, on placeholder answers, so the rules decided on a hard-coded date
of birth that always passed. The engine could always decline a minor; it was
never shown one. Combined with the rule that a field accepts the caller's
answer after three failed checks, a 17-year-old who repeated their real date
of birth three times was onboarded — verified before the fix, not inferred.
The rules now run again on the date the caller actually gave. Recognising a
field gives up after three tries; the law does not.

**"Ek hazaar daal do" deposited Rs 500.** Every offline test passed, because
they fake the classifier, so both sides of that seam agreed with each other
and neither agreed with the model. Five sentences against the live model
found it in under a minute: the amount came back in a slot that the
personal-data filter correctly discards. An amount now has its own slot, and
`scripts/check_intents.py` runs those five sentences on demand, because any
prompt change can break this while the suite stays green.

**A debug line put the caller's words in the log.** Added while chasing a
microphone bug: the full transcript, at INFO, on the one store with no erasure
story. It passed 1052 tests, because nothing had ever checked application
logs. Three tests now do.

**Sarvam reports no transcription confidence**, so the per-state floors above
do nothing on that provider. Two replacements that do not depend on a vendor:
field rules (a mobile is ten digits starting 6–9; a date of birth must give an
age of 18 to 120), and reading the value back before writing it down —
*"I have your full name as Asha Devri. Is that right?"* — for the mobile,
date of birth and name. A number says the recogniser was unsure. A rule says
the answer is impossible, and the caller is the only one who knows if it is
right.

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

**1132 tests, no API key and no network required.** Everything external is
behind a seam with an offline implementation, which is the point. Another 22
run against Postgres when `DATABASE_URL` points at a live database (see
below). The live Sarvam round trip runs only under `SARVAM_LIVE=1`, because it
spends prepaid credit.

The offline suite cannot see what the real model does with a sentence. After
any change to the classifier prompt or schema, also run the live check, which
spends 5 of the free tier's 20 requests a day:

```bash
.venv\Scripts\python.exe scripts/check_intents.py
```

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

If Chrome's recogniser returns nothing on your machine (it fails silently
when it cannot reach Google's speech service), start with `SARVAM_STT=1`
instead. The page then records the audio and the server transcribes it with
Sarvam. That costs about Rs 0.008 per second, and Sarvam reports no confidence,
which is why the read-back above exists.

Durable sessions need Postgres; see [CLAUDE.md](CLAUDE.md) for the container.
Without it, or if the database cannot be reached at startup, the app warns and
runs on in-memory storage, so a restart drops live sessions.

## Putting it on a public URL

The `Dockerfile` builds the image a public demo runs, and it is strict on
purpose. Every session spends from budgets nobody can top up for free --
Gemini's free tier is 20 requests a day, about three conversations, and Sarvam
is a one-time credit -- so an open URL would be drained by the first bot that
found it. With `PUBLIC_DEPLOYMENT=1`, which the image sets, the server will
not start until all of these are set, and it lists every missing one at once:

| variable | why |
|---|---|
| `ACCESS_CODE` | at least 12 characters; nobody without it can start a session |
| `MAX_SESSIONS_PER_DAY` | pick it from your Gemini tier; the session after the cap gets a polite "try tomorrow", not a dead page |
| `DATABASE_URL` | must be reachable; a public server never quietly falls back to memory |
| `GOOGLE_API_KEY` | otherwise the agent understands nothing |

Share the link as `https://your-host/#code=<ACCESS_CODE>`. The part after `#`
never leaves the browser, so the code does not end up in any server or proxy
log. The session token does not either: it travels in headers, never in a
URL.

Any host that offers HTTPS, WebSockets and managed Postgres will do; the
microphone only works on an `https://` page. Mount a persistent volume at
`/data` so the speech cache survives deploys, and run
`python scripts/db.py apply` against the database once.

## Choosing a recogniser with numbers

`scripts/wer.py` scores speech engines on the utterances this flow actually
collects — a spoken date, a mobile number, an uncommon name, a code-mixed
sentence, and the two consent answers, which are about 300 milliseconds long,
carry few acoustic cues, and are irreversible if confused.

```bash
.venv\Scripts\python.exe scripts/wer.py template
```

That writes `corpus/manifest.jsonl`. Record each utterance as `<id>.wav` from a
real speaker, correct each `reference` to what they actually said — people
improvise, and the reference is the truth — then:

```bash
.venv\Scripts\python.exe scripts/wer.py run --engines sarvam
```

Both word and character error rate are reported. **CER is the headline**: word
rate assumes spaces separate words, and Indic engines disagree about compounds
and postpositions, so a single spacing difference can cost two substitutions
while the characters are identical. Substitutions, deletions and insertions
are counted separately, because they are different faults — mishearing,
dropped audio, and hallucinating into silence — and one percentage hides which
you have.

The normalisation is written by hand rather than taken from a library, because
the off-the-shelf ones get Indic text wrong. Punctuation is stripped **by
Unicode category, never by keeping `isalnum()`** — that returns False for
Devanagari vowel signs, so `आशा` becomes `आश` and every Indic word quietly
gets shorter. Text is NFC-normalised first, Devanagari digits fold to ASCII (a
spoken phone number may return either, and both are the same answer), and the
danda is treated as punctuation because recognisers add it inconsistently.

**The browser cannot be scripted.** `SpeechRecognition` accepts microphone
input only and cannot be pointed at a file, so browser numbers require a
person speaking each utterance into the page; those results import with
`--browser results.json`. Any engine the script cannot call — a hosted demo,
a self-hosted IndicConformer — imports the same way, so adding one needs no
code change.

`bootstrap` will synthesise placeholder clips so the harness can be exercised
before a recording session exists. Every report it produces is stamped **not
evidence**: a recogniser scoring well on another engine's clean synthesis says
nothing about a person in a room with a fan running.

The manifest is committed; the audio is not. A recording of somebody's voice
is personal data, and a project that argues about DPDP compliance should not
ship a folder of real people saying their names into a public repository.

## Turn-taking

Tick **hands-free** and the button stops mattering: the agent listens, decides
you have finished, answers, and listens again.

By default, turn-taking runs off the recogniser's own results. Every result,
interim ones included, pushes the end-of-turn timer back, and when results
stop arriving for the current state's window, the turn ends. An earlier
version ran a separate audio detector, which opened a second microphone
stream. On some machines the two competed for the device, the recogniser
lost, and it quietly received silence — indistinguishable from a working
system with nothing to say. The recogniser's results exist whenever
recognition works at all, so turn-taking built on them cannot leave the
caller unable to speak. Barge-in works the same way.

The audio-energy detector is still there, behind the **mic meter** checkbox,
off by default. It reacts sooner and hears sound the recogniser never
transcribes. Under `SARVAM_STT=1` the page records instead of recognising, and
the detector reads that one recording stream for endpointing, so nothing
competes.

Silero VAD was the original plan. It runs server-side on audio frames, so it
does not fit the free path, where recognition happens in the page and the
server never receives audio. Under `SARVAM_STT=1` the server does receive
clips, so the option is open there.

What stays server-side is the **policy**: how long a pause means "finished".
That is a judgement about the conversation rather than about audio, so
`app/flow/turn_taking.py` owns it and sends it with every reply. Same split as
confidence — the client measures, the machine decides.

The windows are deliberately slower than the 500–800ms most agents use. Those
numbers are tuned on people who are fluent, unhurried and used to talking to
machines, and this is explicitly for people who are none of those. Someone
recalling their own address pauses; cutting them off and answering the
fragment is the experience of being rushed through a form by a stranger, which
is the thing being replaced.

**The consent gate waits longest — 1.8 seconds — which looks wrong.** "हाँ" is
one short word. But the second half of a sentence can reverse the first:

    "हाँ..."            endpoint here and consent is recorded
    "...नहीं, रुकिए"     which is what they were actually saying

Endpointing early there does not clip a transcript. It records a consent that
was being withdrawn while the agent stopped listening.

**Barge-in is always allowed** — a person must be able to interrupt a machine.
But interrupting the consent *notice* is written to the audit trail, because
DPDP requires notice as well as consent, and "हाँ" four seconds into a
twenty-second explanation is a weak claim to have received it. It is recorded,
not refused; the point is that the question stays answerable.

The energy thresholds are measured rather than guessed. `scripts/measure_vad.py`
runs the detector's own RMS maths over the corpus clips: real Hindi speech
sits at **RMS 0.26**, four orders of magnitude above the quiet between words,
and stays detectable against simulated room noise up to 0.08. Every constant
in the page is derived from that number, and the script re-derives them.

Detection was never the weak point. Three things were, and none of them were
fixable by choosing a better number:

- **One threshold makes speech flicker.** A signal hovering near it toggles
  voiced/unvoiced every frame and chops a sentence into fragments. Entering
  speech now needs 3.5× the noise floor; staying in it needs 2.0×. "Is this
  speech" and "are they still talking" are different questions.
- **Calibrating on the median deafens the detector.** In hands-free, people
  answer before the agent has finished — so the "ambient" window contains
  their voice, and the median of it becomes the threshold. It now takes the
  20th percentile and is capped at a third of speech level, so it cannot
  calibrate itself deaf.
- **`requestAnimationFrame` stops when the tab is hidden.** Switching tabs
  mid-sentence froze the detector and the turn never ended. It also paced at
  the monitor's refresh rate, which is nobody's sampling decision. Now a
  50 ms interval.

With **mic meter** ticked, the level meter beside the controls shows the
microphone and turns red when speech is detected. What remains unverified is *your* room and *your*
microphone — the numbers above come from real speech, not from real ambient
noise. The decision logic has its own cases:

```bash
open http://localhost:8000/?selftest=1
```

## Measuring the thing it optimises

The project argues that activation is the bottleneck and that the metric is
first-transaction completion rate. For most of its life nothing computed that
number, which is a comfortable position to argue from and a bad one to be in.

```bash
.venv\Scripts\python.exe scripts/funnel.py
```

```
reached                 n   of all  lost here   median
START                  21    100%                   -
CONSENT                21    100%                 50s
COLLECT_DATA           17     81%          2      54s
KYC_PENDING            13     62%                 68s
ACCOUNT_CREATE          7     33%          6      28s  <- worst
COMPLETED               3     14%                 65s

first-transaction completion      14%   (3/21)
accounts opened                     7
opened and never used               4   <- the problem being solved
```

It reads the **audit trail**, not `sessions.state`. Where somebody is now says
nothing about the journey — a session sitting in `ABANDONED` does not tell you
whether they gave up before consent or after their account was open, and those
are opposite problems with opposite fixes. The trail is append-only, so a
drop-off number cannot be quietly improved by editing history; that property
was built for regulators and turns out to be what makes the metric
trustworthy.

Lapses are counted separately from exits, because going quiet is not an
ending — the session waits. "They go silent while giving their address" and
"they abandon at consent" are different findings.

The worst step is chosen by **absolute** people lost, not by rate: a 50% drop
on a step four people reached is noise, and chasing it would waste a study.

This is what makes the user study a study. Ten people without it produce ten
anecdotes; with it they produce "six of ten stopped at the KYC handoff".

## What is real and what is not

**Real:** the state machine, the versioned eligibility rules, the append-only
audit trail in Postgres, the intent/fact boundary, the confidence policy, the
web layer, and Indic speech in and out.

**Mocked:** the video-KYC vendor, core banking, and the payment rail. Not
oversight — these are unobtainable as an individual developer. They sit behind
protocols with scripted implementations, which is why the KYC retry path and
the suspend/resume handoff are testable at all.

**Not done:** nobody who is not me has used it yet. The design argues that
activation is the bottleneck; proving it needs a drop-off study with real
non-English-reading users, and until that exists this is a well-tested
argument rather than a validated one. The instrument for that study now
exists — see above — so what is missing is the people, not the tooling.

The recogniser comparison has the same shape — the harness is built and
tested, but it has only been run against synthesised audio, which proves the
harness and nothing about accuracy. Both gaps need recordings of real people,
which is the honest boundary of what one developer can do alone at a desk.

Turn-taking is built and its energy thresholds are derived from measured
speech, but the audio path has never run against a live microphone — this
machine's automation browser blocks capture. The endpoint *windows* are
reasoned from the product rather than measured against recordings of people
hesitating, which is the same thing everything else here is waiting on.

**Also open:** nothing yet fires a lapse when a caller goes quiet (the method
exists and is tested; no timer calls it). The eligibility pre-screen still
runs on placeholder answers, so it turns nobody away before collection; the
re-screen on the real date of birth is what currently enforces the age rule.
And with Sarvam transcription, the consent gate has no confidence check — the
read-back covers the mobile, date of birth and name, but not consent. The
full list, with the reasoning, is in [CLAUDE.md](CLAUDE.md).

## Where to read next

[CLAUDE.md](CLAUDE.md) has the working notes: architecture rationale, measured
API facts, the machine-specific traps, and the current open list. Every count
in it was read out of the code rather than remembered.

The commit messages are written to be read. Most of them explain a decision and
what it cost.
