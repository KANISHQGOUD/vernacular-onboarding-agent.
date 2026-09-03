"""Compare speech recognisers on the utterances this agent actually hears.

    scripts/wer.py template            write a starter manifest
    scripts/wer.py bootstrap           synthesise placeholder audio (COSTS MONEY)
    scripts/wer.py run                 score every engine, print a table
    scripts/wer.py run --engines sarvam --json out.json

WHY THIS EXISTS
---------------
"Which recogniser should this use" is currently answered by opinion. Sarvam
transcribing its own clean synthesis proved the plumbing works and nothing
about accuracy. This turns the question into a number, on the utterances that
actually matter here rather than on a generic benchmark.

THE BROWSER CANNOT BE SCRIPTED
------------------------------
`SpeechRecognition` takes microphone input only -- it cannot be pointed at a
file. So browser results cannot be produced by this script; they have to be
captured by a person speaking into a page, and imported:

    scripts/wer.py run --browser browser-results.json

That file is `{"utterance-id": "what the browser heard", ...}`. Any other
engine that cannot be called from here -- a hosted demo, a colleague's phone --
imports the same way.

NORMALISATION CONFOUNDS THE COMPARISON -- READ THE TRANSCRIPTS
--------------------------------------------------------------
Found on the first real run, and it is the reason this prints transcripts
rather than only a percentage. Sarvam normalises as it transcribes:

    said   एक जनवरी उन्नीस सौ नब्बे    heard  1 जनवरी 1990
    said   पाँच सौ रुपये                heard  ₹500
    said   मेरा account खोलना है        heard  मेरा अकाउंट खोलना है

All three are correct. All three score as near-total errors, and the run came
out at 150% WER for an engine that misheard nothing.

That matters beyond one confusing number: an engine that normalises is being
punished against one that does not, so a naive table would recommend the
*worse* recogniser. For this project the normalising one is actively better --
the classifier downstream wants 1990, not उन्नीस सौ नब्बे.

Two honest ways to handle it. Write the references in the normalised form the
engines actually emit, or read the transcripts and judge. This prints every
non-matching pair so the second is possible; nobody should quote the aggregate
without having looked.

WHAT COUNTS AS EVIDENCE
-----------------------
Recordings of real people. `bootstrap` exists to smoke-test this harness end
to end without waiting for a recording session, and its output is NOT
evidence: a TTS engine's clean synthesis is far easier than a person in a room
with a fan on, and Sarvam scoring well on its own voice tells you nothing.
Every report says so on its face.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from app.speech.wer import character_error_rate, word_error_rate

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
MANIFEST = CORPUS / "manifest.jsonl"

# The awkward cases, chosen because they are what this flow collects and where
# recognisers actually differ -- not because they make a nice demo.
TEMPLATE = [
    {
        "id": "consent-yes",
        "language": "hi-IN",
        "reference": "हाँ",
        "note": "the consent gate. ~300ms, few acoustic cues, irreversible if wrong",
    },
    {
        "id": "consent-no",
        "language": "hi-IN",
        "reference": "नहीं",
        "note": "the other half. confusing these two is the worst failure here",
    },
    {
        "id": "date-spoken",
        "language": "hi-IN",
        "reference": "एक जनवरी उन्नीस सौ नब्बे",
        "note": "a date said in words, not digits",
    },
    {
        "id": "mobile-number",
        "language": "hi-IN",
        "reference": "९८१२३४५६७८",
        "note": "ten digits. transcription may come back Devanagari or ASCII",
    },
    {
        "id": "name-uncommon",
        "language": "hi-IN",
        "reference": "आशा देवी राठौड़",
        "note": "the case a strict confidence floor would punish",
    },
    {
        "id": "codemix",
        "language": "hi-IN",
        "reference": "मेरा account खोलना है",
        "note": "Hinglish. how people actually speak, and where generic ASR fails",
    },
    {
        "id": "address",
        "language": "hi-IN",
        "reference": "चौदह गांधी मार्ग जयपुर",
        "note": "place names, which are mostly out-of-vocabulary",
    },
    {
        "id": "amount",
        "language": "hi-IN",
        "reference": "पाँच सौ रुपये",
        "note": "money, spoken as words",
    },
]


@dataclass(frozen=True, slots=True)
class Utterance:
    id: str
    language: str
    reference: str
    audio: Path | None
    note: str = ""


def load_manifest(path: Path) -> list[Utterance]:
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}\n\nWrite a starter one with:\n"
            f"  .venv\\Scripts\\python.exe scripts/wer.py template"
        )

    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        audio = raw.get("audio")
        out.append(
            Utterance(
                id=raw["id"],
                language=raw.get("language", "hi-IN"),
                reference=raw["reference"],
                audio=(path.parent / audio) if audio else None,
                note=raw.get("note", ""),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def sarvam_engine():
    from app.speech import SarvamSpeech
    from app.speech.provider import Audio

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise SystemExit("SARVAM_API_KEY is empty; cannot run the sarvam engine")

    provider = SarvamSpeech(api_key=key)

    def transcribe(item: Utterance) -> str:
        if item.audio is None or not item.audio.exists():
            return ""
        clip = Audio(data=item.audio.read_bytes(), media_type="audio/wav")
        return provider.transcribe(clip, item.language).text

    return transcribe


def imported_engine(path: Path):
    """Results captured somewhere this script cannot reach. See the docstring."""
    heard = json.loads(path.read_text(encoding="utf-8"))
    return lambda item: heard.get(item.id, "")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_template(args) -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not args.force:
        raise SystemExit(f"{MANIFEST} exists; pass --force to overwrite")

    with MANIFEST.open("w", encoding="utf-8", newline="\n") as fh:
        for row in TEMPLATE:
            fh.write(json.dumps({**row, "audio": f"{row['id']}.wav"}, ensure_ascii=False) + "\n")

    print(f"wrote {MANIFEST} with {len(TEMPLATE)} utterances")
    print()
    print("Now record each one as <id>.wav in that folder, from a REAL speaker.")
    print("Edit `reference` to match exactly what they said, not what you asked")
    print("them to say -- the reference is the truth, and people improvise.")


def cmd_bootstrap(args) -> None:
    """Synthesise placeholder audio so the harness can be exercised today."""
    from app.speech import SarvamSpeech

    items = load_manifest(MANIFEST)
    chars = sum(len(i.reference) for i in items)
    cost = chars / 10_000 * 30

    print(f"About to synthesise {len(items)} clips, {chars} characters.")
    print(f"Estimated cost: ~Rs {cost:.2f} of a one-time prepaid balance.")
    if not args.yes:
        raise SystemExit("Re-run with --yes to spend it.")

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise SystemExit("SARVAM_API_KEY is empty")
    provider = SarvamSpeech(api_key=key)

    for item in items:
        target = item.audio or (CORPUS / f"{item.id}.wav")
        audio = provider.synthesize(item.reference, item.language)
        target.write_bytes(audio.data)
        print(f"  {item.id:16} {len(audio):7} bytes")

    # Remembered on disk, so the warning survives this process. Otherwise a
    # synthetic run reported next week would look like real evidence.
    (CORPUS / ".synthetic").write_text(
        chr(10).join(item.id for item in items), encoding="utf-8"
    )

    print()
    print("NOT EVIDENCE. This is a recogniser being tested on another engine's")
    print("clean synthesis, which is far easier than a person in a real room.")
    print("Replace these with recordings of actual speakers before reporting.")


def cmd_run(args) -> None:
    items = load_manifest(MANIFEST)

    engines: dict[str, object] = {}
    if "sarvam" in args.engines:
        engines["sarvam"] = sarvam_engine()
    if args.browser:
        engines["browser"] = imported_engine(Path(args.browser))
    for spec in args.imported or []:
        name, _, path = spec.partition("=")
        engines[name] = imported_engine(Path(path))

    if not engines:
        raise SystemExit("no engines selected; try --engines sarvam or --browser FILE")

    synthetic = any(
        (item.audio and item.audio.exists() and item.id in _synthetic_ids()) for item in items
    )

    results: dict[str, list[dict]] = {}
    for name, transcribe in engines.items():
        rows = []
        for item in items:
            hypothesis = transcribe(item)
            wer = word_error_rate(item.reference, hypothesis)
            cer = character_error_rate(item.reference, hypothesis)
            rows.append(
                {
                    "id": item.id,
                    "reference": item.reference,
                    "heard": hypothesis,
                    "wer": wer.rate,
                    "cer": cer.rate,
                    "substitutions": wer.substitutions,
                    "deletions": wer.deletions,
                    "insertions": wer.insertions,
                }
            )
        results[name] = rows

    _print_report(items, results, synthetic=synthetic)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


def _synthetic_ids() -> set[str]:
    marker = CORPUS / ".synthetic"
    if not marker.exists():
        return set()
    return set(marker.read_text(encoding="utf-8").split())


def _print_report(items, results, *, synthetic: bool) -> None:
    names = list(results)

    print()
    print("PER UTTERANCE (character error rate)")
    print("-" * (24 + 14 * len(names)))
    print(f"{'utterance':<24}" + "".join(f"{n:>14}" for n in names))
    for index, item in enumerate(items):
        cells = "".join(f"{results[n][index]['cer']:>13.1%} " for n in names)
        print(f"{item.id:<24}{cells}")

    print()
    print("OVERALL")
    print("-" * 46)
    for name in names:
        rows = results[name]
        wer = sum(r["wer"] for r in rows) / len(rows)
        cer = sum(r["cer"] for r in rows) / len(rows)
        subs = sum(r["substitutions"] for r in rows)
        dels = sum(r["deletions"] for r in rows)
        ins = sum(r["insertions"] for r in rows)
        print(f"{name:<12} WER {wer:6.1%}   CER {cer:6.1%}   S{subs} D{dels} I{ins}")

    print()
    print("CER is the headline: Indic engines disagree about word boundaries,")
    print("so one spacing difference can cost two word-substitutions while the")
    print("characters are identical.")

    print()
    print("EVERY DISAGREEMENT, because the aggregate lies when engines")
    print("normalise differently -- 1990 for उन्नीस सौ नब्बे is not an error.")
    print("-" * 46)

    clean = True
    for name in names:
        for row in results[name]:
            if row["cer"] == 0:
                continue
            clean = False
            print(f"{name} / {row['id']}  (CER {row['cer']:.1%})")
            print(f"  said  : {row['reference']}")
            print(f"  heard : {row['heard'] or '(nothing)'}")
    if clean:
        print("none -- every transcript matched exactly")

    if synthetic:
        print()
        print("!! Some clips are synthesised, not recorded. Not evidence. !!")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="write a starter manifest")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_template)

    b = sub.add_parser("bootstrap", help="synthesise placeholder audio (costs money)")
    b.add_argument("--yes", action="store_true", help="confirm the spend")
    b.set_defaults(func=cmd_bootstrap)

    r = sub.add_parser("run", help="score the engines")
    r.add_argument("--engines", nargs="*", default=[], help="callable engines, e.g. sarvam")
    r.add_argument("--browser", help="JSON of results captured in a browser")
    r.add_argument("--imported", nargs="*", help="name=path.json for any other engine")
    r.add_argument("--json", help="write the full report here")
    r.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
