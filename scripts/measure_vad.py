"""What do real speech and real silence actually measure?

The browser detector thresholds were reasoned, not measured. There is no
microphone here, but there IS real Hindi speech: the corpus clips from the WER
bootstrap. Half the question can be answered honestly with those.

  measurable    what speech RMS looks like, and how far above the quiet parts
                between words it sits
  NOT           a real room's noise floor, because TTS silence is digital
                zero. That half is simulated by adding noise at known SNRs.
"""

import math
import statistics
import struct
import sys
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS = Path(__file__).resolve().parent.parent / "corpus"

# The browser reads 512 samples per frame from an AnalyserNode.
FRAME = 512


def frames(path: Path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())

    assert width == 2, f"{path.name}: expected 16-bit, got {width * 8}"
    count = len(raw) // 2
    samples = struct.unpack(f"<{count}h", raw)
    if channels == 2:
        samples = samples[::2]

    # Float in -1..1, exactly what getFloatTimeDomainData gives the browser.
    scaled = [s / 32768.0 for s in samples]
    for i in range(0, len(scaled) - FRAME, FRAME):
        block = scaled[i : i + FRAME]
        yield math.sqrt(sum(x * x for x in block) / FRAME), rate


def describe(path: Path):
    rms = [r for r, _ in frames(path)]
    rate = next(frames(path))[1]
    if not rms:
        return None

    ordered = sorted(rms)
    # Anything at or below the 10th percentile is between-words quiet; the
    # top decile is unambiguously voiced.
    quiet = ordered[: max(1, len(ordered) // 10)]
    loud = ordered[-max(1, len(ordered) // 10) :]
    return {
        "name": path.stem,
        "rate": rate,
        "frames": len(rms),
        "frame_ms": FRAME / rate * 1000,
        "quiet_median": statistics.median(quiet),
        "loud_median": statistics.median(loud),
        "overall_median": statistics.median(ordered),
        "peak": ordered[-1],
    }


clips = sorted(CORPUS.glob("*.wav"))
if not clips:
    raise SystemExit("no clips; run scripts/wer.py bootstrap --yes first")

print(f"{len(clips)} clips, {FRAME}-sample frames\n")
print(f"{'clip':<16}{'frame ms':>9}{'quiet':>10}{'speech':>10}{'ratio':>8}")
print("-" * 53)

rows = [describe(c) for c in clips]
for r in rows:
    ratio = r["loud_median"] / r["quiet_median"] if r["quiet_median"] > 1e-9 else float("inf")
    print(
        f"{r['name']:<16}{r['frame_ms']:>9.1f}{r['quiet_median']:>10.5f}"
        f"{r['loud_median']:>10.4f}{ratio:>8.0f}"
    )

print()
speech = statistics.median([r["loud_median"] for r in rows])
quiet = statistics.median([r["quiet_median"] for r in rows])
print(f"median speech RMS      : {speech:.4f}")
print(f"median between-word RMS: {quiet:.5f}   (TTS silence -- NOT a real room)")
print(f"frame length           : {rows[0]['frame_ms']:.1f} ms at {rows[0]['rate']} Hz")

# --- how much room noise can this survive? ---------------------------------
print()
print("Against simulated room noise, threshold = noise_median * MARGIN:")
print(f"{'room noise RMS':>16}{'threshold@3':>13}{'speech clears?':>16}{'headroom':>10}")
print("-" * 55)
for noise in (0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08):
    threshold = noise * 3.0
    print(
        f"{noise:>16.3f}{threshold:>13.4f}"
        f"{('yes' if speech > threshold else 'NO'):>16}{speech / threshold:>10.1f}x"
    )

print()
print(f"MIN_SPEECH_MS must exceed one frame ({rows[0]['frame_ms']:.0f} ms) or it")
print("cannot filter anything; two frames is the practical floor.")
