"""
A3 — streaming, and the number that decides whether your voice agent works.

This runs the SAME prompt three ways and times each one:

  1. blocking, low effort   — wait for the whole reply, then print it
  2. streaming, low effort  — print each piece as it arrives
  3. streaming, high effort — same, but the model thinks harder first

Watch for one number in particular: time to first SENTENCE. Text-to-speech
speaks sentences, not tokens, so that is the earliest moment your agent could
physically start making noise. It is the real floor on how alive it feels, and
in B3 you'll be trying to keep the whole round trip under 1.5 seconds.

Run it:  .venv\\Scripts\\python.exe learn\\a3_streaming.py
"""

import sys
import time

import anthropic
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
client = anthropic.Anthropic()

MODEL = "claude-opus-5"
PROMPT = (
    "In three short sentences, explain to a first-time bank customer "
    "what UPI is and why they would use it."
)

# Naive on purpose: "Rs. 500" and "3.5%" would fool it. Good enough to show the
# idea; a real agent needs something sturdier before it feeds a TTS engine.
SENTENCE_ENDS = (".", "?", "!", "।")  # । is the Devanagari danda


def has_full_sentence(text: str) -> bool:
    return any(ch in text for ch in SENTENCE_ENDS)


def run_blocking(effort: str) -> float:
    """Ask, wait, print. Nothing reaches the user until everything is ready."""
    start = time.monotonic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
        output_config={"effort": effort},
    )
    total = time.monotonic() - start

    text = "".join(b.text for b in response.content if b.type == "text")
    print(text)
    return total


def run_streaming(effort: str) -> tuple[float, float | None, float]:
    """Same request, but we read the reply as it is being written."""
    start = time.monotonic()
    ttft = None            # first character of any kind
    first_sentence = None  # first point TTS could start speaking

    # .stream() opens the connection and hands back events as they arrive.
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
        output_config={"effort": effort},
    ) as stream:
        # The stream carries several event types. "text" is the convenience one
        # the SDK adds: .text is the newly arrived piece, .snapshot is
        # everything received so far. We skip the rest, including the model's
        # private reasoning — which still happens first, so its cost is baked
        # into the time-to-first-token below.
        for event in stream:
            if event.type != "text":
                continue
            now = time.monotonic()
            if ttft is None:
                ttft = now - start
            if first_sentence is None and has_full_sentence(event.snapshot):
                first_sentence = now - start
            print(event.text, end="", flush=True)

    print()
    return ttft or 0.0, first_sentence, time.monotonic() - start


print("=" * 62)
print("1. BLOCKING, effort=low  — silence, then everything at once")
print("=" * 62)
blocking_total = run_blocking("low")
print(f"\n  total: {blocking_total:.2f}s")

print("\n" + "=" * 62)
print("2. STREAMING, effort=low — same request, read as it is written")
print("=" * 62)
s_ttft, s_sentence, s_total = run_streaming("low")

print("\n" + "=" * 62)
print("3. STREAMING, effort=high — the model thinks harder first")
print("=" * 62)
h_ttft, h_sentence, h_total = run_streaming("high")


def fmt(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "n/a"


print("\n" + "=" * 62)
print("RESULTS")
print("=" * 62)
print(f"{'':28}{'first token':>12}{'first sentence':>16}{'done':>8}")
print(f"{'1. blocking, low':28}{fmt(blocking_total):>12}"
      f"{fmt(blocking_total):>16}{fmt(blocking_total):>8}")
print(f"{'2. streaming, low':28}{fmt(s_ttft):>12}"
      f"{fmt(s_sentence):>16}{fmt(s_total):>8}")
print(f"{'3. streaming, high':28}{fmt(h_ttft):>12}"
      f"{fmt(h_sentence):>16}{fmt(h_total):>8}")

print("""
Read the table like this:

  Row 1 has the same number three times. That is the point of it. With no
  streaming there is no 'first token' — the reply does not exist until it is
  finished, so the user waits the full duration hearing nothing.

  Row 2's 'first sentence' column is your real budget. Everything else in the
  pipeline — speech-to-text before this, text-to-speech after it — has to fit
  around that number to stay under 1.5 seconds end to end.

  Row 3 shows what thinking costs you. Effort is the main dial you have, and
  in a voice loop it is usually the difference between alive and broken.
""")
