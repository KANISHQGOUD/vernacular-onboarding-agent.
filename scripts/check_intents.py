"""Does the classifier still read real sentences the way the map expects?

    .venv\\Scripts\\python.exe scripts/check_intents.py

OPT IN, LIKE THE SARVAM ROUND TRIP
----------------------------------
This spends real quota -- 20 requests a day and 5 a minute, per project per
model -- so it is a script somebody runs on purpose rather than a test that
runs itself. `tests/test_llm.py` covers the same machinery against a fake
client and costs nothing; what it cannot cover is whether the live model
picks the intent we think it will, which is the only question here.

WHY IT EXISTS
-------------
The activation intents shipped with their glosses checked and the prompt
inspected, and both were fine -- and the amounts still came back empty every
time, because `_read_field` discarded them. Nothing offline caught it: the
suite fakes the classifier, so it faked the bug away too. Four live calls
found it in under a minute.

Sleeps between calls because the free tier allows five a minute.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Named explicitly rather than found: `find_dotenv` walks up from the calling
# frame, and this file lives one directory below the .env it needs.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.flow.states import State
from app.llm.classifier import GeminiIntentClassifier
from app.llm.client import LlmClient, api_key_available

# (state, what somebody says, the intent expected, the amount expected)
#
# Written as sentences a real caller would say, in the mixture of scripts they
# would say them in -- not as tidy English, which is the one input this
# product will never receive.
CASES: tuple[tuple[State, str, str, str | None], ...] = (
    (State.FUNDING, "haan, paanch sau daal do", "DEPOSIT_REQUESTED", "500"),
    (State.FUNDING, "हाँ, एक हज़ार रुपये डाल दीजिए", "DEPOSIT_REQUESTED", "1000"),
    (State.FIRST_TRANSACTION, "haan bhej do, pachaas rupaye", "PAYMENT_REQUESTED", "50"),
    # A bare yes must name no amount, or the suggestion it is supposed to
    # accept gets overwritten by a number nobody said.
    (State.FUNDING, "haan theek hai", "DEPOSIT_REQUESTED", None),
    # And a no must not be read as enthusiasm for spending money.
    (State.FUNDING, "nahi, mujhe nahi karna", "USER_QUIT", None),
)

SECONDS_BETWEEN = 13


def main() -> int:
    if not api_key_available():
        print("GOOGLE_API_KEY is empty. Nothing to check.")
        return 1

    classifier = GeminiIntentClassifier(LlmClient())
    passed = 0

    for index, (state, said, expected_intent, expected_amount) in enumerate(CASES):
        reading = classifier.read(said, state)
        intent = reading.intent.name if reading else "None"
        amount = reading.amount if reading else None
        ok = intent == expected_intent and amount == expected_amount
        passed += ok

        print(f"{'OK  ' if ok else 'MISS'} [{state.value:17}] {said}")
        if not ok:
            print(f"        expected {expected_intent:20} amount={expected_amount!r}")
            print(f"        got      {intent:20} amount={amount!r}")

        if index < len(CASES) - 1:
            time.sleep(SECONDS_BETWEEN)

    print(f"\n{passed}/{len(CASES)} correct")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
