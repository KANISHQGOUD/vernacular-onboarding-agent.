"""
A1 — One LLM API call, by hand.

Goal: send a single request over the internet and print what comes back.
We build the request piece by piece with `requests` so you can SEE the four
parts of an HTTP call. Production code would use the `anthropic` SDK instead
(see a1_sdk.py) — but the SDK hides exactly the parts you're here to learn.

Run it:  .venv\\Scripts\\python.exe learn\\a1_raw_http.py
"""

import json
import os

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. The key
# ---------------------------------------------------------------------------
# load_dotenv() reads the .env file and loads its contents into "the
# environment" — a bag of name/value pairs that belongs to this running
# process. Nothing is written to your code, so nothing can leak into git.
load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise SystemExit(
        "No ANTHROPIC_API_KEY found.\n"
        "Open the .env file in this folder and paste your key after the `=`."
    )

# ---------------------------------------------------------------------------
# 1. URL — where the request goes
# ---------------------------------------------------------------------------
URL = "https://api.anthropic.com/v1/messages"

# ---------------------------------------------------------------------------
# 2. Headers — information ABOUT the request
# ---------------------------------------------------------------------------
# Not the question itself; the metadata around it. Your key lives here, which
# is how the server knows the request is yours and whose account to bill.
HEADERS = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",  # pins the API's behaviour so future
                                        # changes can't silently break your code
    "content-type": "application/json",  # "the body below is JSON"
}

# ---------------------------------------------------------------------------
# 3. Body — what you actually want
# ---------------------------------------------------------------------------
BODY = {
    "model": "claude-opus-5",
    # max_tokens caps the response LENGTH. On this model it covers the model's
    # private reasoning AND the visible reply together, so leave headroom —
    # too small and the answer gets cut off mid-sentence.
    "max_tokens": 2048,
    # `messages` is a list because a conversation is a list. Right now it has
    # one entry. In A2 you'll append to it and watch memory appear.
    "messages": [
        {"role": "user", "content": "Explain UPI in one sentence."}
    ],
}

# ---------------------------------------------------------------------------
# 4. Send it, and read what comes back
# ---------------------------------------------------------------------------
# POST means "here is some data, do something with it and reply".
# json=BODY tells requests to serialise BODY to JSON and send it.
print("Sending request...\n")
response = requests.post(URL, headers=HEADERS, json=BODY, timeout=60)

# Every HTTP reply carries a status code. Learn these four:
print(f"HTTP status: {response.status_code}")

if response.status_code == 401:
    raise SystemExit("401 Unauthorized — the key is wrong, revoked, or empty.")
if response.status_code == 429:
    raise SystemExit("429 Too Many Requests — you're going too fast. Wait and retry.")
if response.status_code >= 500:
    raise SystemExit(f"{response.status_code} — their problem, not yours. Retry shortly.")
if response.status_code != 200:
    raise SystemExit(f"Unexpected status.\n{response.text}")

# 200 means fine. The body of the reply is JSON; .json() parses it into a
# Python dict.
data = response.json()

print("\n--- the whole reply, as it arrived ---")
print(json.dumps(data, indent=2, ensure_ascii=False))

# `content` is a LIST of blocks, not a single string. A reply can contain a
# thinking block (the model's private reasoning) before the text block. That's
# why we filter by type instead of grabbing content[0] — do that and you'll
# eventually crash on a response that happens to start with thinking.
print("\n--- just the answer ---")
for block in data["content"]:
    if block["type"] == "text":
        print(block["text"])

# `usage` is what you paid. Watch this number as your prompts grow.
usage = data["usage"]
print(
    f"\nTokens — in: {usage['input_tokens']}, out: {usage['output_tokens']}"
)
