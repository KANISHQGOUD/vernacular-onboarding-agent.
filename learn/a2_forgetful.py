"""
A2, part 1 — the loop that forgets. This script is SUPPOSED to fail.

Run it and have this exact conversation:

    You: My name is Dhananjay.
    You: What is my name?

It will not know. Nothing is broken and nothing is misconfigured — this is
what the API actually does. Understand why before you read part 2.

Run it:  .venv\\Scripts\\python.exe learn\\a2_forgetful.py
"""

import sys

import anthropic
from dotenv import load_dotenv

# Windows consoles often default to a non-UTF-8 codepage, which mangles any
# non-English text. Do this in every script from here on.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
client = anthropic.Anthropic()

print("Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"quit", "exit"}:
        break
    if not user_input:
        continue

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        # LOOK HERE. Every single time round this loop we send exactly one
        # message: whatever was just typed. Nothing that came before it is
        # in this request, so nothing that came before it can be known.
        messages=[
            {"role": "user", "content": user_input}
        ],
    )

    for block in response.content:
        if block.type == "text":
            print(f"Claude: {block.text}\n")
