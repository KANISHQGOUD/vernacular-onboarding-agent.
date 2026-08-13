"""
A2, part 2 — the same loop, but you keep the memory.

The only real change from a2_forgetful.py: `messages` is created ONCE, outside
the loop, and every turn appends to it. That list is the conversation. There is
no server-side session, no thread ID, no "chat" object — just a list you own
and resend.

Commands inside the loop:
    /history   print the raw messages list (this is the whole trick, visible)
    /forget    empty the list and watch amnesia return mid-conversation
    quit       exit

Run it:  .venv\\Scripts\\python.exe learn\\a2_conversation.py
"""

import json
import sys
import time

import anthropic
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
client = anthropic.Anthropic()

# The system prompt: standing instructions that apply to the whole conversation.
# In this API it is a SEPARATE top-level argument, not an entry in `messages`.
# `messages` holds only "user" and "assistant" turns.
SYSTEM = (
    "You are a helpful assistant helping test a conversation loop. "
    "Keep replies to two sentences unless asked for more."
)

# THE conversation. One list, living outside the loop. This is the memory.
messages = []

# Running cost, so you can watch what resending history actually costs you.
total_in = 0
total_out = 0

print("Try: 'My name is Dhananjay.' then 'What is my name?'")
print("Commands: /history  /forget  quit\n")

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

    if user_input == "/history":
        print("\n--- the list being sent every turn ---")
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        print(f"--- {len(messages)} messages ---\n")
        continue

    if user_input == "/forget":
        messages.clear()
        print("\n[history cleared — it will now forget everything above]\n")
        continue

    # 1. Add what the human said.
    messages.append({"role": "user", "content": user_input})

    started = time.monotonic()
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        system=SYSTEM,
        # 2. Send the ENTIRE list. Not the latest message — all of it, every
        #    time. This is the only reason the model appears to remember.
        messages=messages,
        # `effort` controls how hard the model thinks before answering. "low"
        # keeps a chat loop snappy; you'll tune this deliberately later.
        output_config={"effort": "low"},
    )
    elapsed = time.monotonic() - started

    reply = "".join(b.text for b in response.content if b.type == "text")

    # 3. Add what the model said, so the next turn can see it too. Forget this
    #    step and the model will keep re-answering as if it never spoke.
    #
    #    We store just the text here because it keeps `messages` readable. From
    #    A4 onward you will append the full `response.content` instead, because
    #    tool-call blocks have to survive the round trip.
    messages.append({"role": "assistant", "content": reply})

    print(f"Claude: {reply}")

    total_in += response.usage.input_tokens
    total_out += response.usage.output_tokens
    print(
        f"  [{elapsed:.1f}s | this turn sent {response.usage.input_tokens} "
        f"input tokens | session total {total_in} in / {total_out} out]\n"
    )

print(f"\nConversation was {len(messages)} messages.")
print(f"Total tokens: {total_in} in, {total_out} out.")
