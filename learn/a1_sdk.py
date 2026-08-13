"""
A1b — the same call, the way you'll actually write it from A2 onward.

Identical request to a1_raw_http.py. The URL, the headers, the version pin and
the JSON parsing are all still happening — the SDK just does them for you.
Read a1_raw_http.py first, then this, and notice what disappeared.

From A3 (streaming) and A4 (tool calling) on, the SDK stops being a
convenience and starts being the difference between 20 lines and 200.

Run it:  .venv\\Scripts\\python.exe learn\\a1_sdk.py
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

# No api_key= argument: the client looks for ANTHROPIC_API_KEY in the
# environment on its own, which load_dotenv() just put there.
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=2048,
    messages=[
        {"role": "user", "content": "Explain UPI in one sentence."}
    ],
)

# Same content-block list as the raw version, now as typed Python objects:
# block.type instead of block["type"].
for block in response.content:
    if block.type == "text":
        print(block.text)

print(
    f"\nTokens — in: {response.usage.input_tokens}, "
    f"out: {response.usage.output_tokens}"
)
