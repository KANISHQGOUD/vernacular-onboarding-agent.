"""Print the Gemini SDK surface this project actually depends on.

Run this after any `pip install -U google-genai`, and paste the output over the
verified-surface block at the top of app/llm/client.py.

    .venv\\Scripts\\python.exe scripts/introspect_sdk.py

The point: a comment claiming "the config field is called response_schema" is
a memory. This is a measurement. When they disagree, the measurement wins.
"""

from __future__ import annotations

import inspect
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai
from google.genai import errors, types

print(f"google-genai {genai.version.__version__}\n")

print("Client(...)")
print("   ", inspect.signature(genai.Client.__init__))

print("\nGenerateContentConfig fields the project uses")
used = (
    "system_instruction",
    "response_mime_type",
    "response_schema",
    "thinking_config",
    "safety_settings",
    "temperature",
    "max_output_tokens",
    "tools",
    "tool_config",
)
fields = types.GenerateContentConfig.model_fields
for name in used:
    mark = "ok " if name in fields else "GONE"
    print(f"    [{mark}] {name}")

missing = [n for n in used if n not in fields]
if missing:
    print(f"\n    !! these no longer exist: {missing}")

print("\nThinkingConfig")
for name, f in types.ThinkingConfig.model_fields.items():
    print(f"    {name}: {f.annotation}")
print("    ThinkingLevel:", [e.value for e in types.ThinkingLevel])

print("\nSchema (enum constraint is what the classifier rides on)")
for name in ("type", "enum", "properties", "required"):
    if name in types.Schema.model_fields:
        print(f"    {name}: {types.Schema.model_fields[name].annotation}")
print("    Type:", [e.value for e in types.Type])

print("\nResponse accessors")
for name in ("text", "parsed", "function_calls"):
    where = (
        "property"
        if isinstance(getattr(types.GenerateContentResponse, name, None), property)
        else "field"
        if name in types.GenerateContentResponse.model_fields
        else "MISSING"
    )
    print(f"    .{name}: {where}")

print("\nErrors")
print("    APIError bases:", [b.__name__ for b in errors.APIError.__bases__])
for name in ("ClientError", "ServerError"):
    cls = getattr(errors, name)
    print(f"    {name} bases:", [b.__name__ for b in cls.__bases__])
