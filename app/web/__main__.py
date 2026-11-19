"""Run the server the way it is deployed: `python -m app.web`.

The settings that matter for a public server are fixed here in code rather
than left to whoever types the uvicorn command:

- `ws_max_size` comes from server.WS_MAX_SIZE, derived from the audio cap, so
  the frame limit cannot drift from the number it was derived from. Leaving
  it unset means uvicorn's 16 MB default per message.
- One worker. The daily session cap in app/web/gate.py is counted in this
  process; N workers would enforce N times the configured cap.
- `proxy_headers`, because every host puts a TLS proxy in front, and without
  it the scheme and client address are the proxy's.

`PORT` is read from the environment, which is how most hosts say where to
listen. Local development still uses the plain uvicorn command in the README.
"""

from __future__ import annotations

import os

import uvicorn

from app.web.server import WS_MAX_SIZE


def main() -> None:
    uvicorn.run(
        "app.web.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        ws_max_size=WS_MAX_SIZE,
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"),
    )


if __name__ == "__main__":
    main()
