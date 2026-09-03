"""Who is allowed to talk to which session.

A WebSocket that accepts a client-supplied session id and starts talking is an
IDOR: session ids appear in URLs, logs, and browser history, and a half-finished
bank application is exactly the thing you do not want a stranger attaching to.

So creating a session mints a bearer token, and the token -- not the id -- is
the thing that proves ownership. The id stays a name; the token stays a secret.

The token deliberately does NOT live on `Session`. Session is the durable
record that gets written to storage, replayed for audit, and handed to
Postgres. An access credential has a different lifetime and a different blast
radius, and mixing the two is the same mistake as putting operational notes in
the applicant dict.

ONLY THE HASH IS EVER STORED
----------------------------
What is kept is `sha256(token)`, never the token. The value the browser holds
is the only copy that exists.

That matters least in memory and most in Postgres, but doing it the same way
in both means there is no version of this code that writes a live credential
somewhere it can be read back -- not a database dump, not a backup on a
laptop, not a support engineer with SELECT.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Protocol

from app.session import InMemorySessionStore, Session, SessionStore

TOKEN_BYTES = 32


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenStore(Protocol):
    """The seam Postgres drops into, same as SessionStore."""

    def put(self, session_id: str, token_sha256: str) -> None: ...

    def get(self, session_id: str) -> str | None: ...

    def forget(self, session_id: str) -> None: ...


@dataclass
class InMemoryTokenStore:
    _hashes: dict[str, str] = field(default_factory=dict)

    def put(self, session_id: str, token_sha256: str) -> None:
        self._hashes[session_id] = token_sha256

    def get(self, session_id: str) -> str | None:
        return self._hashes.get(session_id)

    def forget(self, session_id: str) -> None:
        self._hashes.pop(session_id, None)


@dataclass
class SessionRegistry:
    """Sessions plus the tokens that authorise access to them."""

    store: SessionStore = field(default_factory=InMemorySessionStore)
    tokens: TokenStore = field(default_factory=InMemoryTokenStore)

    def create(self) -> tuple[Session, str]:
        session = Session()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self.store.save(session)
        self.tokens.put(session.session_id, hash_token(token))
        # The only time the plaintext token exists on this side.
        return session, token

    def open(self, session_id: str, token: str) -> Session | None:
        """Return the session only if the token proves ownership.

        One return value for every failure -- unknown id, wrong token, missing
        token -- so a caller cannot distinguish "no such session" from "not
        yours" and enumerate live sessions by watching the difference.
        """
        expected = self.tokens.get(session_id)
        if expected is None or not token:
            return None

        # Constant-time: a plain == leaks how much of the hash was right,
        # which over enough attempts is enough to reconstruct it.
        if not secrets.compare_digest(expected, hash_token(token)):
            return None

        return self.store.load(session_id)

    def save(self, session: Session) -> None:
        self.store.save(session)

    def forget(self, session_id: str) -> None:
        """Drop the token. The session record itself is untouched."""
        self.tokens.forget(session_id)
