"""Who is allowed to talk to which session.

A WebSocket that accepts a client-supplied session id and starts talking is an
IDOR: session ids appear in URLs, logs, and browser history, and a half-finished
bank application is exactly the thing you do not want a stranger attaching to.

So creating a session mints a bearer token, and the token -- not the id -- is
the thing that proves ownership. The id stays a name; the token stays a secret.

The token deliberately does NOT live on `Session`. Session is the durable
record that gets written to storage, replayed for audit, and one day handed to
Postgres. An access credential has a different lifetime and a different blast
radius, and mixing the two is the same mistake as putting operational notes in
the applicant dict.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from app.session import InMemorySessionStore, Session, SessionStore

TOKEN_BYTES = 32


@dataclass
class SessionRegistry:
    """Sessions plus the tokens that authorise access to them."""

    store: SessionStore = field(default_factory=InMemorySessionStore)
    _tokens: dict[str, str] = field(default_factory=dict, init=False)

    def create(self) -> tuple[Session, str]:
        session = Session()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self.store.save(session)
        self._tokens[session.session_id] = token
        return session, token

    def open(self, session_id: str, token: str) -> Session | None:
        """Return the session only if the token proves ownership.

        One return value for every failure -- unknown id, wrong token, missing
        token -- so a caller cannot distinguish "no such session" from "not
        yours" and enumerate live sessions by watching the difference.
        """
        expected = self._tokens.get(session_id)
        if expected is None or not token:
            return None

        # Constant-time: a plain == leaks how much of the token was right,
        # which over enough attempts is enough to reconstruct it.
        if not secrets.compare_digest(expected, token):
            return None

        return self.store.load(session_id)

    def save(self, session: Session) -> None:
        self.store.save(session)

    def forget(self, session_id: str) -> None:
        """Drop the token. The session record itself is untouched."""
        self._tokens.pop(session_id, None)
