# app/middleware/ws_auth.py
from typing import Iterable, Optional, Set
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Scope, Receive, Send
from jose import jwt, JWTError
from app.core.config_env import settings

class TokenWebSocketAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, protected_paths: Iterable[str] = ("/ws/alerts",), allowed_roles: Optional[Set[str]] = None):
        super().__init__(app)
        self.protected_paths = set(protected_paths)
        self.allowed_roles = allowed_roles or {"admin", "officer"}

    async def dispatch(self, request, call_next):
        return await call_next(request)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            return await super().__call__(scope, receive, send)

        path = scope.get("path", "")
        if path not in self.protected_paths:
            return await super().__call__(scope, receive, send)

        # token из query (?token=...) или Authorization: Bearer ...
        query_string = scope.get("query_string", b"").decode("utf-8")
        token = None
        if query_string:
            for part in query_string.split("&"):
                if part.startswith("token="):
                    token = part.split("=", 1)[1]
                    break
        if not token:
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            auth = headers.get("authorization")
            if auth and auth.lower().startswith("bearer "):
                token = auth.split(" ", 1)[1]
        if not token:
            await self._close_with_error(send, code=4401, reason="Missing token")
            return

        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            role = (payload.get("role") or "").lower()          # ← СНАЧАЛА роль
            scope.setdefault("auth", {})
            scope["auth"]["user_id"] = payload.get("sub")       # ← КЛАДЁМ user_id и роль в scope
            scope["auth"]["role"] = role
        except JWTError:
            await self._close_with_error(send, code=4401, reason="Invalid token")
            return

        if role not in self.allowed_roles:
            await self._close_with_error(send, code=4403, reason="Forbidden")
            return

        await super().__call__(scope, receive, send)

    async def _close_with_error(self, send: Send, code: int, reason: str = ""):
        await send({"type": "websocket.close", "code": code, "reason": reason})
