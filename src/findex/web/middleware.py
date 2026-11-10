import logging
import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from findex.web.logs import request_id

log = logging.getLogger("findex.web")

_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class RequestContextMiddleware:
    # Чистий ASGI: app(scope, receive, send). Сидить між uvicorn і FastAPI:
    # 1) дає кожному запиту id (або бере X-Request-ID від проксі платформи);
    # 2) дописує X-Request-ID у відповідь;
    # 3) неочікуваний виняток -> стек у лог з id, назовні 500 без стека.

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan, websocket — без змін
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        rid = incoming if _VALID_ID.fullmatch(incoming) else uuid.uuid4().hex[:12]
        token = request_id.set(rid)
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                MutableHeaders(scope=message).append("X-Request-ID", rid)
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception(
                "unhandled error: %s %s",
                scope["method"],
                scope["path"],
                extra={"request_id": rid},
            )
            if started:  # заголовки вже пішли клієнту — 500 не надіслати, лише обірвати
                raise
            response = JSONResponse(
                {"detail": "internal server error", "request_id": rid},
                status_code=500,
                headers={"X-Request-ID": rid},
            )
            await response(scope, receive, send)
        finally:
            request_id.reset(token)
