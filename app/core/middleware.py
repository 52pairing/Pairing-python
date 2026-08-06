"""요청 미들웨어."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import TRACE_ID_HEADER, set_trace_id

logger = logging.getLogger(__name__)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """스프링이 보낸 traceId 를 이어받고, 없으면 새로 만든다. 응답 헤더로 돌려준다."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = set_trace_id(request.headers.get(TRACE_ID_HEADER))

        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000

        response.headers[TRACE_ID_HEADER] = trace_id
        logger.info(
            "%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms
        )
        return response
