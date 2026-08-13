"""요청 미들웨어."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.context import TRACE_ID_HEADER, set_trace_id
from app.core.metrics import (
    http_request_duration_seconds,
    http_requests_in_progress,
    http_requests_total,
    route_template,
)

logger = logging.getLogger(__name__)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """스프링이 보낸 traceId 를 이어받고, 없으면 새로 만든다. 응답 헤더로 돌려준다.

    같은 자리에서 Prometheus 메트릭도 기록한다. 이미 소요 시간을 재고 있어서 계측 지점이
    하나로 모인다.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = set_trace_id(request.headers.get(TRACE_ID_HEADER))

        started = time.perf_counter()
        http_requests_in_progress.inc()
        try:
            response = await call_next(request)
        except Exception:
            # 예외가 핸들러까지 못 갔을 때도 요청이 있었다는 사실은 남겨야 한다.
            # 여기서 세지 않으면 500 이 통계에서 사라져 "에러율 0%" 로 보인다.
            elapsed = time.perf_counter() - started
            path = route_template(request)
            http_requests_total.labels(request.method, path, "500").inc()
            http_request_duration_seconds.labels(request.method, path).observe(elapsed)
            raise
        finally:
            http_requests_in_progress.dec()

        elapsed = time.perf_counter() - started
        # 라우트 템플릿은 라우팅이 끝난 뒤에야 확정된다. 그래서 call_next 다음에 읽는다.
        path = route_template(request)
        http_requests_total.labels(request.method, path, str(response.status_code)).inc()
        http_request_duration_seconds.labels(request.method, path).observe(elapsed)

        response.headers[TRACE_ID_HEADER] = trace_id
        logger.info(
            "%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed * 1000
        )
        return response
