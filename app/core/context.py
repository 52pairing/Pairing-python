"""요청 단위 컨텍스트.

스프링이 X-Trace-Id 를 붙여 보내면 그 값을 그대로 이어 쓴다.
두 서버의 로그를 같은 ID 로 맞춰야 "이 요청이 왜 느렸나"를 한 번에 추적할 수 있다.
"""

import uuid
from contextvars import ContextVar

TRACE_ID_HEADER = "X-Trace-Id"

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(value: str | None) -> str:
    trace_id = value or uuid.uuid4().hex[:8]
    _trace_id.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    return _trace_id.get()
