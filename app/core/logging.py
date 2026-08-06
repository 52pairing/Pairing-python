"""로깅 설정. 모든 줄에 traceId 가 붙는다."""

import logging
import sys

from app.core.context import get_trace_id


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s [%(trace_id)s] %(name)s - %(message)s")
    )
    handler.addFilter(TraceIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # 요청마다 SQL 을 다 찍으면 로그가 임베딩 벡터로 뒤덮인다.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
