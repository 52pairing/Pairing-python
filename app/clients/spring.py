"""스프링 백엔드 호출 클라이언트.

AI 서버가 DB 를 직접 고치지 않고 스프링에 위임할 때 쓴다.
(예: 매칭 결과 확정, 알림 발송처럼 도메인 규칙이 걸린 작업)

호출할 때 traceId 를 그대로 실어 보내야 두 서버 로그가 이어진다.
"""

import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.context import TRACE_ID_HEADER, get_trace_id
from app.core.errors import AiErrorCode, AiException
from app.core.security import INTERNAL_API_KEY_HEADER

logger = logging.getLogger(__name__)


class SpringClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=settings.spring_base_url,
            timeout=settings.spring_timeout_seconds,
            headers={INTERNAL_API_KEY_HEADER: settings.internal_api_key},
        )

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                path, json=payload, headers={TRACE_ID_HEADER: get_trace_id()}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("스프링 호출 실패: path=%s, cause=%s", path, exc)
            raise AiException(AiErrorCode.SPRING_CALL_FAILED) from exc

        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


_client: SpringClient | None = None


def get_spring_client() -> SpringClient:
    global _client
    if _client is None:
        _client = SpringClient()
    return _client
