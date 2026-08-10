"""AI 호출 기록 저장. SQL 은 이 계층 밖으로 새지 않는다."""

import logging
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.ai_log.models import AiAgentLog

logger = logging.getLogger(__name__)


class AgentType(StrEnum):
    """`ai_agent_log.agent_type` 허용값(백엔드 스키마 주석 기준)."""

    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    MATCHER = "MATCHER"
    GUARD = "GUARD"
    NEGOTIATOR = "NEGOTIATOR"
    CONTRACT = "CONTRACT"
    CHATBOT = "CHATBOT"


class RefType(StrEnum):
    """`ai_agent_log.ref_type` — 스프링 관리자 화면이 이 값으로 로그를 찾는다."""

    FREELANCER = "FREELANCER"
    POSITION = "POSITION"
    NEGOTIATION = "NEGOTIATION"
    CONTRACT = "CONTRACT"


class LogStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class AiCallRecord:
    agent_type: AgentType
    status: LogStatus
    ref_type: RefType | None = None
    ref_id: int | None = None
    model: str | None = None
    request_json: dict | None = None
    response_json: dict | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    retry_count: int = 0
    error_message: str | None = None


class AiAgentLogRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(self, call: AiCallRecord) -> None:
        """호출 1건을 남긴다.

        **로그 실패가 본 기능을 죽이면 안 된다.** 추천이 잘 나왔는데 로그 INSERT 하나 때문에
        500이 나가는 게 훨씬 나쁘다. 그래서 여기서 예외를 삼키고 경고만 남긴다.
        """
        try:
            self._session.add(
                AiAgentLog(
                    agent_type=str(call.agent_type),
                    ref_type=str(call.ref_type) if call.ref_type else None,
                    ref_id=call.ref_id,
                    model=call.model,
                    request_json=call.request_json,
                    response_json=call.response_json,
                    prompt_tokens=call.prompt_tokens,
                    output_tokens=call.output_tokens,
                    cost_amount=None,  # 위 models.py 주석 참고
                    latency_ms=call.latency_ms,
                    retry_count=call.retry_count,
                    status=str(call.status),
                    error_message=call.error_message,
                )
            )
            await self._session.flush()
        except Exception as exc:  # noqa: BLE001 - 로그 실패는 본 기능을 막지 않는다
            logger.warning(
                "ai_agent_log 기록 실패(무시하고 진행): agent_type=%s ref=%s/%s cause=%s",
                call.agent_type,
                call.ref_type,
                call.ref_id,
                exc,
            )
