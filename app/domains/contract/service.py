"""계약서 문구 정리.

계약서는 법적 문서라 LLM 이 값을 만들어내면 안 된다. 여기서는 **이미 확정된 원문을
계약서 문체로 압축**하는 일만 한다. 금액·기간·날짜·당사자는 스프링이 DB 값으로 직접 채운다.

LLM 이 실패해도 계약 체결이 막히면 안 되므로, 스프링 쪽에서 기본값으로 대체하고 진행한다.
"""

import json
import logging

from app.clients.gemini import GeminiClient, GeminiTask, GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import (
    AgentType,
    AiAgentLogRepository,
    AiCallRecord,
    LogStatus,
    RefType,
)
from app.domains.contract.schemas import DraftTextRequest, DraftTextResponse

logger = logging.getLogger(__name__)

# 계약서 표 칸이 깨지지 않는 길이. 프롬프트로도 제한하지만 LLM 이 어기므로 코드에서 자른다.
_MAX_MAIN_TASK = 80
_MAX_DETAIL_SCOPE = 200
_NO_SPECIAL_TERMS = "별도의 특약사항 없음"

_PARSE_FAILED = "응답 JSON 파싱 실패"
_EMPTY_MAIN_TASK = "담당 업무 요약이 비어 있습니다."

_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "main_task_summary": {"type": "string"},
        "detail_scope_summary": {"type": "string"},
        "special_terms": {"type": "string"},
    },
    "required": ["main_task_summary", "detail_scope_summary", "special_terms"],
}


class ContractService:
    def __init__(
        self,
        gemini: GeminiClient,
        ai_log_repository: AiAgentLogRepository | None = None,
    ):
        self._gemini = gemini
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def draft(self, request: DraftTextRequest) -> DraftTextResponse:
        model = self._gemini.model_for(GeminiTask.CONTRACT)

        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.CONTRACT, self._build_prompt(request), _DRAFT_SCHEMA
            )
        except AiException as exc:
            await self._record_call(request, model, None, None, str(exc))
            raise

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("계약서 문구 응답 파싱 실패: contract_id=%s", request.contract_id)
            # 응답 원문을 남긴다. "LLM 이 무엇을 돌려줘서 못 썼는지"는 나중에 되짚을 수 없다.
            await self._record_call(request, model, raw, usage, _PARSE_FAILED)
            raise AiException(AiErrorCode.CONTRACT_DRAFT_FAILED) from exc

        main_task = self._clip(parsed.get("main_task_summary"), _MAX_MAIN_TASK)
        if not main_task:
            # 담당 업무는 제2조 필수 칸이다. 비면 계약서가 성립하지 않는다.
            await self._record_call(request, model, raw, usage, _EMPTY_MAIN_TASK)
            raise AiException(AiErrorCode.CONTRACT_DRAFT_FAILED, _EMPTY_MAIN_TASK)

        # 원문이 없으면 요약도 없다. LLM 이 지어내는 것을 막는다.
        detail_scope = (
            self._clip(parsed.get("detail_scope_summary"), _MAX_DETAIL_SCOPE)
            if request.detail_scope
            else ""
        )

        # 합의된 게 없으면 특약도 없다. 프롬프트에 써도 LLM 이 문장을 만들어낸다.
        special_terms = (
            self._clip(parsed.get("special_terms"), _MAX_DETAIL_SCOPE) or _NO_SPECIAL_TERMS
            if request.agreed_notes
            else _NO_SPECIAL_TERMS
        )

        await self._record_call(request, model, raw, usage, None)

        return DraftTextResponse(
            contract_id=request.contract_id,
            model=model,
            main_task_summary=main_task,
            detail_scope_summary=detail_scope,
            special_terms=special_terms,
        )

    async def _record_call(
        self,
        request: DraftTextRequest,
        model: str,
        raw: str | None,
        usage: GeminiUsage | None,
        error_message: str | None,
    ) -> None:
        """계약서 문구 LLM 호출 1건 = ai_agent_log 1행.

        스프링은 이 호출이 실패하면 계약을 DRAFT 에 남기고 30분간 재시도한다. 그 사이 스프링
        로그에는 "실패했다"만 남고 **무엇이 실패했는지는 여기밖에 없다** — 키 한도인지, 응답이
        깨졌는지, 요약이 비었는지를 가르려면 이 기록이 필요하다.

        `request_json` 에는 프롬프트가 아니라 원문 셋만 넣는다. 프롬프트의 대부분은 매번 똑같은
        지시문이라 통째로 남기면 모든 행에 같은 내용이 복사된다. 실제로 봐야 하는 건 어떤 원문을
        넣었더니 어떤 요약이 나왔는지다.
        """
        if self._ai_log_repository is None:
            return
        await self._ai_log_repository.record(
            AiCallRecord(
                agent_type=AgentType.CONTRACT,
                status=LogStatus.FAILED if error_message else LogStatus.SUCCESS,
                ref_type=RefType.CONTRACT,
                ref_id=request.contract_id,
                model=model,
                request_json={
                    "main_task": request.main_task,
                    "detail_scope": request.detail_scope,
                    "agreed_notes": request.agreed_notes,
                },
                response_json={"raw": raw} if raw is not None else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                latency_ms=usage.latency_ms if usage else None,
                retry_count=usage.retry_count if usage else 0,
                error_message=error_message,
            )
        )

    @staticmethod
    def _clip(value: object, limit: int) -> str:
        return value.strip()[:limit] if isinstance(value, str) else ""

    def _build_prompt(self, request: DraftTextRequest) -> str:
        notes = "\n".join(f"- {note}" for note in request.agreed_notes) or "(없음)"
        detail = request.detail_scope or "(없음)"

        return (
            "너는 프리랜서 용역 계약서의 문구를 다듬는 보조자다.\n"
            "주어진 원문을 계약서 문체로 압축한다. 없는 내용을 추가하지 않는다.\n"
            "\n"
            "1) main_task_summary\n"
            f"   담당 업무 원문을 한 문장으로 요약한다. {_MAX_MAIN_TASK}자 이내.\n"
            "   명사형으로 끝낸다. 예: \"모바일 앱용 RESTful API 설계 및 개발\"\n"
            "   \"~합니다\", \"~해주세요\" 같은 서술형·청유형을 쓰지 않는다.\n"
            "\n"
            "2) detail_scope_summary\n"
            f"   세부 업무 범위 원문을 한두 문장으로 요약한다. {_MAX_DETAIL_SCOPE}자 이내.\n"
            "   항목을 쉼표로 나열한다.\n"
            "   예: \"Node.js 기반 RESTful API 설계 및 구현, 데이터베이스 스키마 설계 및 최적화\"\n"
            "   원문이 (없음) 이면 빈 문자열을 반환한다.\n"
            "\n"
            "3) special_terms\n"
            "   합의 메모에 있는 내용만 문장으로 정리한다.\n"
            f"   메모가 (없음) 이면 정확히 \"{_NO_SPECIAL_TERMS}\" 을 반환한다.\n"
            "   메모에 없는 조건을 만들어내지 않는다.\n"
            "\n"
            "법적 효력이 있는 문서다. 금액·기간·인원·날짜는 절대 넣지 않는다.\n"
            "그 값들은 계약서의 다른 조항에서 따로 다루며, 여기서 언급하면 서로 어긋난다.\n"
            "\n"
            f"[담당 업무 원문]\n{request.main_task}\n"
            f"\n[세부 업무 범위 원문]\n{detail}\n"
            f"\n[협상 합의 메모]\n{notes}\n"
        )
