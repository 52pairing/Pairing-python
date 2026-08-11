"""FAQ 챗봇 답변 생성.

세션/사용량 관리는 스프링(심판) 소관이다. 여기는 질문 하나를 받아 정책 범위 안에서
답하는 역할만 한다. 자유 텍스트를 그대로 받으면 프롬프트가 바뀔 때마다 파싱이 깨지므로,
협상 도메인과 같은 방식으로 response_schema 로 구조를 강제한다.

전용 GeminiTask/설정을 새로 추가하지 않고 기존 NEGOTIATION 용도(자연어 생성 계열)를
그대로 빌려 쓴다. 챗봇 전용 모델이 필요해지면 core/config.py・clients/gemini.py에
GeminiTask.CHATBOT을 추가하고 이 파일의 참조만 바꾸면 된다.
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
)
from app.domains.chatbot.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# 답변과 이어지는 화면 코드. LLM 에게 URL 을 만들게 하면 없는 경로를 지어내므로,
# 고를 수 있는 값을 여기서 닫아 둔다. 실제 경로 매핑은 스프링이 한다.
_INTENTS = [
    "RESUME_EDIT",
    "PAYMENT_METHOD",
    "MY_PROJECTS",
    "INQUIRY_NEW",
    "NONE",
]

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "intent": {"type": "string", "enum": _INTENTS},
    },
    "required": ["answer", "intent"],
}

_INTENT_GUIDE = """
- RESUME_EDIT: 이력서·포트폴리오 작성이나 수정을 안내할 때
- PAYMENT_METHOD: 카드·계좌 등 결제수단 등록·변경을 안내할 때
- MY_PROJECTS: 내 프로젝트나 내 계약의 진행 상황을 확인하라고 안내할 때
- INQUIRY_NEW: 답할 수 없어 1:1 문의를 권할 때
- NONE: 위에 해당하지 않거나 단순 설명으로 끝날 때
"""

# 초기 버전은 정책 요약을 프롬프트에 직접 박아 넣는다(RAG 없음). 정책이 바뀌면 여기를 갱신한다.
_POLICY_CONTEXT = """
- 매칭: 프로젝트를 등록하면 AI가 조건에 맞는 프리랜서를 추천한다.
- 매칭 요청: 클라이언트가 매칭을 요청하고 프리랜서가 수락하면 AI 대리인 협상이 시작된다.
- 협상: AI 대리인이 금액·기간·근무방식 등 조건을 자동으로 조율한다.
- 협상 결렬: 정해진 라운드 안에 타결되지 않으면 결렬된다.
- 계약: 협상이 타결되면 표준계약서가 자동 생성되고, 양측이 서명해야 발효된다.
- 결제/수수료: 착수금 수수료는 계약 체결 시점에 발생하고, 성공보수 수수료는 프로젝트 완료 후 정산된다.
- 재추천: 무료 재추천은 정해진 횟수까지 무료이고, 초과하면 유료 재추천을 사용해야 한다.
- 리뷰: 계약이 완료되고 정산까지 끝나야 상대 평가와 사이트 후기를 작성할 수 있다.
- 회원 탈퇴: 진행 중인 프로젝트나 미납 요금이 있으면 탈퇴할 수 없다.
"""


class ChatbotService:
    def __init__(self, gemini: GeminiClient, ai_log_repository: AiAgentLogRepository | None = None):
        self._gemini = gemini
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def ask(self, request: AskRequest) -> AskResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)
        prompt = self._build_prompt(request.question)

        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.NEGOTIATION, prompt, _ANSWER_SCHEMA
            )
        except AiException as exc:
            await self._record_call(request.question, model, None, None, str(exc))
            raise
        await self._record_call(request.question, model, raw, usage, None)

        try:
            parsed = json.loads(raw)
            answer = parsed["answer"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("챗봇 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "답변이 비어 있습니다.")

        return AskResponse(answer=answer.strip(), intent=self._read_intent(parsed), model=model)

    @staticmethod
    def _read_intent(parsed: dict) -> str:
        """목록에 없는 값이면 NONE 으로 떨어뜨린다.

        response_schema 의 enum 으로 이미 한 번 막지만, 모델이 어기는 경우가 있다. 여기서
        걸러 두면 스프링이 매핑에 실패할 일이 없고, 최악이라도 버튼만 안 나온다.
        """
        intent = parsed.get("intent")
        return intent if intent in _INTENTS else "NONE"

    async def _record_call(
        self,
        question: str,
        model: str,
        raw: str | None,
        usage: GeminiUsage | None,
        error_message: str | None,
    ) -> None:
        """챗봇 LLM 호출 1건 = ai_agent_log 1행.

        `ref_type`/`ref_id` 는 비운다. 챗봇은 연관 리소스가 없다 — 질문 하나만 받고 세션은
        스프링이 관리한다.

        `request_json` 에는 프롬프트 전체가 아니라 질문만 넣는다. 프롬프트의 대부분은 매번 똑같은
        정책 텍스트라, 통째로 남기면 모든 행에 같은 내용이 복사되면서 로그만 커진다. 실제로 봐야
        하는 건 사용자가 무엇을 물었는지다.
        """
        if self._ai_log_repository is None:
            return
        await self._ai_log_repository.record(
            AiCallRecord(
                agent_type=AgentType.CHATBOT,
                status=LogStatus.FAILED if error_message else LogStatus.SUCCESS,
                model=model,
                request_json={"question": question},
                response_json={"raw": raw} if raw is not None else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                latency_ms=usage.latency_ms if usage else None,
                retry_count=usage.retry_count if usage else 0,
                error_message=error_message,
            )
        )

    def _build_prompt(self, question: str) -> str:
        return (
            "너는 '페어링' 플랫폼의 FAQ 챗봇이다. 아래 정책 범위 안에서만 한국어로 간결하게 답한다.\n"
            "정책에 없는 내용이거나 개인정보·법률·의료 등 답할 수 없는 질문이면, 아는 척하지 말고 "
            "1:1 문의를 이용해 달라고 안내한다.\n"
            "answer 와 함께 intent 를 하나 고른다. 답변을 읽은 사용자가 바로 갈 만한 화면이 "
            "있을 때만 고르고, 애매하면 NONE 을 쓴다. answer 안에 링크나 경로를 쓰지 않는다.\n"
            f"[intent]\n{_INTENT_GUIDE}\n"
            f"[정책]\n{_POLICY_CONTEXT}\n"
            f"[질문]\n{question}"
        )
