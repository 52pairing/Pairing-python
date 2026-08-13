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
from app.core.config import get_settings
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import (
    AgentType,
    AiAgentLogRepository,
    AiCallRecord,
    LogStatus,
)
from app.domains.chatbot.repository import ChatbotKnowledgeRepository
from app.domains.chatbot.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

# 범위 밖 질문에 돌려주는 고정 문구. LLM 이 만들지 않는다 — 매번 말이 달라지면 안 된다.
# 1:1 문의를 권하지 않는다. "1+1은?" 에 문의를 유도하면 관리자에게 이상한 문의만 쌓인다.
_OUT_OF_SCOPE_ANSWER = (
    "페어링 서비스 관련 질문에만 답변드릴 수 있어요. 이용 방법이나 정책에 대해 물어봐 주세요."
)

# 답변과 이어지는 화면 코드. LLM 에게 URL 을 만들게 하면 없는 경로를 지어내므로,
# 고를 수 있는 값을 여기서 닫아 둔다. 실제 경로 매핑은 스프링이 한다.
_INTENTS = [
    "RESUME_EDIT",
    "PROJECT_CREATE",
    "PAYMENT_METHOD",
    "SETTLEMENTS",
    "MY_PROJECTS",
    "NEGOTIATION_LIST",
    "CONTRACTS",
    "INQUIRY_NEW",
    "NONE",
]

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "intent": {"type": "string", "enum": _INTENTS},
        # 임베딩 게이트를 통과했더라도 모델이 다시 한 번 판단한다. 2중 방어다.
        "out_of_scope": {"type": "boolean"},
    },
    "required": ["answer", "intent", "out_of_scope"],
}

_INTENT_GUIDE = """
- RESUME_EDIT: 이력서·포트폴리오 작성이나 수정을 안내할 때 (프리랜서 화면)
- PROJECT_CREATE: 프로젝트 등록 방법을 안내할 때 (클라이언트 화면)
- PAYMENT_METHOD: 카드·계좌 등 결제수단 등록·변경을 안내할 때
- SETTLEMENTS: 착수금·성공보수 등 수수료 금액이나 결제 내역을 안내할 때
- MY_PROJECTS: 프로젝트 진행 상황, 매칭·추천·재추천을 안내할 때
- NEGOTIATION_LIST: AI 대리인 협상의 진행·라운드·결렬을 안내할 때.
  금액 제안·조건 변경처럼 챗봇이 대신 해줄 수 없는 협상 행위를 요청받았을 때도 이걸 고른다.
- CONTRACTS: 계약서 작성·서명·계약 상태를 안내할 때
- INQUIRY_NEW: 페어링 서비스에 관한 질문인데 정책 범위 밖이라 답할 수 없을 때
- NONE: 위에 해당하지 않을 때. 단순 설명으로 끝나는 경우, 인사, 그리고
  날씨·상식처럼 페어링 서비스와 무관한 질문도 여기에 해당한다.
  (서비스와 무관한 질문에 1:1 문의를 권하면 안 된다)

질문한 사람의 역할(클라이언트/프리랜서)은 알 수 없다. 내용만 보고 고르면 되고,
역할에 맞지 않는 화면은 스프링이 걸러낸다.
"""

# 정책 요약을 프롬프트에 직접 박아 넣는다(RAG 없음). 정책이 바뀌면 여기를 갱신한다.
#
# 숫자를 반드시 적는다. "정해진 횟수", "정해진 라운드" 처럼 쓰면 모델이 그 표현을 그대로
# 되뇌어서 아무것도 알려주지 못하는 답이 나간다. 실제로 그렇게 돌고 있었다 —
# 컨텍스트에 값이 있는 항목만 정확하고 나머지는 전부 얼버무렸다.
#
# 출처: docs/spec/policy.md (P01·P34~P41·P48~P51·P61), db/init/02-create-schema.sql 상단 상수표,
#       Negotiation.MAX_ROUND=15, ChatbotQuota.DAILY_LIMIT=10, DepositFeePolicy, SuccessFeePolicy
_POLICY_CONTEXT = """
[매칭]
- 클라이언트가 프로젝트를 등록하면 AI가 조건에 맞는 프리랜서를 추천한다.
- 클라이언트가 매칭을 요청하면 프리랜서는 3일 안에 수락/거절해야 한다. 3일이 지나면 자동 만료된다.
- 프리랜서가 수락하면 AI 대리인 협상이 시작된다.

[재추천]
- 무료 재추천: 프로젝트당 1회. 요청받은 프리랜서가 전부 거절했거나 3일이 지나 자동 만료된 경우에만 쓸 수 있다.
  협상 결렬·계약 파기·중도 종료는 무료 재추천 조건에 해당하지 않는다.
- 유료 재추천: 프로젝트당 최대 5회, 1회 10,000원.

[협상]
- AI 대리인이 금액·기간·근무방식 등 조건을 자동으로 조율한다.
- 라운드 상한은 15회다. 15회 안에 타결되지 않으면 자동으로 결렬된다.
- 협상 중 언제든 포기할 수 있고, 그러면 그 자리에서 결렬된다.

[계약]
- 협상이 타결되면 표준계약서가 자동 생성된다. 양측이 모두 서명해야 발효된다.

[수수료]
- 착수금 수수료는 계약 체결 시점, 성공보수 수수료는 프로젝트 완료 후 정산된다.
- 프로젝트 금액 1억 원 미만: 착수금 클라이언트 3% / 프리랜서 4%, 성공보수 클라이언트 7% / 프리랜서 6%.
- 1억 원 이상: 착수금 클라이언트 2% / 프리랜서 4%, 성공보수 클라이언트 6% / 프리랜서 6%.
- 최고 등급(클라이언트 다이아 / 프리랜서 마스터)은 착수금·성공보수에서 각각 1%씩, 총 2% 인하된다.

[등급]
- 프리랜서: 주니어(기본) → 시니어(평균 별점 3점 이상 + 완료 5건 이상) → 마스터(4점 이상 + 10건 이상).
- 클라이언트: 실버(기본) → 골드(3점 이상 + 10건 이상) → 다이아(4점 이상 + 20건 이상).
- 매월 1일에 자동으로 다시 산정된다. 조건을 채워도 그 즉시 오르지 않고 다음 1일에 반영된다.
- 최근 프로젝트 경험이 없으면 등급이 한 단계 내려간다(프리랜서 6개월 / 클라이언트 12개월 기준).

[리뷰]
- 프로젝트가 종료되고 성공보수 수수료 결제까지 끝나야 작성할 수 있다.
- 상대 평가와 사이트 이용 후기를 함께 작성한다. 별점은 둘 다 필수, 글은 둘 다 선택이며 500자까지다.
- 한 번 작성하면 수정·삭제할 수 없다.

[회원 탈퇴]
- 진행 중인 프로젝트·계약·협상이 있거나 미납 수수료가 있으면 탈퇴할 수 없다.
- 탈퇴 후 30일 동안 같은 이메일·휴대폰으로 같은 역할로는 다시 가입할 수 없다.
- 완료된 프로젝트·계약·리뷰·협상 채팅은 남는다. 개인정보는 1년 뒤 파기된다.

[챗봇]
- AI 상담은 하루 10회까지다. 매일 초기화된다.
"""

# 화면 경로. "작성 및 수정 메뉴" 같은 두루뭉술한 안내를 막으려고 실제 메뉴 이름을 적어 둔다.
# 버튼(intent)이 같이 나가더라도 답변 본문이 어디로 가야 하는지 말해줘야 한다 —
# 버튼만 있고 설명이 없으면 사용자는 그 버튼이 어디로 가는지 모른 채 눌러야 한다.
_SCREEN_GUIDE = """
- 이력서·포트폴리오 작성/수정: 마이페이지 > 내 이력서/포트폴리오
- 프로젝트 등록: 프로젝트 등록 화면 (클라이언트만)
- 결제수단(카드·계좌) 등록/변경: 마이페이지 > 결제수단
- 수수료 결제 내역: 마이페이지 > 수수료 결제 내역
- 등급과 혜택: 마이페이지 > 등급 및 혜택
- 받은 리뷰 / 작성한 리뷰: 마이페이지 > 리뷰 관리
- 회원 탈퇴: 마이페이지 > 회원 탈퇴
- 협상 진행 상황: 협상 목록
- 계약서 확인·서명: 계약 관리
- 1:1 문의: 고객센터 > 1:1 문의
"""


class ChatbotService:
    def __init__(
        self,
        gemini: GeminiClient,
        ai_log_repository: AiAgentLogRepository | None = None,
        knowledge_repository: ChatbotKnowledgeRepository | None = None,
    ):
        self._gemini = gemini
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository
        # 없으면 관련성 게이트를 건너뛴다. 게이트는 부가 장치라, 이것 때문에 챗봇이 멈추면 안 된다.
        self._knowledge_repository = knowledge_repository
        self._settings = get_settings()

    async def ask(self, request: AskRequest) -> AskResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)

        if not await self._is_relevant(request.question):
            # LLM 을 부르지 않고 여기서 끝낸다. 이게 이 게이트의 존재 이유다 —
            # 무관한 질문에 토큰을 쓰고 사용자 한도까지 깎던 것을 막는다.
            return AskResponse(answer=_OUT_OF_SCOPE_ANSWER, intent="NONE", model=model, out_of_scope=True)

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

        # 빈 답변 검사보다 먼저 본다. 범위 밖이면 answer 를 비우라고 지시했으므로,
        # 순서를 바꾸면 정상 동작이 LLM_RESPONSE_INVALID 로 터진다.
        #
        # 게이트를 통과했어도 LLM 이 "이건 우리 얘기가 아니다"라고 판단할 수 있다. 2중 방어다 —
        # 임계값을 느슨하게 잡아 둔 만큼 통과하는 질문이 생기고, 그건 여기서 걸린다.
        if parsed.get("out_of_scope") is True:
            logger.info("[챗봇 범위 밖 - LLM 판정] 질문=%s", request.question)
            return AskResponse(
                answer=_OUT_OF_SCOPE_ANSWER, intent="NONE", model=model, out_of_scope=True
            )

        if not isinstance(answer, str) or not answer.strip():
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "답변이 비어 있습니다.")

        return AskResponse(answer=answer.strip(), intent=self._read_intent(parsed), model=model)

    async def _is_relevant(self, question: str) -> bool:
        """질문이 페어링 정책 범위 안인지 임베딩 유사도로 판정한다.

        판정할 수 없는 상황(지식 미시딩, 임베딩 실패)은 <b>전부 통과</b>시킨다.
        게이트는 부가 장치다. 판정이 안 된다고 막아버리면 정상 질문까지 전부 차단된다 —
        무관한 질문에 답하는 것보다 훨씬 나쁘다.
        """
        if self._knowledge_repository is None:
            return True

        try:
            vectors = await self._gemini.embed([question])
            nearest = await self._knowledge_repository.find_nearest(vectors[0])
        except Exception:
            logger.warning("챗봇 관련성 판정 실패 - 통과시킨다", exc_info=True)
            return True

        if nearest is None:
            logger.warning("챗봇 지식 청크가 비어 있다 - 관련성 판정을 건너뛴다")
            return True

        threshold = self._settings.chatbot_relevance_threshold
        relevant = nearest.similarity >= threshold

        if not relevant:
            # 차단된 질문을 남긴다. 임계값을 조이거나 청크를 보강할 근거가 이 로그다.
            logger.info(
                "[챗봇 범위 밖 차단] similarity=%.3f (임계 %.2f), 최근접=%s, 질문=%s",
                nearest.similarity,
                threshold,
                nearest.chunk_key,
                question,
            )
        return relevant

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
            "너는 '페어링' 플랫폼의 FAQ 챗봇이다. 아래 정책 범위 안에서만 한국어로 답한다.\n"
            "\n"
            "[답변 규칙]\n"
            "1. 정책에 숫자가 있으면 반드시 그 숫자를 말한다. '정해진 횟수', '일정 기간' 처럼 "
            "얼버무리지 않는다. 물어본 이유가 그 숫자를 알고 싶어서다.\n"
            "2. 어디서 하는지 묻는 질문이면 [화면] 의 메뉴 이름을 그대로 answer 에 적는다. "
            "(예: '마이페이지 > 내 이력서/포트폴리오 에서 작성하실 수 있습니다')\n"
            "3. URL 이나 /mypage/resume 같은 경로는 쓰지 않는다. 메뉴 이름만 쓴다.\n"
            "4. 2~3문장으로 끝낸다. 조건이나 예외가 있으면 그것까지 말하고, 없는 말은 덧붙이지 않는다.\n"
            "5. 페어링 정책 범위인데 아래 내용에 없으면, 아는 척하지 말고 1:1 문의를 안내한다. "
            "**정책에 있는 내용을 1:1 문의로 넘기지 않는다.**\n"
            "\n"
            "[out_of_scope]\n"
            "페어링 서비스와 <b>무관한</b> 질문이면 out_of_scope 를 true 로 하고 answer 는 빈 문자열로 둔다. "
            "수학 계산·날씨·상식·잡담·타사 서비스가 여기 해당한다. 절대 답을 알려주지 않는다 — "
            "'1+1은?' 에 '2입니다' 라고 답하면 안 된다.\n"
            "페어링 이용 방법·정책에 관한 질문이면 답할 수 있든 없든 out_of_scope 는 false 다. "
            "(정책에 없어서 못 답하는 것과 우리 서비스 얘기가 아닌 것은 다르다)\n"
            "\n"
            "answer 와 함께 intent 를 하나 고른다. 답변을 읽은 사용자가 바로 갈 만한 화면이 "
            "있을 때만 고르고, 애매하면 NONE 을 쓴다.\n"
            f"[intent]\n{_INTENT_GUIDE}\n"
            f"[화면]\n{_SCREEN_GUIDE}\n"
            f"[정책]\n{_POLICY_CONTEXT}\n"
            f"[질문]\n{question}"
        )
