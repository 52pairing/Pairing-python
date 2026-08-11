"""A2A 협상 제안 생성.

백엔드(스프링)가 심판(라운드/락/타결)을 맡고, 여기서는 **두 대리인**(클라이언트 대리 /
프리랜서 대리)이 서로 제안·역제안·수락을 주고받는 A2A 대화를 생성한다.

- 클라 대리: 예산 상한·클라 마지노선 안에서 클라에게 유리하게.
- 프리 대리: 프리 마지노선 안에서 프리에게 유리하게.
- 마지노선은 가드로만 쓰고 발언 텍스트에 그대로 노출하지 않는다.
- 모든 발언에 근거(reason)를 붙인다. (요구사항 P13)

한 번 호출로 대화 전체(트랜스크립트)와 쟁점별 최종 결과를 돌려준다. 스프링이 이를 라운드로
기록하고, agreed 조건은 자동 락, 나머지는 사람 승인/재지시로 넘긴다.
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
from app.domains.negotiation.schemas import (
    AgentMessage,
    ConditionOutcome,
    ProposeRequest,
    ProposeResponse,
)

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string", "enum": ["CLIENT_AGENT", "FREELANCER_AGENT"]},
                    "condition_id": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["PROPOSAL", "COUNTER", "ACCEPT"]},
                    "proposed_value": {"type": "string"},
                    "content": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["sender", "condition_id", "kind", "proposed_value", "content", "reason"],
            },
        },
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition_id": {"type": "integer"},
                    "proposed_value": {"type": "string"},
                    "agreed": {"type": "boolean"},
                },
                "required": ["condition_id", "proposed_value", "agreed"],
            },
        },
    },
    "required": ["messages", "outcomes"],
}


class NegotiationService:
    def __init__(
        self,
        gemini: GeminiClient,
        ai_log_repository: AiAgentLogRepository | None = None,
    ):
        self._gemini = gemini
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)
        prompt = self._build_prompt(request)
        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.NEGOTIATION, prompt, _RESPONSE_SCHEMA
            )
        except AiException as exc:
            await self._record_call(request, model, prompt, None, None, str(exc))
            raise

        try:
            messages, outcomes = self._parse(raw, request)
        except AiException as exc:
            # 파싱·검증 실패는 호출 실패보다 원인 찾기가 어렵다. 응답 원문을 남겨야
            # "LLM 이 무엇을 돌려줘서 못 썼는지"를 나중에 되짚을 수 있다.
            await self._record_call(request, model, prompt, raw, usage, str(exc))
            raise

        await self._record_call(request, model, prompt, raw, usage, None)
        return ProposeResponse(
            negotiation_id=request.negotiation_id, model=model, messages=messages, outcomes=outcomes
        )

    def _parse(
        self, raw: str, request: ProposeRequest
    ) -> tuple[list[AgentMessage], list[ConditionOutcome]]:
        try:
            parsed = json.loads(raw)
            messages = [AgentMessage(**m) for m in parsed["messages"]]
            outcomes = [ConditionOutcome(**o) for o in parsed["outcomes"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("A2A 협상 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 요청에 없던 condition_id 를 지어낼 수 있다. 요청한 조건만 남긴다.
        allowed = {c.condition_id for c in request.conditions}
        messages = [m for m in messages if m.condition_id in allowed]
        outcomes = [o for o in outcomes if o.condition_id in allowed]

        # 모든 쟁점에 결과가 있어야 스프링이 라운드를 확정할 수 있다.
        covered = {o.condition_id for o in outcomes}
        missing = allowed - covered
        if missing:
            logger.warning("A2A 결과 누락 쟁점: %s", missing)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "일부 쟁점 결과가 비어 있습니다.")
        if not messages:
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "대화가 비어 있습니다.")

        return messages, outcomes

    async def _record_call(
        self,
        request: ProposeRequest,
        model: str,
        prompt: str,
        raw: str | None,
        usage: GeminiUsage | None,
        error_message: str | None,
    ) -> None:
        """A2A 제안 LLM 호출 1건 = ai_agent_log 1행.

        협상은 stub 폴백이 있어서 파이썬이 실패해도 스프링 쪽은 계속 돈다. 그래서 실패가
        화면에 드러나지 않는다 — 여기 남기지 않으면 "왜 stub 대화가 나왔는지"를 알 수 없다.
        """
        if self._ai_log_repository is None:
            return
        await self._ai_log_repository.record(
            AiCallRecord(
                agent_type=AgentType.NEGOTIATOR,
                status=LogStatus.FAILED if error_message else LogStatus.SUCCESS,
                ref_type=RefType.NEGOTIATION,
                ref_id=request.negotiation_id,
                model=model,
                request_json={"prompt": prompt, "round": request.round},
                response_json={"raw": raw} if raw is not None else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                latency_ms=usage.latency_ms if usage else None,
                retry_count=usage.retry_count if usage else 0,
                error_message=error_message,
            )
        )

    def _build_prompt(self, request: ProposeRequest) -> str:
        lines = []
        for c in request.conditions:
            # 마지노선은 방향을 함께 적는다. "클라마지노선=3300000" 만 주면 LLM 이 그게 상한인지
            # 하한인지 몰라 그 값을 넘겨 합의해 버린다(실제로 프리 하한 480만인데 330만을 수락했다).
            line = (
                f"- condition_id={c.condition_id}, 쟁점={c.type}, "
                f"클라희망={c.client_value}, 프리희망={c.freelancer_value}, "
                f"클라상한(이_값을_초과하면_클라가_거절)={c.client_floor}, "
                f"프리하한(이_값에_못_미치면_프리가_거절)={c.freelancer_floor}"
            )
            # 선택형은 후보를, 형식이 정해진 쟁점은 표기법을 함께 준다(없는 값·형식 이탈 방지).
            if c.allowed_values:
                line += f", 허용값={'|'.join(c.allowed_values)}"
            if c.value_format:
                line += f", 값형식={c.value_format}"
            lines.append(line)
        budget = f"{request.budget_cap}" if request.budget_cap is not None else "미지정"
        return (
            "너는 프리랜서-클라이언트 채용 조건 협상을 **두 AI 대리인의 대화**로 시뮬레이션한다.\n"
            "- CLIENT_AGENT(클라이언트 대리): 예산 상한과 클라 상한 안에서 클라에게 유리하게 협상한다.\n"
            "- FREELANCER_AGENT(프리랜서 대리): 프리 하한 위에서 프리에게 유리하게 협상한다.\n"
            "\n"
            "**가장 중요한 제약 — 마지노선은 넘을 수 없는 선이다.**\n"
            "- 클라상한을 초과하는 값에 CLIENT_AGENT 가 동의해서는 안 된다.\n"
            "- 프리하한에 못 미치는 값에 FREELANCER_AGENT 가 동의해서는 안 된다.\n"
            "- 합의를 성사시키는 것보다 이 선을 지키는 것이 우선이다. "
            "선을 지키면서 합의할 수 없으면 **합의하지 말고 agreed=false 로 남긴다.** "
            "결렬은 실패가 아니라 정상적인 결과다.\n"
            "- 클라상한 < 프리하한 이면 접점이 없다. 이때는 절대 합의하지 말고 agreed=false 로 남긴다.\n"
            "\n"
            "규칙:\n"
            "1) 각 쟁점마다 두 대리인이 번갈아 제안(PROPOSAL)·역제안(COUNTER)·수락(ACCEPT)을 주고받되, "
            "쟁점당 2~4개 발언으로 간결하게 수렴시킨다.\n"
            "2) 금액(AMOUNT)은 프리하한 이상 클라상한 이하에서, 예산 상한을 절대 넘지 않게 합의한다. "
            "그 구간이 비어 있으면 합의하지 않는다.\n"
            "3) 근무형태/방식 등 선택형은 양측 수용 가능한 값으로 합의한다.\n"
            "3-1) 쟁점에 '허용값'이 주어지면 proposed_value 는 **반드시 그 목록 안의 값 그대로**만 쓴다. "
            "목록에 없는 값을 새로 만들지 않는다"
            "(예: 허용값이 REMOTE|ONSITE|ANY 인데 HYBRID 를 쓰면 안 된다). "
            "양측이 겹치는 허용값을 못 찾으면 억지로 만들지 말고 agreed=false 로 남긴다.\n"
            "3-2) 쟁점에 '값형식'이 주어지면 proposed_value 는 그 형식을 정확히 지킨다"
            "(예: 값형식=<숫자> MONTH 이면 '3' 이 아니라 '3 MONTH').\n"
            "4) 두 마지노선이 겹쳐 합의 가능한 쟁점은 "
            "outcomes.agreed=true 와 최종 proposed_value 로 마무리한다. "
            "겹치지 않아 합의 불가한 쟁점은 마지막 역제안 값을 "
            "proposed_value 로 두고 agreed=false 로 남긴다.\n"
            "4-1) outcomes 의 proposed_value 는 "
            "**그 쟁점 대화의 마지막 발언 proposed_value 와 반드시 같아야 한다.** "
            "대화는 330만원으로 끝났는데 outcomes 에 480만원을 적는 식으로 어긋나면 안 된다. "
            "사람이 대화를 읽고 이해한 값이 곧 계약 값이 된다.\n"
            "5) proposed_value 는 설명 없이 값만 담는다(금액은 단위·콤마 없는 숫자 문자열). "
            "'값형식'·'허용값'이 주어진 쟁점은 그 지시가 우선한다.\n"
            "6) content 는 사람에게 보일 한국어 한 문장, reason 은 한국어 한 문장 근거. 모든 발언에 필수.\n"
            "7) 마지노선 숫자를 발언 텍스트에 그대로 노출하지 않는다(가드로만 사용).\n"
            "8) messages 는 모든 쟁점의 대화를 시간순으로, outcomes 는 쟁점별 최종 결과를 담는다.\n"
            f"\n예산 상한(원): {budget}\n"
            f"현재 라운드: {request.round}\n"
            "쟁점 목록:\n" + "\n".join(lines)
        )
