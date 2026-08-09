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

from app.clients.gemini import GeminiClient, GeminiTask
from app.core.errors import AiErrorCode, AiException
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
    def __init__(self, gemini: GeminiClient):
        self._gemini = gemini

    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)
        raw = await self._gemini.generate_json(
            GeminiTask.NEGOTIATION, self._build_prompt(request), _RESPONSE_SCHEMA
        )

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

        return ProposeResponse(
            negotiation_id=request.negotiation_id, model=model, messages=messages, outcomes=outcomes
        )

    def _build_prompt(self, request: ProposeRequest) -> str:
        lines = []
        for c in request.conditions:
            lines.append(
                f"- condition_id={c.condition_id}, 쟁점={c.type}, "
                f"클라희망={c.client_value}, 프리희망={c.freelancer_value}, "
                f"클라마지노선={c.client_floor}, 프리마지노선={c.freelancer_floor}"
            )
        budget = f"{request.budget_cap}" if request.budget_cap is not None else "미지정"
        return (
            "너는 프리랜서-클라이언트 채용 조건 협상을 **두 AI 대리인의 대화**로 시뮬레이션한다.\n"
            "- CLIENT_AGENT(클라이언트 대리): 예산 상한과 클라 마지노선 안에서 클라에게 유리하게 협상한다.\n"
            "- FREELANCER_AGENT(프리랜서 대리): 프리 마지노선 안에서 프리에게 유리하게 협상한다.\n"
            "\n"
            "규칙:\n"
            "1) 각 쟁점마다 두 대리인이 번갈아 제안(PROPOSAL)·역제안(COUNTER)·수락(ACCEPT)을 주고받되, "
            "쟁점당 2~4개 발언으로 간결하게 수렴시킨다.\n"
            "2) 금액(AMOUNT)은 두 마지노선 사이에서, 예산 상한을 절대 넘지 않게 합의한다.\n"
            "3) 근무형태/방식 등 선택형은 양측 수용 가능한 값으로 합의한다.\n"
            "4) 두 마지노선이 겹쳐 합의 가능한 쟁점은 outcomes.agreed=true 와 최종 proposed_value 로 마무리한다. "
            "겹치지 않아 합의 불가한 쟁점은 마지막 역제안 값을 proposed_value 로 두고 agreed=false 로 남긴다.\n"
            "5) proposed_value 는 값만(금액은 숫자 문자열, 기간은 개월 수 문자열, 선택형은 옵션명).\n"
            "6) content 는 사람에게 보일 한국어 한 문장, reason 은 한국어 한 문장 근거. 모든 발언에 필수.\n"
            "7) 마지노선 숫자를 발언 텍스트에 그대로 노출하지 않는다(가드로만 사용).\n"
            "8) messages 는 모든 쟁점의 대화를 시간순으로, outcomes 는 쟁점별 최종 결과를 담는다.\n"
            f"\n예산 상한(원): {budget}\n"
            f"현재 라운드: {request.round}\n"
            "쟁점 목록:\n" + "\n".join(lines)
        )
