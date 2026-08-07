"""협상 제안 생성.

백엔드(스프링)가 심판(라운드/락/타결)을 맡고, 여기서는 **중재자 관점**으로 각 쟁점의
수렴 제안값과 근거를 만든다. 양측 희망값·마지노선·예산을 모두 보고 공정한 값을 제안한다.
(두 대리인이 서로 안 보고 밀당하는 완전 A2A 는 후속 확장.)
"""

import json
import logging

from app.clients.gemini import GeminiClient, GeminiTask
from app.core.errors import AiErrorCode, AiException
from app.domains.negotiation.schemas import ConditionProposal, ProposeRequest, ProposeResponse

logger = logging.getLogger(__name__)

_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition_id": {"type": "integer"},
                    "proposed_value": {"type": "string"},
                    "content": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["condition_id", "proposed_value", "content", "reason"],
            },
        }
    },
    "required": ["proposals"],
}


class NegotiationService:
    def __init__(self, gemini: GeminiClient):
        self._gemini = gemini

    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)
        raw = await self._gemini.generate_json(
            GeminiTask.NEGOTIATION, self._build_prompt(request), _PROPOSAL_SCHEMA
        )

        try:
            parsed = json.loads(raw)
            proposals = [ConditionProposal(**item) for item in parsed["proposals"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("협상 제안 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 요청에 없던 condition_id 를 지어낼 수 있다. 요청한 조건만 남긴다.
        allowed = {c.condition_id for c in request.conditions}
        filtered = [p for p in proposals if p.condition_id in allowed]
        if not filtered:
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "제안이 비어 있습니다.")

        return ProposeResponse(negotiation_id=request.negotiation_id, model=model, proposals=filtered)

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
            "너는 프리랜서-클라이언트 협상의 중립 중재자다. 각 쟁점에서 양측이 받아들일 만한 "
            "수렴 제안값을 하나씩 만든다.\n"
            "규칙:\n"
            "1) 금액(AMOUNT)은 양측 마지노선 사이의 값으로, 예산 상한을 넘지 않게 제안한다.\n"
            "2) 근무방식/형태 등 선택형은 양측이 수용 가능한 쪽으로 제안한다.\n"
            "3) proposed_value 는 값만(금액은 숫자 문자열), content 는 한국어 한 문장 제안, "
            "reason 은 한국어 한 문장 근거로 쓴다.\n"
            "4) 마지노선은 참고만 하고 응답에 그대로 노출하지 않는다.\n"
            f"예산 상한(원): {budget}\n"
            f"라운드: {request.round}\n"
            "쟁점 목록:\n" + "\n".join(lines)
        )
