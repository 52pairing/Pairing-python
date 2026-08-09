"""협상 제안 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 파싱/필터 로직만 본다."""

import json

import pytest

from app.clients.gemini import GeminiTask
from app.core.errors import AiException
from app.domains.negotiation.schemas import ConditionContext, ProposeRequest
from app.domains.negotiation.service import NegotiationService


class _FakeGemini:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json(self, task, prompt, schema) -> str:
        return json.dumps(self._payload)


def _amount_request() -> ProposeRequest:
    return ProposeRequest(
        negotiation_id=300,
        round=1,
        budget_cap=5_000_000,
        conditions=[
            ConditionContext(
                condition_id=401,
                type="AMOUNT",
                client_value="4000000",
                freelancer_value="6000000",
                client_floor="4500000",
                freelancer_floor="5500000",
            )
        ],
    )


async def test_propose_parses_and_drops_unknown_condition_ids():
    fake = _FakeGemini(
        {
            "proposals": [
                {
                    "condition_id": 401,
                    "proposed_value": "5000000",
                    "content": "월 500만원을 제안합니다.",
                    "reason": "양측 마지노선 사이 값입니다.",
                },
                # 요청에 없던 조건 → 필터링돼야 함
                {"condition_id": 999, "proposed_value": "x", "content": "c", "reason": "r"},
            ]
        }
    )
    result = await NegotiationService(fake).propose(_amount_request())

    assert result.negotiation_id == 300
    assert result.model == "fake-model"
    assert len(result.proposals) == 1
    assert result.proposals[0].condition_id == 401
    assert result.proposals[0].proposed_value == "5000000"


async def test_propose_raises_when_no_valid_proposal():
    fake = _FakeGemini(
        {"proposals": [{"condition_id": 999, "proposed_value": "x", "content": "c", "reason": "r"}]}
    )
    with pytest.raises(AiException):
        await NegotiationService(fake).propose(_amount_request())
