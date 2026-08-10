"""협상 A2A 제안 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 파싱/필터 로직만 본다."""

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


def _message(condition_id: int, sender: str, kind: str, value: str) -> dict:
    return {
        "sender": sender,
        "condition_id": condition_id,
        "kind": kind,
        "proposed_value": value,
        "content": "제안 문구",
        "reason": "근거 문구",
    }


async def test_propose_parses_dialogue_and_drops_unknown_condition_ids():
    """요청에 없던 쟁점을 LLM 이 지어내도 대화·결과 양쪽에서 걸러진다."""
    fake = _FakeGemini(
        {
            "messages": [
                _message(401, "CLIENT_AGENT", "PROPOSAL", "4500000"),
                _message(401, "FREELANCER_AGENT", "ACCEPT", "5000000"),
                # 요청에 없던 조건 → 필터링돼야 함
                _message(999, "CLIENT_AGENT", "PROPOSAL", "x"),
            ],
            "outcomes": [
                {"condition_id": 401, "proposed_value": "5000000", "agreed": True},
                {"condition_id": 999, "proposed_value": "x", "agreed": True},
            ],
        }
    )

    result = await NegotiationService(fake).propose(_amount_request())

    assert result.negotiation_id == 300
    assert result.model == "fake-model"

    assert len(result.messages) == 2
    assert {m.condition_id for m in result.messages} == {401}
    assert [m.sender for m in result.messages] == ["CLIENT_AGENT", "FREELANCER_AGENT"]

    assert len(result.outcomes) == 1
    assert result.outcomes[0].condition_id == 401
    assert result.outcomes[0].proposed_value == "5000000"
    assert result.outcomes[0].agreed is True


async def test_propose_raises_when_outcome_missing_for_requested_condition():
    """모든 쟁점에 결과가 있어야 스프링이 라운드를 확정할 수 있다. 누락되면 실패로 본다."""
    fake = _FakeGemini(
        {
            "messages": [_message(401, "CLIENT_AGENT", "PROPOSAL", "4500000")],
            "outcomes": [{"condition_id": 999, "proposed_value": "x", "agreed": True}],
        }
    )
    with pytest.raises(AiException):
        await NegotiationService(fake).propose(_amount_request())


async def test_propose_raises_when_dialogue_is_empty():
    """결과만 있고 대화가 비면 화면에 보여 줄 로그가 없다."""
    fake = _FakeGemini(
        {
            "messages": [],
            "outcomes": [{"condition_id": 401, "proposed_value": "5000000", "agreed": True}],
        }
    )
    with pytest.raises(AiException):
        await NegotiationService(fake).propose(_amount_request())


async def test_propose_raises_when_schema_is_not_a2a():
    """구 스키마(proposals)가 오면 파싱 단계에서 걸러야 한다."""
    fake = _FakeGemini(
        {"proposals": [{"condition_id": 401, "proposed_value": "5000000", "content": "c", "reason": "r"}]}
    )
    with pytest.raises(AiException):
        await NegotiationService(fake).propose(_amount_request())
