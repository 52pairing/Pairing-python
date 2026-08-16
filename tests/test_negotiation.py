"""협상 A2A 제안 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 파싱/필터 로직만 본다."""

import json
from unittest.mock import AsyncMock

import pytest

from app.clients.gemini import GeminiTask, GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import AgentType, LogStatus, RefType
from app.domains.negotiation.schemas import ConditionContext, ProposeRequest
from app.domains.negotiation.service import NegotiationService

_USAGE = GeminiUsage(
    model="fake-model", prompt_tokens=210, output_tokens=88, latency_ms=1520, retry_count=1
)


class _FakeGemini:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json_with_usage(self, task, prompt, schema) -> tuple[str, GeminiUsage]:
        return json.dumps(self._payload), _USAGE


class _FailingGemini:
    """LLM 호출 자체가 실패하는 경우(타임아웃·API 오류)."""

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json_with_usage(self, task, prompt, schema) -> tuple[str, GeminiUsage]:
        raise AiException(AiErrorCode.LLM_CALL_FAILED)


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


async def test_propose_records_ai_agent_log_with_ref_and_tokens():
    """스프링 관리자 화면이 (ref_type, ref_id)로 로그를 찾으므로 이 두 값이 정확해야 한다."""
    ai_log = AsyncMock()
    fake = _FakeGemini(
        {
            "messages": [_message(401, "CLIENT_AGENT", "PROPOSAL", "5000000")],
            "outcomes": [{"condition_id": 401, "proposed_value": "5000000", "agreed": True}],
        }
    )

    await NegotiationService(fake, ai_log).propose(_amount_request())

    ai_log.record.assert_awaited_once()
    call = ai_log.record.await_args.args[0]
    assert call.agent_type == AgentType.NEGOTIATOR
    assert call.ref_type == RefType.NEGOTIATION
    assert call.ref_id == 300
    assert call.status == LogStatus.SUCCESS
    assert call.model == "fake-model"
    assert call.prompt_tokens == 210
    assert call.output_tokens == 88
    assert call.latency_ms == 1520
    assert call.retry_count == 1
    assert call.error_message is None


async def test_propose_records_failed_log_when_llm_call_fails():
    """호출 자체가 실패해도 기록은 남아야 한다 — 스프링은 stub 으로 폴백해 화면엔 안 드러난다."""
    ai_log = AsyncMock()

    with pytest.raises(AiException):
        await NegotiationService(_FailingGemini(), ai_log).propose(_amount_request())

    call = ai_log.record.await_args.args[0]
    assert call.status == LogStatus.FAILED
    assert call.error_message is not None
    # 호출이 안 됐으니 응답·토큰은 비어 있다.
    assert call.response_json is None
    assert call.prompt_tokens is None


def test_build_prompt_carries_last_values_and_no_backward_rule():
    """직전제시가 주어지면 프롬프트에 노출하고, 오프닝 이어가기·역주행 금지 규칙을 싣는다."""
    request = ProposeRequest(
        negotiation_id=300,
        round=3,
        budget_cap=5_000_000,
        conditions=[
            ConditionContext(
                condition_id=401,
                type="AMOUNT",
                client_value="4000000",
                freelancer_value="6000000",
                client_floor="4500000",
                freelancer_floor="5500000",
                client_last_value="4400000",
                freelancer_last_value="5000000",
            )
        ],
    )

    prompt = NegotiationService(_FakeGemini({}))._build_prompt(request)

    # 직전제시가 조건 라인에 노출된다.
    assert "클라직전제시=4400000" in prompt
    assert "프리직전제시=5000000" in prompt
    # 오프닝을 직전제시에서 이어가라는 규칙 + 역주행/작화 금지 제약이 실린다.
    assert "직전제시" in prompt
    assert "세 번째 제약" in prompt


def test_build_prompt_renders_start_date_floor_as_upper_with_available_from():
    """START_DATE 는 프리 마지노선도 상한(MAX)으로 내려온다 — '프리하한'이 아니라 '프리상한' + 가용 시작일."""
    request = ProposeRequest(
        negotiation_id=300,
        round=1,
        conditions=[
            ConditionContext(
                condition_id=402,
                type="START_DATE",
                client_value="2026-09-01",
                freelancer_value="2026-10-01",
                client_floor="2026-09-22",
                freelancer_floor="2026-10-21",
                client_floor_direction="MAX",
                freelancer_floor_direction="MAX",
            )
        ],
    )

    prompt = NegotiationService(_FakeGemini({}))._build_prompt(request)

    # 조건 라인에 프리 마지노선이 '상한'으로 렌더된다(방향 뒤집힘이면 이 구체 문자열이 없다).
    assert "프리상한(이_값을_초과하면_프리가_거절)=2026-10-21" in prompt
    assert "클라상한(이_값을_초과하면_클라가_거절)=2026-09-22" in prompt
    # 프리 물리적 하한(가용 시작일) 명시.
    assert "프리가용시작일(이보다_이르게는_시작_불가)=2026-10-01" in prompt
    # 뒤집힌 '프리하한=10/21' 은 나오면 안 된다(규칙 텍스트의 일반 '프리하한'과 구분되는 구체 라벨).
    assert "프리하한(이_값에_못_미치면_프리가_거절)=2026-10-21" not in prompt


def test_build_prompt_defaults_amount_directions_when_missing():
    """방향 필드가 없으면(구 백엔드) 기존 가정으로 폴백 — 클라=상한, 프리=하한."""
    prompt = NegotiationService(_FakeGemini({}))._build_prompt(_amount_request())
    # _amount_request: client_floor=4500000, freelancer_floor=5500000, 방향 없음 → 폴백.
    assert "클라상한(이_값을_초과하면_클라가_거절)=4500000" in prompt
    assert "프리하한(이_값에_못_미치면_프리가_거절)=5500000" in prompt


def test_build_prompt_omits_last_values_on_first_round():
    """직전제시가 없으면(첫 라운드) 조건 라인에 값 필드를 노출하지 않는다 — 그때는 희망값에서 시작.

    (규칙 설명 텍스트에는 '직전제시'라는 말이 항상 들어가므로, 조건 라인의 값 필드로 확인한다.)
    """
    prompt = NegotiationService(_FakeGemini({}))._build_prompt(_amount_request())
    assert "클라직전제시=" not in prompt
    assert "프리직전제시=" not in prompt


async def test_propose_records_failed_log_with_raw_response_when_parsing_fails():
    """파싱 실패는 응답 원문이 있어야 원인을 되짚을 수 있다."""
    ai_log = AsyncMock()
    fake = _FakeGemini(
        {
            "messages": [_message(401, "CLIENT_AGENT", "PROPOSAL", "5000000")],
            "outcomes": [],  # 요청한 쟁점 결과 누락 → 검증 실패
        }
    )

    with pytest.raises(AiException):
        await NegotiationService(fake, ai_log).propose(_amount_request())

    call = ai_log.record.await_args.args[0]
    assert call.status == LogStatus.FAILED
    assert call.response_json is not None  # 원문 보존
    assert call.prompt_tokens == 210  # 호출은 됐으므로 토큰은 있다
