"""챗봇 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 파싱 로직과 ai_agent_log 기록만 본다."""

import json
from unittest.mock import AsyncMock

import pytest

from app.clients.gemini import GeminiTask, GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import AgentType, LogStatus
from app.domains.chatbot.schemas import AskRequest
from app.domains.chatbot.service import ChatbotService

_MODEL = "fake-model"


class _FakeGemini:
    def __init__(self, payload: dict | None = None, raw: str | None = None, error: AiException | None = None):
        self._payload = payload
        self._raw = raw
        self._error = error

    def model_for(self, task: GeminiTask) -> str:
        return _MODEL

    async def generate_json_with_usage(
        self, task, prompt, schema, *, temperature=None
    ) -> tuple[str, GeminiUsage]:
        if self._error is not None:
            raise self._error
        raw = self._raw if self._raw is not None else json.dumps(self._payload)
        return raw, GeminiUsage(
            model=_MODEL,
            prompt_tokens=120,
            output_tokens=35,
            latency_ms=940,
            retry_count=0,
        )


async def test_ask_returns_answer_and_model():
    fake = _FakeGemini({"answer": "착수금 수수료는 계약 체결 시점에 발생합니다."})

    result = await ChatbotService(fake).ask(AskRequest(question="착수금 수수료는 언제 결제하나요?"))

    assert result.model == _MODEL
    assert result.answer == "착수금 수수료는 계약 체결 시점에 발생합니다."


async def test_ask_raises_when_answer_missing():
    fake = _FakeGemini({"answer": "   "})

    with pytest.raises(AiException):
        await ChatbotService(fake).ask(AskRequest(question="아무 질문"))


async def test_ask_raises_when_response_is_not_valid_json():
    fake = _FakeGemini(raw="이건 JSON이 아닙니다")

    with pytest.raises(AiException):
        await ChatbotService(fake).ask(AskRequest(question="아무 질문"))


async def test_ask_records_ai_agent_log_without_ref():
    """챗봇은 연관 리소스가 없다 — ref_type/ref_id 를 비워야 한다(질문 하나만 받는다)."""
    ai_log = AsyncMock()
    fake = _FakeGemini({"answer": "착수금 수수료는 계약 체결 시점에 발생합니다."})

    await ChatbotService(fake, ai_log).ask(AskRequest(question="착수금 수수료는 언제 결제하나요?"))

    ai_log.record.assert_awaited_once()
    call = ai_log.record.await_args.args[0]
    assert call.agent_type == AgentType.CHATBOT
    assert call.status == LogStatus.SUCCESS
    assert call.ref_type is None
    assert call.ref_id is None
    assert call.model == _MODEL
    assert call.prompt_tokens == 120
    assert call.output_tokens == 35
    assert call.latency_ms == 940
    assert call.error_message is None


async def test_ask_logs_question_only_not_whole_prompt():
    """프롬프트의 대부분은 매번 같은 정책 텍스트다. 통째로 남기면 모든 행에 복사돼 로그만 커진다."""
    ai_log = AsyncMock()
    fake = _FakeGemini({"answer": "답변"})

    await ChatbotService(fake, ai_log).ask(AskRequest(question="협상은 몇 번까지 가능한가요?"))

    call = ai_log.record.await_args.args[0]
    assert call.request_json == {"question": "협상은 몇 번까지 가능한가요?"}
    # 정책 텍스트가 딸려 들어가지 않았는지 확인한다.
    assert "회원 탈퇴" not in json.dumps(call.request_json, ensure_ascii=False)


async def test_ask_records_failed_log_when_llm_call_fails():
    """실패 기록이 제일 필요한 로그다. 예외를 다시 던지기 전에 FAILED 행을 남겨야 한다."""
    ai_log = AsyncMock()
    fake = _FakeGemini(error=AiException(AiErrorCode.LLM_CALL_FAILED))

    with pytest.raises(AiException):
        await ChatbotService(fake, ai_log).ask(AskRequest(question="아무 질문"))

    ai_log.record.assert_awaited_once()
    call = ai_log.record.await_args.args[0]
    assert call.agent_type == AgentType.CHATBOT
    assert call.status == LogStatus.FAILED
    assert call.error_message is not None
    assert call.response_json is None
    # 토큰 정보는 못 받았지만 로그 자체는 남는다.
    assert call.prompt_tokens is None
    assert call.retry_count == 0


async def test_ask_works_without_log_repository():
    """로그 리포지토리를 안 넣어도 본 기능은 그대로 동작한다."""
    fake = _FakeGemini({"answer": "답변"})

    result = await ChatbotService(fake).ask(AskRequest(question="아무 질문"))

    assert result.answer == "답변"
