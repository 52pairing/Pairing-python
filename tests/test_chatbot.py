"""챗봇 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 파싱 로직만 본다."""

import json

import pytest

from app.clients.gemini import GeminiTask
from app.core.errors import AiException
from app.domains.chatbot.schemas import AskRequest
from app.domains.chatbot.service import ChatbotService


class _FakeGemini:
    def __init__(self, payload: dict | None = None, raw: str | None = None):
        self._payload = payload
        self._raw = raw

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json(self, task, prompt, schema) -> str:
        if self._raw is not None:
            return self._raw
        return json.dumps(self._payload)


async def test_ask_returns_answer_and_model():
    fake = _FakeGemini({"answer": "착수금 수수료는 계약 체결 시점에 발생합니다."})

    result = await ChatbotService(fake).ask(AskRequest(question="착수금 수수료는 언제 결제하나요?"))

    assert result.model == "fake-model"
    assert result.answer == "착수금 수수료는 계약 체결 시점에 발생합니다."


async def test_ask_raises_when_answer_missing():
    fake = _FakeGemini({"answer": "   "})

    with pytest.raises(AiException):
        await ChatbotService(fake).ask(AskRequest(question="아무 질문"))


async def test_ask_raises_when_response_is_not_valid_json():
    fake = _FakeGemini(raw="이건 JSON이 아닙니다")

    with pytest.raises(AiException):
        await ChatbotService(fake).ask(AskRequest(question="아무 질문"))
