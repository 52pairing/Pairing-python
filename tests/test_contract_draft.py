"""계약서 문구 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 가드 로직만 본다.

계약서는 법적 문서라 LLM 이 입력에 없던 내용을 만들어내면 안 된다.
프롬프트로도 막지만 LLM 은 어기므로, 코드 가드가 실제로 도는지 여기서 확인한다.
"""

import json

import pytest

from app.clients.gemini import GeminiTask
from app.core.errors import AiException
from app.domains.contract.schemas import DraftTextRequest
from app.domains.contract.service import ContractService


class _FakeGemini:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json(self, task, prompt, schema) -> str:
        return json.dumps(self._payload)


def _request(detail_scope: str | None = None, agreed_notes: list[str] | None = None) -> DraftTextRequest:
    return DraftTextRequest(
        contract_id=600,
        main_task="모바일 앱 백엔드 API 를 설계하고 개발합니다. REST API 연동이 포함됩니다.",
        detail_scope=detail_scope,
        agreed_notes=agreed_notes or [],
    )


async def test_draft_returns_summaries():
    fake = _FakeGemini(
        {
            "main_task_summary": "모바일 앱용 RESTful API 설계 및 개발",
            "detail_scope_summary": "약 15개 엔드포인트 구현, AWS 배포 파이프라인 구축",
            "special_terms": "매주 월요일 오전 스크럼에 참여한다.",
        }
    )
    result = await ContractService(fake).draft(
        _request(detail_scope="엔드포인트 15개, AWS 배포", agreed_notes=["매주 스크럼 참여"])
    )

    assert result.contract_id == 600
    assert result.model == "fake-model"
    assert result.main_task_summary == "모바일 앱용 RESTful API 설계 및 개발"
    assert result.detail_scope_summary.startswith("약 15개 엔드포인트")
    assert result.special_terms == "매주 월요일 오전 스크럼에 참여한다."


async def test_special_terms_is_fixed_when_no_agreed_notes():
    """합의 메모가 없으면 LLM 이 특약을 지어내도 고정 문구로 덮는다."""
    fake = _FakeGemini(
        {
            "main_task_summary": "모바일 앱용 RESTful API 설계 및 개발",
            "detail_scope_summary": "",
            "special_terms": "을은 매주 보고서를 제출한다.",  # 합의된 적 없는 조건
        }
    )
    result = await ContractService(fake).draft(_request(agreed_notes=[]))

    assert result.special_terms == "별도의 특약사항 없음"


async def test_detail_scope_is_empty_when_source_is_none():
    """원문이 없으면 요약도 비어야 한다. 없는 업무 범위가 계약서에 실리면 안 된다."""
    fake = _FakeGemini(
        {
            "main_task_summary": "모바일 앱용 RESTful API 설계 및 개발",
            "detail_scope_summary": "데이터베이스 스키마 설계 및 최적화",  # 원문에 없던 내용
            "special_terms": "별도의 특약사항 없음",
        }
    )
    result = await ContractService(fake).draft(_request(detail_scope=None))

    assert result.detail_scope_summary == ""


async def test_long_summaries_are_clipped():
    """길이 제한을 넘기면 잘라낸다. 계약서 표 칸이 깨진다."""
    fake = _FakeGemini(
        {
            "main_task_summary": "가" * 200,
            "detail_scope_summary": "나" * 500,
            "special_terms": "다" * 500,
        }
    )
    result = await ContractService(fake).draft(
        _request(detail_scope="원문 있음", agreed_notes=["합의 있음"])
    )

    assert len(result.main_task_summary) == 80
    assert len(result.detail_scope_summary) == 200
    assert len(result.special_terms) == 200


async def test_draft_raises_when_main_task_summary_is_empty():
    """담당 업무는 제2조 필수 칸이라 비면 계약서가 성립하지 않는다."""
    fake = _FakeGemini(
        {"main_task_summary": "   ", "detail_scope_summary": "", "special_terms": "별도의 특약사항 없음"}
    )
    with pytest.raises(AiException):
        await ContractService(fake).draft(_request())
