"""계약서 문구 서비스 검증. 실제 Gemini 없이 가짜 클라이언트로 가드 로직만 본다.

계약서는 법적 문서라 LLM 이 입력에 없던 내용을 만들어내면 안 된다.
프롬프트로도 막지만 LLM 은 어기므로, 코드 가드가 실제로 도는지 여기서 확인한다.
"""

import json

import pytest

from app.clients.gemini import GeminiTask, GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import AgentType, AiCallRecord, LogStatus, RefType
from app.domains.contract.schemas import DraftTextRequest
from app.domains.contract.service import ContractService

_USAGE = GeminiUsage(
    model="fake-model", prompt_tokens=120, output_tokens=40, latency_ms=350, retry_count=1
)


class _FakeGemini:
    def __init__(self, payload: dict | str):
        self._payload = payload

    def model_for(self, task: GeminiTask) -> str:
        return "fake-model"

    async def generate_json_with_usage(self, task, prompt, schema) -> tuple[str, GeminiUsage]:
        raw = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return raw, _USAGE


class _RaisingGemini(_FakeGemini):
    """Gemini 호출 자체가 실패하는 경우. 키 소진·타임아웃이 여기로 온다."""

    def __init__(self):
        super().__init__({})

    async def generate_json_with_usage(self, task, prompt, schema):
        raise AiException(AiErrorCode.LLM_CALL_FAILED)


class _FakeLogRepository:
    """ai_agent_log 대신 메모리에 쌓는다. 실제 세션을 열지 않는다."""

    def __init__(self):
        self.records: list[AiCallRecord] = []

    async def record(self, call: AiCallRecord) -> None:
        self.records.append(call)


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


# ---------------------------------------------------------------------------
# ai_agent_log 기록
#
# 스프링은 이 호출이 실패하면 계약을 DRAFT 에 남기고 30분간 재시도한다. 그 사이 스프링
# 로그에는 "실패했다"만 남으므로, 무엇이 실패했는지는 이 기록에서만 확인할 수 있다.
# ---------------------------------------------------------------------------


async def test_success_is_recorded_with_usage():
    fake = _FakeGemini(
        {
            "main_task_summary": "모바일 앱용 RESTful API 설계 및 개발",
            "detail_scope_summary": "엔드포인트 15개 구현",
            "special_terms": "별도의 특약사항 없음",
        }
    )
    log = _FakeLogRepository()

    await ContractService(fake, log).draft(_request(detail_scope="엔드포인트 15개"))

    assert len(log.records) == 1
    record = log.records[0]
    assert record.agent_type is AgentType.CONTRACT
    assert record.ref_type is RefType.CONTRACT
    assert record.ref_id == 600
    assert record.status is LogStatus.SUCCESS
    assert record.model == "fake-model"
    assert record.latency_ms == 350
    assert record.retry_count == 1
    assert record.error_message is None
    # 프롬프트가 아니라 원문을 남긴다. 어떤 입력이 어떤 요약이 됐는지가 봐야 할 것이다.
    assert record.request_json["main_task"].startswith("모바일 앱 백엔드")


async def test_gemini_failure_is_recorded_without_response():
    log = _FakeLogRepository()

    with pytest.raises(AiException):
        await ContractService(_RaisingGemini(), log).draft(_request())

    assert len(log.records) == 1
    record = log.records[0]
    assert record.status is LogStatus.FAILED
    # 호출 자체가 실패했으니 남길 응답이 없다.
    assert record.response_json is None


async def test_parse_failure_records_raw_response():
    """파싱 실패는 호출 실패보다 원인 찾기가 어렵다. 응답 원문이 있어야 되짚을 수 있다."""
    log = _FakeLogRepository()

    with pytest.raises(AiException):
        await ContractService(_FakeGemini("이건 JSON 이 아니다"), log).draft(_request())

    assert len(log.records) == 1
    record = log.records[0]
    assert record.status is LogStatus.FAILED
    assert record.response_json == {"raw": "이건 JSON 이 아니다"}


async def test_empty_main_task_is_recorded_as_failure():
    fake = _FakeGemini(
        {"main_task_summary": "   ", "detail_scope_summary": "", "special_terms": "별도의 특약사항 없음"}
    )
    log = _FakeLogRepository()

    with pytest.raises(AiException):
        await ContractService(fake, log).draft(_request())

    assert log.records[0].status is LogStatus.FAILED
    assert "담당 업무" in log.records[0].error_message


async def test_works_without_log_repository():
    """로그 저장소가 없어도 본 기능은 그대로 돈다."""
    fake = _FakeGemini(
        {
            "main_task_summary": "모바일 앱용 RESTful API 설계 및 개발",
            "detail_scope_summary": "",
            "special_terms": "별도의 특약사항 없음",
        }
    )

    result = await ContractService(fake).draft(_request())

    assert result.main_task_summary == "모바일 앱용 RESTful API 설계 및 개발"
