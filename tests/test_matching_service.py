"""MatchingService 프롬프트 조립/후보 필터링 검증. DB/Gemini는 모두 fake 로 대체한다."""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clients.gemini import GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import AgentType, LogStatus, RefType
from app.domains.embedding.schemas import SimilarFreelancer, SimilaritySearchResponse
from app.domains.matching.repository import FreelancerProfile, PositionRequirement
from app.domains.matching.service import MatchingService

_POSITION = PositionRequirement(
    project_title="사내 ERP 고도화",
    job_category="DEVELOPMENT",
    job_role="BACKEND",
    min_career_years=5,
    current_situation="레거시 모놀리식 운영중",
    main_task="주문 도메인 API 개발",
    detail_scope="주문/결제 API 설계 및 구현",
    extra_note="핀테크 도메인 경험 우대",
    work_style="REMOTE",
    work_form="FULL_TIME",
    skills=["SPRING_BOOT", "POSTGRESQL"],
    budget_amount=Decimal("48000000"),
    period_value=4,
    period_unit="MONTH",
    start_desired_date=date(2026, 9, 1),
    start_negotiable=False,
)


def _profile(freelancer_id: int, **overrides) -> FreelancerProfile:
    defaults = {
        "freelancer_id": freelancer_id,
        "job_category": "DEVELOPMENT",
        "job_role": "BACKEND",
        "career_years": 6,
        "has_freelance_exp": True,
        "self_introduction": "백엔드 6년차입니다.",
        "career_summary": "A사 백엔드팀: 주문 시스템 개발",
        "skills": ["SPRING_BOOT"],
        "pay_unit": "MONTHLY",
        "pay_amount": Decimal("6200000"),
        "work_style": "REMOTE",
        "work_form": "FULL_TIME",
        "available_from": date(2026, 9, 1),
        "start_negotiable": False,
        "period_value": 4,
        "period_unit": "MONTH",
    }
    return FreelancerProfile(**{**defaults, **overrides})


def _make_service(
    profiles: dict[int, FreelancerProfile],
    llm_response: str,
    ai_log_repository: AsyncMock | None = None,
) -> MatchingService:
    embedding_service = AsyncMock()
    embedding_service.search_candidates.return_value = SimilaritySearchResponse(
        position_id=1,
        candidates=[SimilarFreelancer(freelancer_id=fid, score=0.9) for fid in profiles],
    )

    # model_for()는 동기 메서드다 — AsyncMock 전체로 만들면 코루틴을 돌려줘서 조립 결과가 깨진다.
    gemini = MagicMock()
    gemini.model_for.return_value = "gemini-3.5-flash"
    gemini.generate_json_with_usage = AsyncMock(
        return_value=(
            llm_response,
            GeminiUsage(
                model="gemini-3.5-flash",
                prompt_tokens=195,
                output_tokens=49,
                latency_ms=1840,
                retry_count=0,
            ),
        )
    )

    directory_repository = AsyncMock()
    directory_repository.find_position_requirement.return_value = _POSITION
    directory_repository.find_freelancer_profiles.return_value = profiles

    return MatchingService(embedding_service, gemini, directory_repository, ai_log_repository)


@pytest.mark.asyncio
async def test_prompt_includes_position_requirement_and_candidate_resume():
    service = _make_service({101: _profile(101)}, llm_response="{}")

    prompt = service._build_prompt(
        _POSITION,
        SimilaritySearchResponse(
            position_id=1, candidates=[SimilarFreelancer(freelancer_id=101, score=0.9)]
        ),
        {101: _profile(101)},
        recruit_count=1,
    )

    assert "사내 ERP 고도화" in prompt
    assert "핀테크 도메인 경험 우대" in prompt
    assert "freelancer_id=101" in prompt
    assert "백엔드 6년차입니다." in prompt
    assert "A사 백엔드팀: 주문 시스템 개발" in prompt
    # 벡터 유사도 수치는 판단 근거로 넘기지 않는다.
    assert "0.9" not in prompt


@pytest.mark.asyncio
async def test_recommend_records_ai_agent_log_with_ref_and_tokens():
    """스프링 관리자 화면이 (ref_type, ref_id)로 로그를 찾으므로 이 두 값이 정확해야 한다."""
    ai_log = AsyncMock()
    llm_response = '{"candidates": [{"freelancer_id": 101, "score": 95, "reason": "경력 충족"}]}'
    service = _make_service({101: _profile(101)}, llm_response=llm_response, ai_log_repository=ai_log)

    await service.recommend(position_id=1, recruit_count=1, pool_multiplier=3)

    ai_log.record.assert_awaited_once()
    call = ai_log.record.await_args.args[0]
    assert call.agent_type == AgentType.MATCHER
    assert call.status == LogStatus.SUCCESS
    assert call.ref_type == RefType.POSITION
    assert call.ref_id == 1
    assert call.model == "gemini-3.5-flash"
    assert call.prompt_tokens == 195
    assert call.output_tokens == 49
    assert call.latency_ms == 1840
    # 나중에 "이 추천이 왜 이렇게 나왔나"를 되짚으려면 실제 보낸 프롬프트가 남아야 한다.
    assert "사내 ERP 고도화" in call.request_json["prompt"]
    assert call.response_json["raw"] == llm_response
    # 무료 등급이라 비용은 안 채운다(토큰 수로 나중에 역산).
    assert call.error_message is None


@pytest.mark.asyncio
async def test_recommend_records_failed_log_when_llm_call_fails():
    ai_log = AsyncMock()
    service = _make_service({101: _profile(101)}, llm_response="{}", ai_log_repository=ai_log)
    service._gemini.generate_json_with_usage = AsyncMock(
        side_effect=AiException(AiErrorCode.LLM_CALL_FAILED)
    )

    with pytest.raises(AiException):
        await service.recommend(position_id=1, recruit_count=1, pool_multiplier=3)

    call = ai_log.record.await_args.args[0]
    assert call.status == LogStatus.FAILED
    assert call.ref_id == 1
    assert call.error_message is not None
    assert call.response_json is None


@pytest.mark.asyncio
async def test_recommend_keeps_pipe_joined_reason_and_drops_ids_outside_pool():
    llm_response = (
        '{"candidates": ['
        '{"freelancer_id": 101, "score": 0.95, "reason": "경력 충족|스킬 보유"},'
        '{"freelancer_id": 999, "score": 0.5, "reason": "풀 밖 후보"}'
        "]}"
    )
    service = _make_service({101: _profile(101)}, llm_response=llm_response)

    result = await service.recommend(position_id=1, recruit_count=1, pool_multiplier=3)

    assert [c.freelancer_id for c in result.candidates] == [101]
    assert result.candidates[0].reason == "경력 충족|스킬 보유"


@pytest.mark.asyncio
async def test_prompt_includes_condition_fields_for_stage_e_penalty():
    """일정/근무조건/단가는 하드필터로 배제하지 않고 LLM 감점 요인으로 넘긴다 — 그 원본 값이 실제로
    프롬프트에 들어가야 판단할 수 있다."""
    service = _make_service({101: _profile(101)}, llm_response="{}")

    prompt = service._build_prompt(
        _POSITION,
        SimilaritySearchResponse(
            position_id=1, candidates=[SimilarFreelancer(freelancer_id=101, score=0.9)]
        ),
        {101: _profile(101)},
        recruit_count=1,
    )

    # 포지션 쪽 비교 기준
    assert "48000000" in prompt
    assert "4개월" in prompt
    assert "2026-09-01" in prompt
    # 후보 쪽 비교 대상
    assert "월급 6200000원" in prompt
    assert "REMOTE/FULL_TIME" in prompt
    # 감점만 하고 제외하지 말라는 지시가 실제로 들어가야 한다
    assert "제외하지 말고" in prompt
    assert "감점" in prompt


@pytest.mark.asyncio
async def test_prompt_pins_score_range_to_0_100():
    """점수 범위를 안 알려주면 LLM이 10점 만점으로 매긴다(실제 Gemini 호출로 확인: score=9.5).

    스프링은 이 값을 matching_candidate.base_score(0~100)에 그대로 저장하고 50점을
    '적합도 낮음' 기준으로 쓰기 때문에, 범위를 안 박아두면 모든 후보가 항상 저품질로 찍힌다.
    """
    service = _make_service({101: _profile(101)}, llm_response="{}")

    prompt = service._build_prompt(
        _POSITION,
        SimilaritySearchResponse(
            position_id=1, candidates=[SimilarFreelancer(freelancer_id=101, score=0.9)]
        ),
        {101: _profile(101)},
        recruit_count=1,
    )

    assert "0~100" in prompt
    # 스프링의 lowScoreWarned 기준선(50점)도 함께 알려줘야 점수가 의미를 갖는다.
    assert "50점" in prompt


@pytest.mark.asyncio
async def test_prompt_marks_negotiable_start_date_so_llm_does_not_penalize_it():
    service = _make_service({101: _profile(101)}, llm_response="{}")

    prompt = service._build_prompt(
        _POSITION,
        SimilaritySearchResponse(
            position_id=1, candidates=[SimilarFreelancer(freelancer_id=101, score=0.9)]
        ),
        {101: _profile(101, available_from=date(2026, 12, 1), start_negotiable=True)},
        recruit_count=1,
    )

    assert "시작 가능일: 2026-12-01 (협의 가능)" in prompt
    assert "'협의 가능'으로 표시된 항목은 어긋나도 감점하지 마라" in prompt


@pytest.mark.asyncio
async def test_prompt_omits_condition_lines_when_values_are_missing():
    """조건을 아직 안 채운 프리랜서도 후보에 들어올 수 있다. None을 그대로 찍으면 LLM이 'None'을
    조건 값으로 오해하므로 줄 자체를 빼야 한다."""
    service = _make_service({101: _profile(101)}, llm_response="{}")

    prompt = service._build_prompt(
        _POSITION,
        SimilaritySearchResponse(
            position_id=1, candidates=[SimilarFreelancer(freelancer_id=101, score=0.9)]
        ),
        {101: _profile(101, pay_amount=None, available_from=None, period_value=None)},
        recruit_count=1,
    )

    # 지시문에도 "희망 급여" 같은 말이 들어가므로, 후보 블록의 실제 줄 형태로 확인한다.
    assert "None" not in prompt
    assert "  희망 급여:" not in prompt
    assert "  시작 가능일:" not in prompt
    assert "  희망 기간:" not in prompt


@pytest.mark.asyncio
async def test_recommend_raises_not_found_when_position_missing():
    service = _make_service({101: _profile(101)}, llm_response="{}")
    service._directory_repository.find_position_requirement.return_value = None

    with pytest.raises(AiException) as exc_info:
        await service.recommend(position_id=1, recruit_count=1, pool_multiplier=3)

    assert exc_info.value.error_code == AiErrorCode.NOT_FOUND
