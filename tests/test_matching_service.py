"""MatchingService 프롬프트 조립/후보 필터링 검증. DB/Gemini는 모두 fake 로 대체한다."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.errors import AiErrorCode, AiException
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
)


def _profile(freelancer_id: int) -> FreelancerProfile:
    return FreelancerProfile(
        freelancer_id=freelancer_id,
        job_category="DEVELOPMENT",
        job_role="BACKEND",
        career_years=6,
        has_freelance_exp=True,
        self_introduction="백엔드 6년차입니다.",
        career_summary="A사 백엔드팀: 주문 시스템 개발",
        skills=["SPRING_BOOT"],
    )


def _make_service(profiles: dict[int, FreelancerProfile], llm_response: str) -> MatchingService:
    embedding_service = AsyncMock()
    embedding_service.search_candidates.return_value = SimilaritySearchResponse(
        position_id=1,
        candidates=[SimilarFreelancer(freelancer_id=fid, score=0.9) for fid in profiles],
    )

    # model_for()는 동기 메서드다 — AsyncMock 전체로 만들면 코루틴을 돌려줘서 조립 결과가 깨진다.
    gemini = MagicMock()
    gemini.model_for.return_value = "gemini-2.0-flash"
    gemini.generate_json = AsyncMock(return_value=llm_response)

    directory_repository = AsyncMock()
    directory_repository.find_position_requirement.return_value = _POSITION
    directory_repository.find_freelancer_profiles.return_value = profiles

    return MatchingService(embedding_service, gemini, directory_repository)


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
async def test_recommend_raises_not_found_when_position_missing():
    service = _make_service({101: _profile(101)}, llm_response="{}")
    service._directory_repository.find_position_requirement.return_value = None

    with pytest.raises(AiException) as exc_info:
        await service.recommend(position_id=1, recruit_count=1, pool_multiplier=3)

    assert exc_info.value.error_code == AiErrorCode.NOT_FOUND
