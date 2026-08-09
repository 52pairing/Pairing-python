"""매칭 추천.

두 단계로 나눈다.
  1) 벡터 검색으로 후보 풀을 좁힌다 (모집 인원 x 3)
  2) LLM 이 그 풀만 보고 순위와 사유를 만든다

전체 프리랜서를 LLM 에 넣지 않는 이유는 비용·지연·컨텍스트 한계 셋 다다.
"""

import json
import logging

from app.clients.gemini import GeminiClient, GeminiTask
from app.core.errors import AiErrorCode, AiException
from app.domains.embedding.schemas import SimilaritySearchResponse
from app.domains.embedding.service import EmbeddingService
from app.domains.matching.repository import (
    DirectoryRepository,
    FreelancerProfile,
    PositionRequirement,
)
from app.domains.matching.schemas import MatchingResponse, RankedCandidate

logger = logging.getLogger(__name__)

_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "freelancer_id": {"type": "integer"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["freelancer_id", "score", "reason"],
            },
        }
    },
    "required": ["candidates"],
}


class MatchingService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        gemini: GeminiClient,
        directory_repository: DirectoryRepository,
    ):
        self._embedding_service = embedding_service
        self._gemini = gemini
        self._directory_repository = directory_repository

    async def recommend(
        self, position_id: int, recruit_count: int, pool_multiplier: int
    ) -> MatchingResponse:
        pool_size = recruit_count * pool_multiplier
        pool = await self._embedding_service.search_candidates(position_id, pool_size)

        if not pool.candidates:
            raise AiException(AiErrorCode.CANDIDATE_POOL_EMPTY)

        position = await self._directory_repository.find_position_requirement(position_id)
        if position is None:
            raise AiException(AiErrorCode.NOT_FOUND, "포지션을 찾을 수 없습니다.")

        freelancer_ids = [candidate.freelancer_id for candidate in pool.candidates]
        profiles = await self._directory_repository.find_freelancer_profiles(freelancer_ids)

        model = self._gemini.model_for(GeminiTask.MATCHING)
        raw = await self._gemini.generate_json(
            GeminiTask.MATCHING,
            self._build_prompt(position, pool, profiles, recruit_count),
            _RANKING_SCHEMA,
        )

        try:
            parsed = json.loads(raw)
            candidates = [RankedCandidate(**item) for item in parsed["candidates"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 후보 풀에 없는 ID 를 지어낼 수 있다. 풀 밖의 값은 버린다.
        allowed = {candidate.freelancer_id for candidate in pool.candidates}
        filtered = [candidate for candidate in candidates if candidate.freelancer_id in allowed]

        return MatchingResponse(position_id=position_id, model=model, candidates=filtered)

    def _build_prompt(
        self,
        position: PositionRequirement,
        pool: SimilaritySearchResponse,
        profiles: dict[int, FreelancerProfile],
        recruit_count: int,
    ) -> str:
        # 벡터 유사도는 풀을 좁히는 데만 쓴다. 프로필이 없는 후보(레이스 컨디션 등)는 판단 근거가
        # 없으니 통째로 뺀다 — 어차피 최종 필터(allowed)에서도 걸러지지 않는다.
        candidate_blocks = [
            self._describe_candidate(profiles[candidate.freelancer_id])
            for candidate in pool.candidates
            if candidate.freelancer_id in profiles
        ]

        return (
            "너는 프리랜서 매칭 심사자다. 아래 포지션 요구조건에 맞춰 후보를 적합한 순서로 정렬하고 "
            f"상위 {recruit_count}명을 골라라.\n"
            'reason은 짧은 근거 여러 개를 "|"로 이어붙인 하나의 문자열로 써라 '
            '(예: "백엔드 경력 5년 이상 충족|Spring 스킬 보유|자기소개에 유사 프로젝트 경험 언급"). '
            "문장으로 쓰지 말고 근거 단위로 끊어라.\n\n"
            f"{self._describe_position(position)}\n\n"
            "후보 목록:\n" + "\n\n".join(candidate_blocks)
        )

    @staticmethod
    def _describe_position(position: PositionRequirement) -> str:
        lines = [
            f"프로젝트: {position.project_title}",
            f"직군/직무: {position.job_category}/{position.job_role}",
            f"필요 경력: {position.min_career_years}년 이상",
            f"필요 스킬: {', '.join(position.skills) if position.skills else '명시 없음'}",
            f"근무 형태: {position.work_style}/{position.work_form}",
        ]
        for label, value in (
            ("현재 상황", position.current_situation),
            ("담당 업무", position.main_task),
            ("업무 범위", position.detail_scope),
            ("우대사항", position.extra_note),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "포지션 요구조건:\n" + "\n".join(lines)

    @staticmethod
    def _describe_candidate(profile: FreelancerProfile) -> str:
        lines = [
            f"- freelancer_id={profile.freelancer_id}",
            f"  직군/직무: {profile.job_category}/{profile.job_role}",
            f"  경력: {profile.career_years}년 이상"
            + ("(프리랜서 경험 있음)" if profile.has_freelance_exp else ""),
            f"  스킬: {', '.join(profile.skills) if profile.skills else '명시 없음'}",
        ]
        if profile.self_introduction:
            lines.append(f"  자기소개: {profile.self_introduction}")
        if profile.career_summary:
            lines.append(f"  경력사항: {profile.career_summary}")
        return "\n".join(lines)
