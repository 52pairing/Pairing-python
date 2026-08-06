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
from app.domains.embedding.service import EmbeddingService
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
    def __init__(self, embedding_service: EmbeddingService, gemini: GeminiClient):
        self._embedding_service = embedding_service
        self._gemini = gemini

    async def recommend(
        self, position_id: int, recruit_count: int, pool_multiplier: int
    ) -> MatchingResponse:
        pool_size = recruit_count * pool_multiplier
        pool = await self._embedding_service.search_candidates(position_id, pool_size)

        if not pool.candidates:
            raise AiException(AiErrorCode.CANDIDATE_POOL_EMPTY)

        model = self._gemini.model_for(GeminiTask.MATCHING)
        raw = await self._gemini.generate_json(
            GeminiTask.MATCHING, self._build_prompt(position_id, pool, recruit_count), _RANKING_SCHEMA
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

    def _build_prompt(self, position_id: int, pool, recruit_count: int) -> str:
        # TODO: 포지션 요구조건과 프리랜서 요약을 붙인다. (스프링에서 받거나 DB 에서 읽는다)
        lines = [
            f"- freelancer_id={c.freelancer_id}, 유사도={c.score:.3f}" for c in pool.candidates
        ]
        return (
            "너는 프리랜서 매칭 심사자다. 아래 후보 중 포지션에 적합한 순서로 정렬하고 "
            f"상위 {recruit_count}명을 골라라. 사유는 한국어 두 문장 이내로 쓴다.\n"
            f"포지션 ID: {position_id}\n"
            "후보 목록:\n" + "\n".join(lines)
        )
