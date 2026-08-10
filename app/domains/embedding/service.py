"""임베딩 생성/검색 유스케이스."""

import hashlib
import logging

from app.clients.gemini import GeminiClient, GeminiTask
from app.core.config import get_settings
from app.core.errors import AiErrorCode, AiException
from app.domains.embedding.repository import EmbeddingRepository
from app.domains.embedding.schemas import (
    EmbeddingResponse,
    SimilarFreelancer,
    SimilaritySearchResponse,
)

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingService:
    def __init__(self, repository: EmbeddingRepository, gemini: GeminiClient):
        self._repository = repository
        self._gemini = gemini
        self._settings = get_settings()

    async def upsert_freelancer(self, freelancer_id: int, text: str) -> EmbeddingResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)
        source_hash = _hash(f"{model}:{text}")

        # 이력서를 저장할 때마다 호출되므로, 내용이 그대로면 API 를 부르지 않는다.
        # (Gemini 임베딩 호출은 돈과 시간이 든다)
        if await self._repository.find_freelancer_hash(freelancer_id) == source_hash:
            return EmbeddingResponse(
                target_id=freelancer_id,
                model=model,
                dimension=self._settings.embedding_dimension,
                skipped=True,
            )

        vector = (await self._gemini.embed([text]))[0]
        await self._repository.upsert_freelancer(freelancer_id, vector, model, source_hash)

        return EmbeddingResponse(
            target_id=freelancer_id, model=model, dimension=len(vector), skipped=False
        )

    async def upsert_position(self, position_id: int, text: str) -> EmbeddingResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)
        vector = (await self._gemini.embed([text]))[0]
        await self._repository.upsert_position(position_id, vector, model, _hash(f"{model}:{text}"))

        return EmbeddingResponse(
            target_id=position_id, model=model, dimension=len(vector), skipped=False
        )

    async def search_candidates(
        self,
        position_id: int,
        limit: int,
        job_category: str | None = None,
        job_role: str | None = None,
        excluded_freelancer_ids: list[int] | None = None,
    ) -> SimilaritySearchResponse:
        vector = await self._repository.find_position_vector(position_id)
        if vector is None:
            # 포지션 임베딩을 아직 안 만든 상태. 스프링이 프로젝트 등록 시 호출해야 한다.
            raise AiException(AiErrorCode.EMBEDDING_NOT_FOUND)

        rows = await self._repository.search_similar_freelancers(
            list(vector), limit, job_category, job_role, excluded_freelancer_ids
        )
        return SimilaritySearchResponse(
            position_id=position_id,
            candidates=[SimilarFreelancer(freelancer_id=fid, score=score) for fid, score in rows],
        )
