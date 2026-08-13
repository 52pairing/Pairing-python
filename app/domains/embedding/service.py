"""임베딩 생성/검색 유스케이스."""

import hashlib
import logging

from app.clients.gemini import GeminiClient, GeminiTask
from app.core.config import get_settings
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import (
    AgentType,
    AiAgentLogRepository,
    AiCallRecord,
    LogStatus,
    RefType,
)
from app.domains.embedding.repository import CandidateConditionRow, EmbeddingRepository
from app.domains.embedding.schemas import (
    EmbeddingResponse,
    SimilarFreelancer,
    SimilaritySearchResponse,
)

logger = logging.getLogger(__name__)

# 임베딩 원문은 이력서 전체라 길다. ai_agent_log 에는 앞부분만 남긴다(원인 파악에는 충분하고,
# 그 테이블이 이력서 사본 저장소가 되면 안 된다). 관리자 원본 로그 화면이 이 값을 쓴다.
#
# **stdout 로그에는 원문을 넣지 않는다.** 로그는 DB 와 접근권한·보존기간이 다르고, 탈퇴 회원의
# 개인정보 삭제 요청에 대응할 수 없다(DB 는 지울 수 있지만 로그는 못 지운다). 길이만 남겨도
# "텍스트가 비었나/짧나"는 판단되므로 text_chars 로 충분하다.
_LOGGED_TEXT_LIMIT = 500


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 벡터는 앞 몇 개만 남긴다. 768 개를 통째로 찍으면 로그 한 줄이 약 9KB 가 되고, 재색인
# 1600 건이면 그것만 30MB 다 — 정작 필요한 예외 스택이 묻혀서 못 찾는다(2026-08-13 실제로 겪음).
# 값이 정상 범위인지 보는 데는 앞 몇 개로 충분하고, 생성 여부는 dimension 으로 판단한다.
# ai_agent_log 도 같은 이유로 차원만 저장한다(_embed_and_log 참고) — 두 곳 기준을 맞춘다.
def _vector_preview(vector: list[float], limit: int = 12) -> list[float]:
    return [round(float(value), 6) for value in vector[:limit]]


class EmbeddingService:
    def __init__(
        self,
        repository: EmbeddingRepository,
        gemini: GeminiClient,
        ai_log_repository: AiAgentLogRepository | None = None,
    ):
        self._repository = repository
        self._gemini = gemini
        self._settings = get_settings()
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def _embed_and_log(self, text: str, ref_type: RefType, ref_id: int) -> list[float]:
        """임베딩 호출 1건 = ai_agent_log 1행. 실패해도 기록하고 예외는 그대로 올린다."""
        try:
            vectors, usage = await self._gemini.embed_with_usage([text])
        except AiException as exc:
            await self._record(
                AiCallRecord(
                    agent_type=AgentType.EMBEDDING,
                    status=LogStatus.FAILED,
                    ref_type=ref_type,
                    ref_id=ref_id,
                    model=self._gemini.model_for(GeminiTask.EMBEDDING),
                    request_json={"text": text[:_LOGGED_TEXT_LIMIT]},
                    error_message=str(exc),
                )
            )
            raise

        await self._record(
            AiCallRecord(
                agent_type=AgentType.EMBEDDING,
                status=LogStatus.SUCCESS,
                ref_type=ref_type,
                ref_id=ref_id,
                model=usage.model,
                request_json={"text": text[:_LOGGED_TEXT_LIMIT]},
                # 벡터 768개를 그대로 넣으면 로그가 사람이 못 읽는 크기가 된다. 차원만 남긴다.
                response_json={"dimension": len(vectors[0])},
                prompt_tokens=usage.prompt_tokens,
                output_tokens=usage.output_tokens,
                latency_ms=usage.latency_ms,
                retry_count=usage.retry_count,
            )
        )
        logger.info(
            "MATCHING_DEBUG python.embedding.generated ref_type=%s ref_id=%s model=%s "
            "dimension=%s vector_preview=%s text_chars=%s",
            ref_type.value,
            ref_id,
            usage.model,
            len(vectors[0]),
            _vector_preview(vectors[0]),
            len(text),
        )
        return vectors[0]

    async def _record(self, call: AiCallRecord) -> None:
        if self._ai_log_repository is not None:
            await self._ai_log_repository.record(call)

    async def upsert_freelancer(self, freelancer_id: int, text: str) -> EmbeddingResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)
        source_hash = _hash(f"{model}:{text}")

        # 이력서를 저장할 때마다 호출되므로, 내용이 그대로면 API 를 부르지 않는다.
        # (Gemini 임베딩 호출은 돈과 시간이 든다)
        if await self._repository.find_freelancer_hash(freelancer_id) == source_hash:
            logger.info(
                "MATCHING_DEBUG python.embedding.skip target=freelancer freelancer_id=%s "
                "model=%s source_hash=%s text_chars=%s",
                freelancer_id,
                model,
                source_hash[:16],
                len(text),
            )
            return EmbeddingResponse(
                target_id=freelancer_id,
                model=model,
                dimension=self._settings.embedding_dimension,
                skipped=True,
            )

        vector = await self._embed_and_log(text, RefType.FREELANCER, freelancer_id)
        await self._repository.upsert_freelancer(freelancer_id, vector, model, source_hash)
        logger.info(
            "MATCHING_DEBUG python.embedding.upserted target=freelancer freelancer_id=%s "
            "model=%s dimension=%s source_hash=%s vector_preview=%s",
            freelancer_id,
            model,
            len(vector),
            source_hash[:16],
            _vector_preview(vector),
        )

        return EmbeddingResponse(
            target_id=freelancer_id, model=model, dimension=len(vector), skipped=False
        )

    async def upsert_position(self, position_id: int, text: str) -> EmbeddingResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)
        source_hash = _hash(f"{model}:{text}")
        vector = await self._embed_and_log(text, RefType.POSITION, position_id)
        await self._repository.upsert_position(position_id, vector, model, source_hash)
        logger.info(
            "MATCHING_DEBUG python.embedding.upserted target=position position_id=%s "
            "model=%s dimension=%s source_hash=%s vector_preview=%s text_chars=%s",
            position_id,
            model,
            len(vector),
            source_hash[:16],
            _vector_preview(vector),
            len(text),
        )

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

        logger.info(
            "MATCHING_DEBUG python.embedding.search position_id=%s dimension=%s vector_preview=%s "
            "limit=%s job_category=%s job_role=%s excluded_count=%s excluded_ids=%s",
            position_id,
            len(vector),
            _vector_preview(list(vector)),
            limit,
            job_category,
            job_role,
            len(excluded_freelancer_ids or []),
            excluded_freelancer_ids or [],
        )
        rows = await self._repository.search_similar_freelancers(
            list(vector), limit, job_category, job_role, excluded_freelancer_ids
        )
        logger.info(
            "MATCHING_DEBUG python.embedding.search.result position_id=%s row_count=%s rows=%s",
            position_id,
            len(rows),
            [
                {"freelancer_id": freelancer_id, "similarity": round(score, 6)}
                for freelancer_id, score in rows
            ],
        )
        return SimilaritySearchResponse(
            position_id=position_id,
            candidates=[SimilarFreelancer(freelancer_id=fid, score=score) for fid, score in rows],
        )

    async def search_scored_candidates(
        self,
        position_id: int,
        job_category: str,
        job_role: str,
        required_skills: list[str],
        excluded_freelancer_ids: list[int] | None = None,
    ) -> list[CandidateConditionRow]:
        """하드필터 통과자 전원 + 채점용 조건 값. 자르지 않는다.

        자르는 건 임베딩 25 + 조건점수 75 를 합산한 뒤 매칭 도메인이 한다.
        """
        vector = await self._repository.find_position_vector(position_id)
        if vector is None:
            # 포지션 임베딩을 아직 안 만든 상태. 스프링이 모집 시작 시 호출해야 한다.
            raise AiException(AiErrorCode.EMBEDDING_NOT_FOUND)

        logger.info(
            "MATCHING_DEBUG python.embedding.scored_search position_id=%s dimension=%s "
            "vector_preview=%s job_category=%s job_role=%s required_skills=%s excluded_count=%s "
            "excluded_ids=%s",
            position_id,
            len(vector),
            _vector_preview(list(vector)),
            job_category,
            job_role,
            required_skills,
            len(excluded_freelancer_ids or []),
            excluded_freelancer_ids or [],
        )
        rows = await self._repository.search_scored_candidates(
            list(vector), job_category, job_role, required_skills, excluded_freelancer_ids
        )
        logger.info(
            "MATCHING_DEBUG python.embedding.scored_search.result position_id=%s row_count=%s",
            position_id,
            len(rows),
        )
        return rows
