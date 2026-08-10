"""임베딩 API. 스프링이 이력서/프로젝트 저장 시점에 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.db.session import get_session
from app.domains.ai_log.repository import AiAgentLogRepository
from app.domains.embedding.repository import EmbeddingRepository
from app.domains.embedding.schemas import (
    EmbeddingResponse,
    FreelancerEmbeddingRequest,
    PositionEmbeddingRequest,
    SimilaritySearchResponse,
)
from app.domains.embedding.service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["01. Embedding"])


def get_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
) -> EmbeddingService:
    return EmbeddingService(EmbeddingRepository(session), gemini, AiAgentLogRepository())


@router.put("/freelancers")
async def upsert_freelancer_embedding(
    request: FreelancerEmbeddingRequest,
    service: Annotated[EmbeddingService, Depends(get_service)],
) -> ApiResponse[EmbeddingResponse]:
    result = await service.upsert_freelancer(request.freelancer_id, request.text)
    return ApiResponse.success("FREELANCER_EMBEDDED", "임베딩을 저장했습니다.", result)


@router.put("/positions")
async def upsert_position_embedding(
    request: PositionEmbeddingRequest,
    service: Annotated[EmbeddingService, Depends(get_service)],
) -> ApiResponse[EmbeddingResponse]:
    result = await service.upsert_position(request.position_id, request.text)
    return ApiResponse.success("POSITION_EMBEDDED", "임베딩을 저장했습니다.", result)


@router.get("/positions/{position_id}/candidates")
async def search_candidates(
    position_id: int,
    service: Annotated[EmbeddingService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
) -> ApiResponse[SimilaritySearchResponse]:
    result = await service.search_candidates(position_id, limit)
    return ApiResponse.success("CANDIDATES_FOUND", "후보를 조회했습니다.", result)
