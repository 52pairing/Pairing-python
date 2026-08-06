"""매칭 API. 스프링의 매칭 도메인이 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.domains.embedding.router import get_service as get_embedding_service
from app.domains.embedding.service import EmbeddingService
from app.domains.matching.schemas import MatchingRequest, MatchingResponse
from app.domains.matching.service import MatchingService

router = APIRouter(prefix="/matchings", tags=["02. Matching"])


def get_service(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
) -> MatchingService:
    return MatchingService(embedding_service, gemini)


@router.post("/recommendations")
async def recommend(
    request: MatchingRequest,
    service: Annotated[MatchingService, Depends(get_service)],
) -> ApiResponse[MatchingResponse]:
    result = await service.recommend(
        request.position_id, request.recruit_count, request.pool_multiplier
    )
    return ApiResponse.success("RECOMMENDATION_COMPLETED", "추천을 완료했습니다.", result)
