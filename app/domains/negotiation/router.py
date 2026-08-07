"""협상 제안 API. 스프링의 협상 도메인이 라운드마다 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.domains.negotiation.schemas import ProposeRequest, ProposeResponse
from app.domains.negotiation.service import NegotiationService

router = APIRouter(prefix="/negotiations", tags=["03. Negotiation"])


def get_service(
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
) -> NegotiationService:
    return NegotiationService(gemini)


@router.post("/propose")
async def propose(
    request: ProposeRequest,
    service: Annotated[NegotiationService, Depends(get_service)],
) -> ApiResponse[ProposeResponse]:
    result = await service.propose(request)
    return ApiResponse.success("NEGOTIATION_PROPOSED", "제안을 생성했습니다.", result)
