"""계약서 문구 생성 API. 스프링의 계약 도메인이 계약서를 만들 때 한 번 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.domains.contract.schemas import DraftTextRequest, DraftTextResponse
from app.domains.contract.service import ContractService

router = APIRouter(prefix="/contracts", tags=["04. Contract"])


def get_service(
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
) -> ContractService:
    return ContractService(gemini)


@router.post("/draft-texts")
async def draft_texts(
    request: DraftTextRequest,
    service: Annotated[ContractService, Depends(get_service)],
) -> ApiResponse[DraftTextResponse]:
    result = await service.draft(request)
    return ApiResponse.success("CONTRACT_DRAFTED", "계약서 문구를 생성했습니다.", result)