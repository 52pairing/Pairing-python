"""FAQ 챗봇 API. 스프링의 support 도메인이 사용자 질문마다 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.domains.chatbot.schemas import AskRequest, AskResponse
from app.domains.chatbot.service import ChatbotService

router = APIRouter(prefix="/chatbot", tags=["18. Support"])


def get_service(
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
) -> ChatbotService:
    return ChatbotService(gemini)


@router.post("/answer")
async def answer(
    request: AskRequest,
    service: Annotated[ChatbotService, Depends(get_service)],
) -> ApiResponse[AskResponse]:
    result = await service.ask(request)
    return ApiResponse.success("CHATBOT_ANSWERED", "응답했습니다.", result)
