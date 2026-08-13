"""FAQ 챗봇 API. 스프링의 support 도메인이 사용자 질문마다 호출한다."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gemini import GeminiClient, get_gemini_client
from app.core.response import ApiResponse
from app.db.session import get_session
from app.domains.ai_log.repository import AiAgentLogRepository
from app.domains.chatbot.repository import ChatbotKnowledgeRepository
from app.domains.chatbot.schemas import AskRequest, AskResponse, KnowledgeReindexResponse
from app.domains.chatbot.seeder import ChatbotKnowledgeSeeder
from app.domains.chatbot.service import ChatbotService

router = APIRouter(prefix="/chatbot", tags=["18. Support"])


def get_service(
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatbotService:
    # AiAgentLogRepository 는 자기 세션을 직접 열어 쓴다. Depends(get_session) 이 필요 없다.
    # 지식 저장소는 읽기만 하므로 요청 세션을 그대로 쓴다.
    return ChatbotService(gemini, AiAgentLogRepository(), ChatbotKnowledgeRepository(session))


def get_seeder(
    gemini: Annotated[GeminiClient, Depends(get_gemini_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatbotKnowledgeSeeder:
    return ChatbotKnowledgeSeeder(gemini, ChatbotKnowledgeRepository(session))


@router.post("/answer")
async def answer(
    request: AskRequest,
    service: Annotated[ChatbotService, Depends(get_service)],
) -> ApiResponse[AskResponse]:
    result = await service.ask(request)
    return ApiResponse.success("CHATBOT_ANSWERED", "응답했습니다.", result)


@router.post("/knowledge/reindex")
async def reindex_knowledge(
    seeder: Annotated[ChatbotKnowledgeSeeder, Depends(get_seeder)],
) -> ApiResponse[KnowledgeReindexResponse]:
    """관련성 판정용 지식 청크를 다시 임베딩한다.

    <p>배포 후 한 번 부르면 된다. <b>부르지 않으면 관련성 게이트가 동작하지 않는다</b> —
    지식이 비어 있으면 판정을 건너뛰고 전부 통과시킨다(정상 질문을 막지 않기 위한 선택).

    <p>여러 번 불러도 안전하다. 문구가 그대로면 임베딩 API 를 호출하지 않는다.
    정책 문구를 고쳤을 때도 이걸 다시 부른다.
    """
    result = await seeder.seed()
    return ApiResponse.success("CHATBOT_KNOWLEDGE_REINDEXED", "지식을 갱신했습니다.", result)
