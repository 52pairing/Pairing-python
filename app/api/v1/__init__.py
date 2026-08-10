"""v1 라우터 조립.

내부 키 검증은 여기서 한 번만 건다. 라우터마다 붙이면 새 도메인에서 빠뜨린다.
헬스체크만 예외로 키 없이 연다. (로드밸런서가 키를 들고 있을 수 없다)
"""

from fastapi import APIRouter, Depends

from app.core.security import verify_internal_caller
from app.domains.chatbot.router import router as chatbot_router
from app.domains.embedding.router import router as embedding_router
from app.domains.health.router import router as health_router
from app.domains.matching.router import router as matching_router
from app.domains.negotiation.router import router as negotiation_router
from app.domains.contract.router import router as contract_router

# 인증 없이 열리는 경로
public_router = APIRouter()
public_router.include_router(health_router)

# 스프링만 호출할 수 있는 경로
internal_router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_internal_caller)])
internal_router.include_router(contract_router)
internal_router.include_router(embedding_router)
internal_router.include_router(matching_router)
internal_router.include_router(negotiation_router)
internal_router.include_router(chatbot_router)
