"""v1 라우터 조립.

내부 키 검증은 여기서 한 번만 건다. 라우터마다 붙이면 새 도메인에서 빠뜨린다.
헬스체크와 메트릭만 예외로 키 없이 연다.
  - 헬스체크: 로드밸런서가 키를 들고 있을 수 없다.
  - /metrics: Prometheus 에 앱 키를 심고 싶지 않다. 이 서버는 VPC 내부 전용이라
    네트워크로 이미 막혀 있다(app/core/security.py 의 전제와 동일).
"""

from fastapi import APIRouter, Depends

from app.core.metrics import router as metrics_router
from app.core.security import verify_internal_caller
from app.domains.chatbot.router import router as chatbot_router
from app.domains.contract.router import router as contract_router
from app.domains.embedding.router import router as embedding_router
from app.domains.health.router import router as health_router
from app.domains.matching.router import router as matching_router
from app.domains.negotiation.router import router as negotiation_router

# 인증 없이 열리는 경로
public_router = APIRouter()
public_router.include_router(health_router)
public_router.include_router(metrics_router)

# 스프링만 호출할 수 있는 경로
internal_router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_internal_caller)])
internal_router.include_router(contract_router)
internal_router.include_router(embedding_router)
internal_router.include_router(matching_router)
internal_router.include_router(negotiation_router)
internal_router.include_router(chatbot_router)
