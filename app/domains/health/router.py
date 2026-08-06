"""헬스체크. 배포/로드밸런서가 본다. 내부 키 없이 호출할 수 있는 유일한 경로다."""

from fastapi import APIRouter
from sqlalchemy import text

from app.core.response import ApiResponse
from app.db.session import SessionFactory

router = APIRouter(tags=["00. Health"])


@router.get("/health")
async def health() -> ApiResponse[dict]:
    """프로세스가 살아 있는지만 본다. DB 를 보지 않아 의존 서비스 장애로 재시작되지 않는다."""
    return ApiResponse.success("HEALTHY", "정상입니다.", {"status": "UP"})


@router.get("/health/ready")
async def readiness() -> ApiResponse[dict]:
    """트래픽을 받을 준비가 됐는지 본다. DB 연결까지 확인한다."""
    async with SessionFactory() as session:
        await session.execute(text("SELECT 1"))

    return ApiResponse.success("READY", "준비되었습니다.", {"db": "UP"})
