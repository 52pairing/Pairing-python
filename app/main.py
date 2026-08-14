"""FastAPI 진입점.

실행: uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import internal_router, public_router
from app.clients.spring import get_spring_client
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.metrics import bind_db_pool_metrics, gemini_stub_mode
from app.core.middleware import TraceIdMiddleware
from app.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    # 커넥션 풀 게이지를 엔진에 연결한다. 엔진이 만들어진 뒤여야 해서 여기서 한다.
    bind_db_pool_metrics(engine)

    # 스텁 여부는 **기동 시점에** 확정한다.
    #
    # GeminiClient 는 첫 AI 요청 때 만들어진다(get_gemini_client 의 지연 생성). 그 안에서만
    # 게이지를 켜면, 서버가 떠 있는데도 첫 호출 전까지 gemini_stub_mode 가 0 으로 보인다.
    # 그러면 이 값을 보고 판단하는 쪽이 전부 틀린다 — 실제로 k6 가 "스텁이 아니다"라고
    # 판정해서 부하 테스트를 시작조차 못 했다(2026-08-14 확인). Grafana 도 마찬가지다.
    gemini_stub_mode.set(1 if settings.ai_stub_mode else 0)
    if settings.ai_stub_mode:
        # 이모지나 em-dash 를 쓰지 않는다. Windows 콘솔(cp949)에서 깨져서 오히려 안 보인다.
        logger.warning(
            "[AI_STUB_MODE] Gemini 를 호출하지 않고 더미로 응답한다. "
            "지연 %dms(+-%dms), 실패율 %.0f%%. 부하 테스트 전용 설정이다.",
            settings.ai_stub_delay_ms,
            settings.ai_stub_jitter_ms,
            settings.ai_stub_fail_rate * 100,
        )
    logger.info(
        "AI 서버 기동: env=%s, embedding=%s, matching=%s",
        settings.app_env,
        settings.gemini_model_embedding,
        settings.gemini_model_matching,
    )

    yield

    await get_spring_client().aclose()
    await engine.dispose()
    logger.info("AI 서버 종료")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Pairing AI Server",
        description="페어링 백엔드(Spring Boot)의 내부 AI 서비스. 외부에 직접 노출하지 않는다.",
        version="0.1.0",
        lifespan=lifespan,
        # CORS 를 열지 않는다. 브라우저가 직접 부르지 않고 스프링만 호출한다.
    )

    app.add_middleware(TraceIdMiddleware)
    register_exception_handlers(app)

    app.include_router(public_router)
    app.include_router(internal_router)

    return app


app = create_app()
