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
from app.core.middleware import TraceIdMiddleware
from app.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
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
