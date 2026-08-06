"""DB 세션.

스프링과 **같은 PostgreSQL** 을 본다. 그래서 소유권 규칙이 중요하다.

  - AI 서버가 쓰기(INSERT/UPDATE)하는 테이블: freelancer_embedding, position_embedding, ai_agent_log
  - 그 외 스프링 소유 테이블: 읽기만. 상태를 바꿔야 하면 스프링 API 를 호출한다.

두 서버가 같은 행을 쓰기 시작하면 트랜잭션 경계가 사라져서 원인 못 찾는 버그가 생긴다.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """AI 서버가 소유하는 테이블만 여기에 매핑한다."""


_settings = get_settings()

engine = create_async_engine(
    _settings.ai_db_url,
    echo=_settings.db_echo,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_pre_ping=True,
)

SessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """라우터 의존성. 정상 종료면 커밋, 예외면 롤백한다."""
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
