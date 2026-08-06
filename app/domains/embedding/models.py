"""AI 서버가 소유하는 테이블 매핑.

스키마는 ai-server/db/init/*.sql 이 단일 소스다. (백엔드 레포의 규칙과 동일)
여기서 테이블을 자동 생성하지 않는다. create_all 을 쓰면 SQL 파일과 조용히 어긋난다.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.session import Base

_DIMENSION = get_settings().embedding_dimension


class FreelancerEmbedding(Base):
    __tablename__ = "freelancer_embedding"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # account.id 가 아니라 freelancer_profile.id 를 가리킨다. (스프링 매칭 도메인과 같은 기준)
    freelancer_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_DIMENSION), nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )


class PositionEmbedding(Base):
    __tablename__ = "position_embedding"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    position_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_DIMENSION), nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
