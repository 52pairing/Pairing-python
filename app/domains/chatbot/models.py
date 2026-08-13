"""챗봇이 소유하는 테이블 매핑.

스키마는 db/init/10-create-ai-schema.sql 이 단일 소스다. 여기서 테이블을 만들지 않는다.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.session import Base

_DIMENSION = get_settings().embedding_dimension


class ChatbotKnowledge(Base):
    """관련성 판정에 쓰는 정책 청크.

    답변 생성에는 쓰지 않는다. 정책 전문이 프롬프트에 통째로 들어가 있어서, 검색으로 일부만
    골라 넣으면 오히려 맥락이 잘린다. 여기는 <b>"이 질문이 우리 서비스 얘기인가"</b>만 판단한다.
    """

    __tablename__ = "chatbot_knowledge"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    chunk_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_DIMENSION), nullable=False)
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
