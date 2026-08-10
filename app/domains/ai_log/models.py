"""AI 호출 기록 테이블 매핑.

`ai_agent_log`는 AI 서버가 쓰기 주인이고 스프링은 읽기만 한다(`app/db/session.py` 소유권 규칙).
스키마 원본은 **백엔드 레포**의 `db/init/02-create-schema.sql`에 있다 — 관리자 화면이 이 테이블을
직접 조회하기 때문에 스프링 스키마 파일에 함께 들어가 있다. 여기서 테이블을 만들지 않는다.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AiAgentLog(Base):
    __tablename__ = "ai_agent_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    agent_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # 스프링 관리자 화면이 (ref_type, ref_id)로 찾는다. 협상 원본 로그 조회는
    # `WHERE ref_type = 'NEGOTIATION' AND ref_id = ?` 라서 이 두 값이 틀리면 화면에 안 뜬다.
    ref_type: Mapped[str | None] = mapped_column(String(30))
    ref_id: Mapped[int | None] = mapped_column(BigInteger)
    model: Mapped[str | None] = mapped_column(String(50))
    request_json: Mapped[dict | None] = mapped_column(JSONB)
    response_json: Mapped[dict | None] = mapped_column(JSONB)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    # 무료 등급이라 실제 청구액이 0이고, Gemini 응답도 금액은 주지 않는다(토큰 수만 준다).
    # 모델별 단가표를 코드에 박으면 구글이 단가를 바꿀 때 조용히 틀린 값이 쌓이므로 비워둔다.
    # 토큰 수가 남아 있어서 나중에 단가만 정해지면 언제든 역산할 수 있다.
    cost_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
