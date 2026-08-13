"""챗봇 지식 청크 저장/검색. SQL 은 이 계층 밖으로 새지 않는다."""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.chatbot.models import ChatbotKnowledge

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NearestChunk:
    """질문과 가장 가까운 청크 하나.

    similarity 는 0~1 이고 클수록 가깝다. pgvector 의 `<=>` 는 코사인 <b>거리</b>(작을수록
    가까움)를 주므로 `1 - 거리` 로 뒤집어서 돌려준다. 임계값을 "이 값 이상이면 통과"로
    읽는 편이 헷갈리지 않는다.
    """

    chunk_key: str
    similarity: float


class ChatbotKnowledgeRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_hash(self, chunk_key: str) -> str | None:
        stmt = select(ChatbotKnowledge.source_hash).where(ChatbotKnowledge.chunk_key == chunk_key)
        return await self._session.scalar(stmt)

    async def upsert(
        self, chunk_key: str, content: str, vector: list[float], model: str, source_hash: str
    ) -> None:
        """같은 키를 다시 넣어도 행이 늘지 않는다. (unique 제약 기준)"""
        stmt = insert(ChatbotKnowledge).values(
            chunk_key=chunk_key,
            content=content,
            embedding=vector,
            model=model,
            source_hash=source_hash,
        )
        await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=[ChatbotKnowledge.chunk_key],
                set_={
                    "content": content,
                    "embedding": vector,
                    "model": model,
                    "source_hash": source_hash,
                },
            )
        )

    async def find_nearest(self, vector: list[float]) -> NearestChunk | None:
        """질문 벡터와 가장 가까운 청크. 지식이 하나도 없으면 None.

        <p>None 을 "관련 없음"으로 해석하면 안 된다. 시딩 전이거나 DB 가 비었을 때도 None 이라,
        그대로 차단하면 <b>모든 질문이 막힌다.</b> 호출부가 이 경우를 통과로 처리한다.
        """
        distance = ChatbotKnowledge.embedding.cosine_distance(vector)
        stmt = select(ChatbotKnowledge.chunk_key, distance).order_by(distance).limit(1)

        row = (await self._session.execute(stmt)).first()
        if row is None:
            return None
        return NearestChunk(chunk_key=row[0], similarity=1.0 - float(row[1]))
