"""챗봇 지식 청크 저장/검색. SQL 은 이 계층 밖으로 새지 않는다."""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.chatbot.models import ChatbotKnowledge

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkMatch:
    """질문에 가장 가까운 청크 하나.

    similarity 는 0~1 이고 클수록 가깝다. pgvector 의 `<=>` 는 코사인 <b>거리</b>(작을수록
    가까움)를 주므로 `1 - 거리` 로 뒤집어서 돌려준다. 임계값을 "이 값 이상이면 통과"로
    읽는 편이 헷갈리지 않는다.
    """

    chunk_key: str
    similarity: float


@dataclass(frozen=True)
class GateEvidence:
    """게이트가 판정에 쓰는 재료. 양성·음성 각각의 최고점을 함께 준다.

    최근접 하나만 보면 안 된다. 음성 청크는 주제어를 여러 개 나열하게 되어서, 짧고 일반적인
    정상 질문("수수료 얼마야?")이 구체적인 숫자로 채워진 양성 청크보다 음성 목록에 더
    붙는다. 실측에서 정상 질문 3개가 이렇게 막혔다. 둘을 나란히 놓고 비교해야 한다.
    """

    positive: ChunkMatch | None
    negative: ChunkMatch | None
    greeting: ChunkMatch | None

    @property
    def is_empty(self) -> bool:
        return self.positive is None and self.negative is None and self.greeting is None


class ChatbotKnowledgeRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_hash(self, chunk_key: str) -> str | None:
        stmt = select(ChatbotKnowledge.source_hash).where(ChatbotKnowledge.chunk_key == chunk_key)
        return await self._session.scalar(stmt)

    async def upsert(
        self,
        chunk_key: str,
        content: str,
        vector: list[float],
        model: str,
        source_hash: str,
        polarity: str,
    ) -> None:
        """같은 키를 다시 넣어도 행이 늘지 않는다. (unique 제약 기준)"""
        stmt = insert(ChatbotKnowledge).values(
            chunk_key=chunk_key,
            content=content,
            embedding=vector,
            model=model,
            source_hash=source_hash,
            polarity=polarity,
        )
        await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=[ChatbotKnowledge.chunk_key],
                set_={
                    "content": content,
                    "embedding": vector,
                    "model": model,
                    "source_hash": source_hash,
                    "polarity": polarity,
                },
            )
        )

    async def find_gate_evidence(self, vector: list[float]) -> GateEvidence:
        """극성별 최근접 청크를 한 번에 가져온다.

        <p>둘 다 None 인 경우를 "관련 없음"으로 해석하면 안 된다. 시딩 전이거나 DB 가 비었을
        때도 그렇게 나오므로, 그대로 차단하면 <b>모든 질문이 막힌다.</b> 호출부가 통과로 처리한다.
        """
        distance = ChatbotKnowledge.embedding.cosine_distance(vector)
        # DISTINCT ON 으로 극성마다 1행만 남긴다. 쿼리를 두 번 보내지 않는다.
        stmt = (
            select(ChatbotKnowledge.polarity, ChatbotKnowledge.chunk_key, distance)
            .distinct(ChatbotKnowledge.polarity)
            .order_by(ChatbotKnowledge.polarity, distance)
        )

        best: dict[str, ChunkMatch] = {}
        for polarity, chunk_key, dist in await self._session.execute(stmt):
            best[polarity] = ChunkMatch(chunk_key=chunk_key, similarity=1.0 - float(dist))

        return GateEvidence(
            positive=best.get("POSITIVE"),
            negative=best.get("NEGATIVE"),
            greeting=best.get("GREETING"),
        )
