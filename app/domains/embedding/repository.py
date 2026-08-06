"""벡터 저장/검색. SQL 은 이 계층 밖으로 새지 않는다."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.embedding.models import FreelancerEmbedding, PositionEmbedding


class EmbeddingRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_freelancer_hash(self, freelancer_id: int) -> str | None:
        stmt = select(FreelancerEmbedding.source_hash).where(
            FreelancerEmbedding.freelancer_id == freelancer_id
        )
        return await self._session.scalar(stmt)

    async def upsert_freelancer(
        self, freelancer_id: int, vector: list[float], model: str, source_hash: str
    ) -> None:
        # 같은 프리랜서를 두 번 보내도 행이 늘지 않게 upsert 한다. (unique 제약 기준)
        stmt = insert(FreelancerEmbedding).values(
            freelancer_id=freelancer_id, embedding=vector, model=model, source_hash=source_hash
        )
        await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=[FreelancerEmbedding.freelancer_id],
                set_={"embedding": vector, "model": model, "source_hash": source_hash},
            )
        )

    async def upsert_position(
        self, position_id: int, vector: list[float], model: str, source_hash: str
    ) -> None:
        stmt = insert(PositionEmbedding).values(
            position_id=position_id, embedding=vector, model=model, source_hash=source_hash
        )
        await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=[PositionEmbedding.position_id],
                set_={"embedding": vector, "model": model, "source_hash": source_hash},
            )
        )

    async def find_position_vector(self, position_id: int) -> list[float] | None:
        stmt = select(PositionEmbedding.embedding).where(PositionEmbedding.position_id == position_id)
        return await self._session.scalar(stmt)

    async def search_similar_freelancers(
        self, vector: list[float], limit: int
    ) -> list[tuple[int, float]]:
        """코사인 거리 기준 최근접 검색. 거리(0~2)를 유사도(1~-1)로 바꿔 돌려준다."""
        distance = FreelancerEmbedding.embedding.cosine_distance(vector)
        stmt = (
            select(FreelancerEmbedding.freelancer_id, distance.label("distance"))
            .order_by(distance)
            .limit(limit)
        )
        rows = await self._session.execute(stmt)
        return [(row.freelancer_id, 1.0 - float(row.distance)) for row in rows]
