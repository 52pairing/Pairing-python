"""벡터 저장/검색. SQL 은 이 계층 밖으로 새지 않는다."""

from sqlalchemy import BigInteger, Boolean, String, column, select, table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.embedding.models import FreelancerEmbedding, PositionEmbedding

# 스프링 소유 테이블. AI 서버는 SELECT 만 한다(app/domains/matching/repository.py와 동일 규칙).
# 여기 결과를 pgvector 코사인 거리와 같은 쿼리에서 조인해야 해서(임베딩 컬럼은 ORM 타입이 필요) 가벼운
# table()/column() 프록시로 선언한다 — ORM 모델(Base)로 선언하면 이 도메인이 그 테이블을 소유하는
# 것처럼 보여서 안 된다.
_freelancer_profile = table(
    "freelancer_profile",
    column("id", BigInteger),
    column("account_id", BigInteger),
    column("ai_matching_agreed", Boolean),
)
_freelancer_condition = table(
    "freelancer_condition",
    column("account_id", BigInteger),
    column("job_category", String),
    column("job_role", String),
)
_account = table(
    "account",
    column("id", BigInteger),
    column("status", String),
)


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
        self,
        vector: list[float],
        limit: int,
        job_category: str | None = None,
        job_role: str | None = None,
        excluded_freelancer_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """코사인 거리 기준 최근접 검색. 거리(0~2)를 유사도(1~-1)로 바꿔 돌려준다.

        하드필터(Stage B): AI매칭 동의 + 계정 활성 + 직군/직무 일치만 본다. 일정/근무조건/단가는
        여기서 거르지 않는다 — 사전검수(P02)가 안내하는 후보 수와 어긋나면 착수금 클레임 리스크가
        생겨서, Stage E(LLM 최종선정) 감점으로 넘긴다(`.ai/STATE.md` "Stage B 조건필터 폐기" 참고).
        job_category/job_role이 없으면(내부용 후보 미리보기 엔드포인트) 필터 없이 순수 벡터 검색만 한다.
        """
        distance = FreelancerEmbedding.embedding.cosine_distance(vector)
        stmt = select(FreelancerEmbedding.freelancer_id, distance.label("distance"))

        if job_category is not None and job_role is not None:
            stmt = (
                stmt.join(_freelancer_profile, _freelancer_profile.c.id == FreelancerEmbedding.freelancer_id)
                .join(
                    _freelancer_condition,
                    _freelancer_condition.c.account_id == _freelancer_profile.c.account_id,
                )
                .join(_account, _account.c.id == _freelancer_profile.c.account_id)
                .where(
                    _freelancer_profile.c.ai_matching_agreed.is_(True),
                    _account.c.status == "ACTIVE",
                    _freelancer_condition.c.job_category == job_category,
                    _freelancer_condition.c.job_role == job_role,
                )
            )

        if excluded_freelancer_ids:
            stmt = stmt.where(FreelancerEmbedding.freelancer_id.notin_(excluded_freelancer_ids))

        stmt = stmt.order_by(distance).limit(limit)
        rows = await self._session.execute(stmt)
        return [(row.freelancer_id, 1.0 - float(row.distance)) for row in rows]
