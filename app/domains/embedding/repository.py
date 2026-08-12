"""벡터 저장/검색. SQL 은 이 계층 밖으로 새지 않는다."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, String, column, select, table, text
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
    column("matching_paused", Boolean),
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


@dataclass
class CandidateConditionRow:
    """하드필터를 통과한 후보 1명 + 채점에 필요한 조건 값.

    이력서 원문(자기소개·경력사항)은 **일부러 안 읽는다.** 하드필터 통과자 전원을 가져오므로
    수천 행이 될 수 있는데, 그 단계에서 필요한 건 채점용 숫자·코드값뿐이다. 원문은 상위
    (모집인원 x 3)명이 확정된 뒤 `DirectoryRepository.find_freelancer_profiles` 가 읽는다.
    """

    freelancer_id: int
    similarity: float
    matched_skill_levels: list[str]
    career_years: int
    pay_unit: str | None
    pay_amount: Decimal | None
    work_style: str | None
    work_form: str | None
    available_from: date | None
    start_negotiable: bool
    period_value: int | None
    period_unit: str | None


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

        하드필터(Stage B): AI매칭 동의 + 매칭 일시중지 아님 + 계정 활성 + 직군/직무 일치만 본다.
        일정/근무조건/단가는 여기서 거르지 않는다 — 사전검수(P02)가 안내하는 후보 수와 어긋나면
        착수금 클레임 리스크가 생겨서, Stage E(LLM 최종선정) 감점으로 넘긴다(`.ai/STATE.md`
        "Stage B 조건필터 폐기" 참고). job_category/job_role이 없으면(내부용 후보 미리보기
        엔드포인트) 필터 없이 순수 벡터 검색만 한다.
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
                    _freelancer_profile.c.matching_paused.is_(False),
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

    async def search_scored_candidates(
        self,
        vector: list[float],
        job_category: str,
        job_role: str,
        required_skills: list[str],
        excluded_freelancer_ids: list[int] | None = None,
    ) -> list[CandidateConditionRow]:
        """하드필터를 통과한 후보 **전원**과 채점용 조건 값을 가져온다.

        `search_similar_freelancers` 와 달리 **유사도로 자르지 않는다(LIMIT 없음).** 유사도로
        먼저 자르면 조건이 좋은데 자기소개가 짧아 유사도가 낮은 사람이 잘려나가, 재설계 전
        문제(스킬·연차가 1차 추림에 반영 안 됨)로 그대로 돌아간다. 자르는 건 임베딩 30 +
        조건점수 70 을 **합산한 뒤**에 파이썬이 한다(`matching/scoring.py`).

        하드필터(여기서만 후보를 배제한다):
        AI매칭 동의 + 매칭 일시중지 아님 + 계정 ACTIVE + 직군/직무 일치
        + **요구 스킬 1개 이상 보유** + 이미 노출된 후보 제외.

        "전부 보유"가 아니라 "1개 이상"인 이유는 5개 중 4개 가진 좋은 사람을 놓치지 않기
        위해서다. 노출은 모집 인원만큼만 하므로 5/5 가 충분하면 3/5 는 화면에 안 뜬다.

        수백~수천 명 규모에서는 전수 조회에 문제가 없다. **수만 명이 되면 재검토할 것.**
        """
        # pgvector 연산자를 그대로 쓴다. `<=>` 가 코사인 거리(0~2)다.
        # matched_skill_levels 는 요구 스킬과 겹치는 것만 모은다 — 요구 밖의 스킬은 채점에
        # 안 들어가므로 여기서 걸러 보내야 파이썬이 다시 거를 필요가 없다.
        sql = text(
            """
            SELECT fp.id                         AS freelancer_id,
                   1 - (fe.embedding <=> CAST(:vector AS vector)) AS similarity,
                   COALESCE(matched.levels, ARRAY[]::varchar[])   AS matched_skill_levels,
                   fc.career_years, fc.pay_unit, fc.pay_amount,
                   fc.work_style, fc.work_form,
                   fc.available_from, fc.start_negotiable,
                   fc.period_value, fc.period_unit
            FROM freelancer_embedding fe
            JOIN freelancer_profile fp ON fp.id = fe.freelancer_id
            JOIN account a ON a.id = fp.account_id
            JOIN freelancer_condition fc ON fc.account_id = fp.account_id
            JOIN LATERAL (
                SELECT array_agg(cs.skill_level) AS levels
                FROM condition_skill cs
                WHERE cs.condition_id = fc.id
                  AND cs.skill_code = ANY(:required_skills)
            ) matched ON TRUE
            WHERE fp.ai_matching_agreed IS TRUE
              AND fp.matching_paused IS FALSE
              AND a.status = 'ACTIVE'
              AND fc.job_category = :job_category
              AND fc.job_role = :job_role
              AND (:skip_skill_filter OR matched.levels IS NOT NULL)
              AND NOT (fp.id = ANY(:excluded_ids))
            """
        )
        rows = (
            await self._session.execute(
                sql,
                {
                    "vector": str(vector),
                    "job_category": job_category,
                    "job_role": job_role,
                    "required_skills": required_skills or [""],
                    # 요구 스킬이 아예 없는 포지션은 이 필터로 전원을 떨어뜨리면 안 된다.
                    "skip_skill_filter": not required_skills,
                    "excluded_ids": excluded_freelancer_ids or [],
                },
            )
        ).mappings().all()

        return [
            CandidateConditionRow(
                freelancer_id=row["freelancer_id"],
                similarity=float(row["similarity"]),
                matched_skill_levels=list(row["matched_skill_levels"] or []),
                career_years=row["career_years"],
                pay_unit=row["pay_unit"],
                pay_amount=row["pay_amount"],
                work_style=row["work_style"],
                work_form=row["work_form"],
                available_from=row["available_from"],
                start_negotiable=row["start_negotiable"],
                period_value=row["period_value"],
                period_unit=row["period_unit"],
            )
            for row in rows
        ]
