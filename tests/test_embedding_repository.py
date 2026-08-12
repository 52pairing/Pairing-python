"""EmbeddingRepository의 하드필터 SQL 검증.

**두 메서드의 검증 방식이 다르다.**

- `search_similar_freelancers`는 SQLAlchemy 표현식으로 조립하므로, 컴파일된 SQL을 보면
  조건 분기(job_category 유무)가 실제로 반영됐는지 증명된다.
- `search_scored_candidates`는 `text()` 리터럴이라 **SQL 문자열을 검사해봐야 동어반복**이다
  (소스에 적힌 글자를 소스에서 찾는 셈이라 절대 실패하지 않고, 의미가 틀려도 통과한다).
  그래서 여기서는 **계산되는 것만** 본다 — 파라미터와 결과 매핑이다. SQL 자체의 의미는
  아래 `test_search_scored_candidates_against_real_db`(실제 pgvector 필요)가 검증한다.
"""

import os

import pytest

from app.domains.embedding.repository import EmbeddingRepository


class _CapturingSession:
    """execute()에 전달된 statement만 붙잡아 컴파일해서 검사한다. 실행은 하지 않는다."""

    def __init__(self):
        self.captured_stmt = None

    async def execute(self, stmt):
        self.captured_stmt = stmt

        class _EmptyResult:
            def __iter__(self):
                return iter([])

        return _EmptyResult()


def _compiled_sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


async def test_search_similar_freelancers_hard_filter_excludes_paused_and_not_agreed():
    session = _CapturingSession()
    repo = EmbeddingRepository(session)

    await repo.search_similar_freelancers(
        vector=[0.1] * 768, limit=5, job_category="DEVELOPMENT", job_role="BACKEND"
    )

    sql = _compiled_sql(session.captured_stmt)
    assert "ai_matching_agreed IS true" in sql
    assert "matching_paused IS false" in sql
    assert "status = 'ACTIVE'" in sql


async def test_search_similar_freelancers_without_job_filter_skips_hard_filter():
    """내부용 후보 미리보기(job_category/job_role 없음)는 순수 벡터 검색만 한다 — 하드필터 조인이 없다."""
    session = _CapturingSession()
    repo = EmbeddingRepository(session)

    await repo.search_similar_freelancers(vector=[0.1] * 768, limit=5)

    sql = _compiled_sql(session.captured_stmt)
    assert "matching_paused" not in sql
    assert "ai_matching_agreed" not in sql


class _CapturingTextSession:
    """text() 쿼리와 파라미터를 붙잡는다. 반환 행은 테스트가 지정한다."""

    def __init__(self, rows: list[dict] | None = None):
        self.captured_params: dict | None = None
        self._rows = rows or []

    async def execute(self, stmt, params=None):
        self.captured_params = params
        rows = self._rows

        class _Result:
            def mappings(self):
                class _Mappings:
                    def all(self):
                        return rows

                return _Mappings()

        return _Result()


def _row(**overrides) -> dict:
    defaults = {
        "freelancer_id": 1,
        "similarity": 0.87,
        "matched_skill_levels": ["ADVANCED"],
        "career_years": 6,
        "pay_unit": "MONTHLY",
        "pay_amount": 6_200_000,
        "work_style": "REMOTE",
        "work_form": "FULL_TIME",
        "available_from": None,
        "start_negotiable": False,
        "period_value": 4,
        "period_unit": "MONTH",
    }
    return {**defaults, **overrides}


async def test_scored_search_turns_skill_filter_on_when_position_requires_skills():
    """`skip_skill_filter`가 뒤집히면 스킬 하드필터가 통째로 무력화된다.

    그래도 결과는 나오고 에러도 안 나서(그냥 후보가 늘 뿐) 발견이 어렵다. 재설계에서 새로
    추가한 유일한 배제 조건이라 여기서 못 잡으면 아무 데서도 못 잡는다.
    """
    session = _CapturingTextSession()
    repo = EmbeddingRepository(session)

    await repo.search_scored_candidates(
        [0.1] * 768, "DEVELOPMENT", "BACKEND", ["SPRING_BOOT", "POSTGRESQL"], None
    )

    assert session.captured_params["skip_skill_filter"] is False
    assert session.captured_params["required_skills"] == ["SPRING_BOOT", "POSTGRESQL"]
    # excluded 가 None 이면 빈 배열이어야 한다. None 을 그대로 넘기면 asyncpg 가 터진다.
    assert session.captured_params["excluded_ids"] == []


async def test_scored_search_turns_skill_filter_off_when_position_requires_none():
    """요구 스킬이 없는 포지션에서 필터가 살아있으면 **전원이 탈락한다**(빈 배열과 겹칠 리 없다).

    완화 재검색도 같은 경로로 요구 스킬 []를 넘기므로, 여기가 틀리면 완화가 완화되지 않는다.
    """
    session = _CapturingTextSession()
    repo = EmbeddingRepository(session)

    await repo.search_scored_candidates([0.1] * 768, "DEVELOPMENT", "BACKEND", [], [7, 9])

    assert session.captured_params["skip_skill_filter"] is True
    assert session.captured_params["excluded_ids"] == [7, 9]


async def test_scored_search_maps_null_skill_levels_to_empty_list():
    """스킬이 하나도 안 겹치면 array_agg 가 NULL 을 준다. 그대로 두면 채점에서 터진다."""
    session = _CapturingTextSession([_row(matched_skill_levels=None)])
    repo = EmbeddingRepository(session)

    [candidate] = await repo.search_scored_candidates([0.1] * 768, "DEVELOPMENT", "BACKEND", [], None)

    assert candidate.matched_skill_levels == []
    assert candidate.similarity == pytest.approx(0.87)


@pytest.mark.skipif(
    not os.getenv("AI_TEST_DB_URL"),
    reason="pgvector 가 있는 DB 가 필요하다. AI_TEST_DB_URL 을 주면 돈다.",
)
async def test_search_scored_candidates_against_real_db():
    """SQL 의 **의미**를 검증하는 유일한 테스트. 위 단위 테스트로는 못 잡는 것들을 본다.

    - `str(vector)` 가 pgvector 의 `CAST(... AS vector)` 로 파싱되는지
    - 빈 리스트 파라미터(`excluded_ids=[]`)가 asyncpg 타입 추론을 통과하는지
    - LATERAL 조인이 요구 스킬 밖의 보유 스킬을 빼는지

    실행 방법(임시 컨테이너):
        docker run -d --rm --name pgv -p 55432:5432 -e POSTGRES_PASSWORD=x \
            -e POSTGRES_DB=t pgvector/pgvector:pg16
        # 아래 _create_fixture_schema 가 만드는 테이블은 스프링 소유(ddl-auto)라
        # 실제 스키마와 어긋날 수 있다. 컬럼 추가 시 같이 손봐야 한다.
        AI_TEST_DB_URL=postgresql+asyncpg://postgres:x@localhost:55432/t pytest -k real_db
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(os.environ["AI_TEST_DB_URL"])
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            await _create_fixture_schema(session)
            repo = EmbeddingRepository(session)
            vector = [0.1] * 768

            matched = await repo.search_scored_candidates(
                vector, "DEVELOPMENT", "BACKEND", ["SPRING_BOOT", "POSTGRESQL"], None
            )
            assert [c.freelancer_id for c in matched] == [1]
            # REACT 는 요구 스킬 밖이라 숙련도 목록에 들어오면 안 된다(들어오면 일치율이 부풀려진다).
            assert matched[0].matched_skill_levels == ["ADVANCED"]

            assert await repo.search_scored_candidates(
                vector, "DEVELOPMENT", "BACKEND", ["FIGMA"], None
            ) == []

            relaxed = await repo.search_scored_candidates(vector, "DEVELOPMENT", "BACKEND", [], [])
            assert [c.freelancer_id for c in relaxed] == [1]

            assert await repo.search_scored_candidates(
                vector, "DEVELOPMENT", "BACKEND", ["SPRING_BOOT"], [1]
            ) == []
    finally:
        await engine.dispose()


async def _create_fixture_schema(session) -> None:
    from sqlalchemy import text

    for statement in (
        "CREATE EXTENSION IF NOT EXISTS vector",
        "DROP TABLE IF EXISTS freelancer_embedding, freelancer_profile, account,"
        " freelancer_condition, condition_skill",
        "CREATE TABLE freelancer_embedding (freelancer_id bigint PRIMARY KEY, embedding vector(768))",
        "CREATE TABLE freelancer_profile (id bigint, account_id bigint,"
        " ai_matching_agreed boolean, matching_paused boolean)",
        "CREATE TABLE account (id bigint, status varchar)",
        "CREATE TABLE freelancer_condition (id bigint, account_id bigint, job_category varchar,"
        " job_role varchar, career_years int, pay_unit varchar, pay_amount numeric,"
        " work_style varchar, work_form varchar, available_from date, start_negotiable boolean,"
        " period_value int, period_unit varchar)",
        "CREATE TABLE condition_skill (condition_id bigint, skill_code varchar, skill_level varchar)",
        "INSERT INTO freelancer_embedding VALUES (1, (SELECT ('['||string_agg('0.1',',')||']')::vector"
        " FROM generate_series(1,768)))",
        "INSERT INTO freelancer_profile VALUES (1, 10, true, false)",
        "INSERT INTO account VALUES (10, 'ACTIVE')",
        "INSERT INTO freelancer_condition VALUES (100, 10, 'DEVELOPMENT', 'BACKEND', 6, 'MONTHLY',"
        " 6200000, 'REMOTE', 'FULL_TIME', '2026-09-01', false, 4, 'MONTH')",
        "INSERT INTO condition_skill VALUES (100, 'SPRING_BOOT', 'ADVANCED'),"
        " (100, 'REACT', 'BEGINNER')",
    ):
        await session.execute(text(statement))
    await session.commit()
