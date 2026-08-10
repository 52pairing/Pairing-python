"""EmbeddingRepository의 하드필터(Stage B) SQL 구성 검증. 실제 DB 없이 생성된 쿼리 텍스트만 본다."""

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
