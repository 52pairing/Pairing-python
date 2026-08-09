"""스프링 소유 테이블(project/project_position/freelancer_condition/resume 등) 읽기 전용 조회.

`app/db/session.py`의 소유권 규칙대로 이 테이블들은 SELECT 만 한다. LLM 재랭킹 프롬프트에 넣을
포지션 요구조건·프리랜서 이력 원문을 모으는 용도로, AI 서버가 쓰기하는 embedding 테이블과는 무관하다.
"""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class PositionRequirement:
    project_title: str
    job_category: str
    job_role: str
    min_career_years: int
    preferred_note: str | None
    current_situation: str | None
    main_task: str | None
    detail_scope: str | None
    extra_note: str | None
    work_style: str
    work_form: str
    skills: list[str]


@dataclass
class FreelancerProfile:
    freelancer_id: int
    job_category: str
    job_role: str
    career_years: int
    has_freelance_exp: bool
    self_introduction: str | None
    career_summary: str | None
    skills: list[str]


class DirectoryRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_position_requirement(self, position_id: int) -> PositionRequirement | None:
        row = (
            await self._session.execute(
                text(
                    """
                    SELECT p.title, pp.job_category, pp.job_role, pp.min_career_years,
                           pp.preferred_note, p.current_situation, p.main_task, p.detail_scope,
                           p.extra_note, p.work_style, p.work_form
                    FROM project_position pp
                    JOIN project p ON p.id = pp.project_id
                    WHERE pp.id = :position_id
                    """
                ),
                {"position_id": position_id},
            )
        ).mappings().first()
        if row is None:
            return None

        skills = (
            await self._session.execute(
                text("SELECT skill_code FROM position_skill WHERE position_id = :position_id"),
                {"position_id": position_id},
            )
        ).scalars().all()

        return PositionRequirement(
            project_title=row["title"],
            job_category=row["job_category"],
            job_role=row["job_role"],
            min_career_years=row["min_career_years"],
            preferred_note=row["preferred_note"],
            current_situation=row["current_situation"],
            main_task=row["main_task"],
            detail_scope=row["detail_scope"],
            extra_note=row["extra_note"],
            work_style=row["work_style"],
            work_form=row["work_form"],
            skills=list(skills),
        )

    async def find_freelancer_profiles(
        self, freelancer_ids: list[int]
    ) -> dict[int, FreelancerProfile]:
        if not freelancer_ids:
            return {}

        rows = (
            await self._session.execute(
                text(
                    """
                    SELECT fc.freelancer_id, fc.job_category, fc.job_role, fc.career_years,
                           fc.has_freelance_exp, r.self_introduction,
                           COALESCE(string_agg(
                               rc.company_name || ' ' || COALESCE(rc.department_rank, '') || ': '
                                   || COALESCE(rc.job_description, ''),
                               ' / ' ORDER BY rc.start_date DESC
                           ), '') AS career_summary
                    FROM freelancer_condition fc
                    LEFT JOIN resume r ON r.freelancer_id = fc.freelancer_id
                    LEFT JOIN resume_career rc ON rc.resume_id = r.id
                    WHERE fc.freelancer_id = ANY(:freelancer_ids)
                    GROUP BY fc.freelancer_id, fc.job_category, fc.job_role, fc.career_years,
                             fc.has_freelance_exp, r.self_introduction
                    """
                ),
                {"freelancer_ids": freelancer_ids},
            )
        ).mappings().all()

        skill_rows = (
            await self._session.execute(
                text(
                    """
                    SELECT fc.freelancer_id, cs.skill_code
                    FROM freelancer_condition fc
                    JOIN condition_skill cs ON cs.condition_id = fc.id
                    WHERE fc.freelancer_id = ANY(:freelancer_ids)
                    """
                ),
                {"freelancer_ids": freelancer_ids},
            )
        ).all()
        skills_by_freelancer: dict[int, list[str]] = {}
        for freelancer_id, skill_code in skill_rows:
            skills_by_freelancer.setdefault(freelancer_id, []).append(skill_code)

        return {
            row["freelancer_id"]: FreelancerProfile(
                freelancer_id=row["freelancer_id"],
                job_category=row["job_category"],
                job_role=row["job_role"],
                career_years=row["career_years"],
                has_freelance_exp=row["has_freelance_exp"],
                self_introduction=row["self_introduction"],
                career_summary=row["career_summary"],
                skills=skills_by_freelancer.get(row["freelancer_id"], []),
            )
            for row in rows
        }
