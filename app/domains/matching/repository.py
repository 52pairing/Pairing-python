"""스프링 소유 테이블(project/project_position/freelancer_condition/resume 등) 읽기 전용 조회.

`app/db/session.py`의 소유권 규칙대로 이 테이블들은 SELECT 만 한다. LLM 재랭킹 프롬프트에 넣을
포지션 요구조건·프리랜서 이력 원문을 모으는 용도로, AI 서버가 쓰기하는 embedding 테이블과는 무관하다.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class PositionRequirement:
    project_title: str
    job_category: str
    job_role: str
    min_career_years: int
    current_situation: str | None
    main_task: str | None
    detail_scope: str | None
    extra_note: str | None
    work_style: str
    work_form: str
    skills: list[str]
    # 일정/근무조건/단가는 하드필터로 후보를 배제하지 않고 Stage E(LLM 최종선정)에서 감점 요인으로만
    # 쓴다(.ai/STATE.md "Stage B 조건필터 폐기" 참고). 그래서 여기서 비교용 원본 값을 같이 읽는다.
    budget_amount: Decimal | None
    period_value: int | None
    period_unit: str | None
    start_desired_date: date | None
    start_negotiable: bool


@dataclass
class FreelancerProfile:
    freelancer_id: int
    # 운영 로그에서 "누가 최종 추천됐나"를 확인하려고 가져온다.
    # **프롬프트에는 절대 넣지 않는다** — LLM 이 이름으로 사람을 편향 판단할 수 있고
    # (_describe_candidate 가 이 필드를 쓰지 않는 이유), 추천 근거는 이력 내용이어야 한다.
    name: str | None
    job_category: str
    job_role: str
    career_years: int
    has_freelance_exp: bool
    self_introduction: str | None
    career_summary: str | None
    skills: list[str]
    pay_unit: str | None
    pay_amount: Decimal | None
    work_style: str | None
    work_form: str | None
    available_from: date | None
    start_negotiable: bool
    period_value: int | None
    period_unit: str | None


class DirectoryRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def find_position_requirement(self, position_id: int) -> PositionRequirement | None:
        row = (
            await self._session.execute(
                text(
                    """
                    SELECT p.title, pp.job_category, pp.job_role, pp.min_career_years,
                           p.current_situation, p.main_task, p.detail_scope,
                           p.extra_note, p.work_style, p.work_form,
                           p.budget_amount, p.period_value, p.period_unit,
                           p.start_desired_date, p.start_negotiable
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
            current_situation=row["current_situation"],
            main_task=row["main_task"],
            detail_scope=row["detail_scope"],
            extra_note=row["extra_note"],
            work_style=row["work_style"],
            work_form=row["work_form"],
            skills=list(skills),
            budget_amount=row["budget_amount"],
            period_value=row["period_value"],
            period_unit=row["period_unit"],
            start_desired_date=row["start_desired_date"],
            start_negotiable=row["start_negotiable"],
        )

    async def find_freelancer_profiles(
        self, freelancer_ids: list[int]
    ) -> dict[int, FreelancerProfile]:
        if not freelancer_ids:
            return {}

        # freelancer_condition/resume은 freelancer_profile.id가 아니라 account_id로 연결된다
        # (스프링 JPA 엔티티 컬럼명 기준 — db/init/*.sql 문서는 여기서 낡아 있다). freelancer_embedding이
        # 쓰는 id는 freelancer_profile.id라 반드시 freelancer_profile을 거쳐 account_id로 다리를 놓는다.
        rows = (
            await self._session.execute(
                text(
                    """
                    SELECT fp.id AS freelancer_id, a.name, fc.job_category, fc.job_role,
                           fc.career_years,
                           fc.has_freelance_experience AS has_freelance_exp, r.self_introduction,
                           fc.pay_unit, fc.pay_amount, fc.work_style, fc.work_form,
                           fc.available_from, fc.start_negotiable, fc.period_value, fc.period_unit,
                           COALESCE(string_agg(
                               rc.company_name || ' ' || COALESCE(rc.department_rank, '') || ': '
                                   || COALESCE(rc.job_description, ''),
                               ' / ' ORDER BY rc.start_date DESC
                           ), '') AS career_summary
                    FROM freelancer_profile fp
                    JOIN account a ON a.id = fp.account_id
                    JOIN freelancer_condition fc ON fc.account_id = fp.account_id
                    LEFT JOIN resume r ON r.account_id = fp.account_id
                    LEFT JOIN resume_career rc ON rc.resume_id = r.id
                    WHERE fp.id = ANY(:freelancer_ids)
                    GROUP BY fp.id, a.name, fc.job_category, fc.job_role, fc.career_years,
                             fc.has_freelance_experience, r.self_introduction,
                             fc.pay_unit, fc.pay_amount, fc.work_style, fc.work_form,
                             fc.available_from, fc.start_negotiable, fc.period_value, fc.period_unit
                    """
                ),
                {"freelancer_ids": freelancer_ids},
            )
        ).mappings().all()

        skill_rows = (
            await self._session.execute(
                text(
                    """
                    SELECT fp.id AS freelancer_id, cs.skill_code
                    FROM freelancer_profile fp
                    JOIN freelancer_condition fc ON fc.account_id = fp.account_id
                    JOIN condition_skill cs ON cs.condition_id = fc.id
                    WHERE fp.id = ANY(:freelancer_ids)
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
                name=row["name"],
                job_category=row["job_category"],
                job_role=row["job_role"],
                career_years=row["career_years"],
                has_freelance_exp=row["has_freelance_exp"],
                self_introduction=row["self_introduction"],
                career_summary=row["career_summary"],
                skills=skills_by_freelancer.get(row["freelancer_id"], []),
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
        }
