"""1차 추림 점수 계산. 임베딩 유사도 30 + DB 조건점수 70.

설계 근거는 백엔드 레포 `.ai/STATE.md` "2026-08-11 갱신 — 매칭 파이프라인 재설계(팀 확정)"의
[2][3] 절이다. 요약하면 **데이터 종류마다 맞는 도구가 다르다**:

- 자유 서술(자기소개·담당업무·우대사항) → 임베딩. 이것만 "주문 시스템 개발 ↔ 결제 API 구축"의
  비슷함을 잡는다
- 집합(스킬) → 교집합 비율. 5개 중 3개 = 60%
- 수치(연차·단가·기간) → 부등호. 임베딩은 `"월 500만원"`과 `"월 5000만원"`을 거의 같게 보고,
  연차는 방향이 반대로 작동한다(`"3년 이상"`에 `"3년"`이 `"10년"`보다 가깝게 나온다)

**여기는 순수 함수만 둔다.** DB도 세션도 안 받는다 — 채점식은 실제 DB 없이 검증할 수 있어야
하고, 이 파일이 그 경계다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# 조건점수 항목별 배점(원점수). 합계는 _CONDITION_MAX 이고, 최종 합산 때 70점으로 환산된다.
# 배점 비율은 팀 확정값이다(U6). 바꾸려면 STATE.md [3] 표와 같이 고쳐야 한다.
_SKILL_MAX = 25.0
_CAREER_MAX = 15.0
_PAY_MAX = 15.0
_WORK_STYLE_MAX = 8.0
_WORK_FORM_MAX = 7.0
_START_DATE_MAX = 5.0
_PERIOD_MAX = 5.0
_CONDITION_MAX = (
    _SKILL_MAX + _CAREER_MAX + _PAY_MAX + _WORK_STYLE_MAX + _WORK_FORM_MAX
    + _START_DATE_MAX + _PERIOD_MAX
)

# 최종 합산 비중. "임베딩 30 : 조건 70"(팀 확정).
SIMILARITY_WEIGHT = 30.0
CONDITION_WEIGHT = 70.0

# 숙련도 보너스: 일치율에 (0.8 + 0.2 x 평균숙련도)를 곱한다. 숙련도는 보너스이지 필수 조건이
# 아니다 — position_skill 에 요구 숙련도 컬럼이 없어서 프로젝트가 "Java 고급 필요"를 표현할
# 방법이 아예 없기 때문이다. 그래서 개수가 이기게 설계했다:
#   5/5 전부 초급 = 25 x 1.0 x 0.8 = 20점  >  3/5 전부 고급 = 25 x 0.6 x 1.0 = 15점
_SKILL_LEVEL_BASE = 0.8
_SKILL_LEVEL_BONUS = 0.2
_SKILL_LEVEL_VALUES = {"BEGINNER": 0.0, "INTERMEDIATE": 0.5, "ADVANCED": 1.0}

# 시작일이 늦을 때 0점이 되는 지연 일수. 두 달 밀리면 사실상 다른 프로젝트다.
_START_DELAY_ZERO_DAYS = 60.0

# 주 -> 개월 환산. 스프링 BudgetCapCalculator/NegotiationConditionCalculator 와 같은 값이어야
# 한다(4주 = 1개월, 2026-08-09 확정). 여기만 다르면 기간 점수가 조용히 어긋난다.
_WEEKS_PER_MONTH = 4.0

# 시급/일급 -> 월단가 환산. budget_cap 이 월단가라 같은 단위로 맞춰야 비교가 성립한다.
# 시급/일급 -> 월단가 환산. **일급 x20, 시급 x160 은 정책 확정값이다.**
# 근거: `.ai/STATE.md` "확정된 설계 결정 3"("프리랜서 단가는 월단가로 통일(일급x20일,
# 시급x160시간)") + 협상 도메인 `FreelancerConditionSnapshot.monthlyPay()` 가 같은 값을 쓴다.
#
# 여기가 협상과 어긋나면 **매칭이 같은 사람을 협상보다 비싸게 봐서 단가 감점이 과하게 들어가고,**
# 정작 협상에 가면 예산 안에 들어온다. 양쪽 다 에러 없이 돌아가서 발견이 어렵다.
# `service.py` 의 `_budget_guidance` 프롬프트 문구(LLM 안내)와도 같아야 한다 — 세 군데다.
_HOURS_PER_MONTH = 160.0
_DAYS_PER_MONTH = 20.0


@dataclass(frozen=True)
class PositionCondition:
    """포지션이 요구하는 조건. 채점 기준이 되는 쪽."""

    required_skills: tuple[str, ...]
    min_career_years: int
    work_style: str | None
    work_form: str | None
    start_desired_date: date | None
    # 프로젝트가 시작일 협의 가능이라고 표시했으면 후보가 늦어도 감점하지 않는다.
    start_negotiable: bool
    period_value: int | None
    period_unit: str | None
    # 스프링이 계산해 넘겨준다. 없으면 단가 항목을 채점에서 뺀다(_score_pay 주석 참고).
    budget_cap: int | None


@dataclass(frozen=True)
class CandidateCondition:
    """후보 한 명의 조건. freelancer_condition + condition_skill 에서 온다."""

    freelancer_id: int
    similarity: float
    # 요구 스킬 중 보유한 것들의 skill_level 목록. 요구 스킬 밖의 보유 스킬은 안 센다.
    matched_skill_levels: tuple[str, ...]
    career_years: int
    pay_unit: str | None
    pay_amount: Decimal | None
    work_style: str | None
    work_form: str | None
    available_from: date | None
    start_negotiable: bool
    period_value: int | None
    period_unit: str | None


@dataclass(frozen=True)
class ScoredCandidate:
    freelancer_id: int
    similarity: float
    similarity_score: float
    condition_score: float
    total_score: float


def _to_months(value: int | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit == "WEEK":
        return value / _WEEKS_PER_MONTH
    return float(value)


def _to_monthly_pay(pay_unit: str | None, pay_amount: Decimal | None) -> float | None:
    """희망 급여를 월단가(원)로 환산한다. 저장 값은 원 단위다.

    화면은 만원 단위로 입력받지만 서버는 만원 배수인지만 검증하고 **원 단위 그대로 저장**한다
    (스프링 `FreelancerCondition.requireInTenThousandUnit` 주석). 여기서 만원으로 착각해
    10000을 곱하면 전원이 예산 초과로 0점이 된다.
    """
    if pay_amount is None or pay_unit is None:
        return None
    amount = float(pay_amount)
    if pay_unit == "HOURLY":
        return amount * _HOURS_PER_MONTH
    if pay_unit == "DAILY":
        return amount * _DAYS_PER_MONTH
    return amount


def _score_skills(position: PositionCondition, candidate: CandidateCondition) -> float:
    if not position.required_skills:
        # 요구 스킬이 없으면 누구도 우열이 없다. 0점을 주면 이 항목이 그냥 사라지는 것과 같아
        # 다른 항목의 비중이 상대적으로 커지므로, 전원 만점으로 둬서 순위에 영향을 없앤다.
        return _SKILL_MAX

    match_ratio = len(candidate.matched_skill_levels) / len(position.required_skills)
    if not candidate.matched_skill_levels:
        return 0.0

    level_values = [_SKILL_LEVEL_VALUES.get(level, 0.0) for level in candidate.matched_skill_levels]
    average_level = sum(level_values) / len(level_values)
    return _SKILL_MAX * match_ratio * (_SKILL_LEVEL_BASE + _SKILL_LEVEL_BONUS * average_level)


def _score_career(position: PositionCondition, candidate: CandidateCondition) -> float:
    """충족하면 만점. 초과 보유에 가점은 없다.

    요구 3년에 10년차나 3년차나 같은 15점이다. "3년이면 충분한데 10년차가 더 나은가"는
    프로젝트 성격에 달린 판단이라 LLM(Stage E)에 맡긴다 — 여기서 가점하면 예산만 비싼
    고연차가 항상 앞에 선다.
    """
    if position.min_career_years <= 0:
        return _CAREER_MAX
    if candidate.career_years >= position.min_career_years:
        return _CAREER_MAX
    return _CAREER_MAX * (candidate.career_years / position.min_career_years)


def _score_pay(position: PositionCondition, candidate: CandidateCondition) -> float:
    """상한 이내면 만점, 넘으면 초과율만큼 감점.

    budget_cap 이 없으면(옛 스프링 배포) 비교 기준 자체가 없다. 이때 0점을 주면 전원이 똑같이
    깎여 순위가 안 변하는 대신 조건점수 총합만 낮아지고, 결과적으로 유사도 30의 비중이 커진다.
    그래서 만점을 준다 — 이 항목을 채점에서 빼는 것과 같은 효과다.
    """
    monthly_pay = _to_monthly_pay(candidate.pay_unit, candidate.pay_amount)
    if position.budget_cap is None or monthly_pay is None or position.budget_cap <= 0:
        return _PAY_MAX
    if monthly_pay <= position.budget_cap:
        return _PAY_MAX
    excess_ratio = (monthly_pay - position.budget_cap) / position.budget_cap
    return max(0.0, _PAY_MAX * (1 - excess_ratio))


def _score_exact_match(required: str | None, offered: str | None, full_score: float) -> float:
    """근무방식/근무형태처럼 값이 같아야 하는 항목. ANY 는 양쪽 어디에 있든 통과다."""
    if required is None or offered is None:
        return full_score
    if required == "ANY" or offered == "ANY" or required == offered:
        return full_score
    return 0.0


def _score_start_date(position: PositionCondition, candidate: CandidateCondition) -> float:
    # 어느 한쪽이라도 협의 가능이면 감점하지 않는다. 프로젝트가 "시작일 협의 가능"으로 올려놨는데
    # 후보만 늦다고 깎으면, 정작 클라이언트는 신경 쓰지 않는 항목으로 순위가 밀린다.
    if candidate.start_negotiable or position.start_negotiable:
        return _START_DATE_MAX
    if position.start_desired_date is None or candidate.available_from is None:
        return _START_DATE_MAX
    delay_days = (candidate.available_from - position.start_desired_date).days
    if delay_days <= 0:
        return _START_DATE_MAX
    return max(0.0, _START_DATE_MAX * (1 - delay_days / _START_DELAY_ZERO_DAYS))


def _score_period(position: PositionCondition, candidate: CandidateCondition) -> float:
    """희망 기간이 요구 기간 이상이면 만점, 짧으면 비율만큼.

    후보가 기간을 안 적었으면 협의 가능으로 본다(입력이 선택 항목이라 '짧다'로 볼 근거가 없다).
    """
    required_months = _to_months(position.period_value, position.period_unit)
    offered_months = _to_months(candidate.period_value, candidate.period_unit)
    if required_months is None or offered_months is None or required_months <= 0:
        return _PERIOD_MAX
    if offered_months >= required_months:
        return _PERIOD_MAX
    return _PERIOD_MAX * (offered_months / required_months)


def condition_score(position: PositionCondition, candidate: CandidateCondition) -> float:
    """조건점수를 0~1 로 돌려준다. 항목별 원점수 합계를 만점으로 나눈 값이다."""
    raw = (
        _score_skills(position, candidate)
        + _score_career(position, candidate)
        + _score_pay(position, candidate)
        + _score_exact_match(position.work_style, candidate.work_style, _WORK_STYLE_MAX)
        + _score_exact_match(position.work_form, candidate.work_form, _WORK_FORM_MAX)
        + _score_start_date(position, candidate)
        + _score_period(position, candidate)
    )
    return raw / _CONDITION_MAX


def _percent_ranks(similarities: list[float]) -> list[float]:
    """유사도를 순위 기반 0~1 로 편다. SQL 의 PERCENT_RANK 와 같은 정의다.

    코사인 유사도는 0.55~0.85 같은 좁은 구간에 몰려서, 그냥 x30 하면 "고정 보너스 16점 +
    변동 9점"이 되어 **조건점수가 순위를 100% 결정**한다. 30:70 이라는 비중이 무의미해진다.
    순위로 펴면 분포가 어떻든 항상 0~30 전체를 쓰고, 정규화 상수를 실측할 필요도 없다.

    대가로 절대적 유사도 차이는 무시된다(1등과 2등이 거의 같아도 순위만큼 벌어진다).
    후보가 1명이면 전원 0점인데, 그때는 순위를 매길 대상이 없으니 문제가 되지 않는다.
    """
    count = len(similarities)
    if count <= 1:
        return [0.0] * count

    # 동점은 같은 순위를 받아야 한다(PERCENT_RANK 정의). 오름차순 정렬 후 첫 등장 위치를 쓴다.
    ascending = sorted(similarities)
    first_index = {}
    for index, value in enumerate(ascending):
        first_index.setdefault(value, index)
    return [first_index[value] / (count - 1) for value in similarities]


def score_candidates(
    position: PositionCondition, candidates: list[CandidateCondition]
) -> list[ScoredCandidate]:
    """합산 점수 내림차순으로 정렬해 돌려준다. 자르지는 않는다.

    **합산이지 순차가 아니다.** 유사도로 먼저 상위 N을 뽑고 그 안에서 조건 정렬하면 누가 후보가
    되는지를 유사도가 100% 정하게 되어 70이라는 비중이 사라진다. 조건이 완벽한데 자기소개가
    짧아 유사도가 낮은 사람이 아예 안 나오면 안 된다.
    """
    ranks = _percent_ranks([candidate.similarity for candidate in candidates])

    scored = []
    for candidate, rank in zip(candidates, ranks, strict=True):
        similarity_score = rank * SIMILARITY_WEIGHT
        condition = condition_score(position, candidate) * CONDITION_WEIGHT
        scored.append(
            ScoredCandidate(
                freelancer_id=candidate.freelancer_id,
                similarity=candidate.similarity,
                similarity_score=similarity_score,
                condition_score=condition,
                total_score=similarity_score + condition,
            )
        )

    # 동점이면 freelancer_id 오름차순 — 순서가 매번 달라지면 같은 입력에 다른 추천이 나온다.
    scored.sort(key=lambda item: (-item.total_score, item.freelancer_id))
    return scored
