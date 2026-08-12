"""조건점수 채점식 검증. DB도 LLM도 안 쓴다 — 순수 함수라 숫자를 직접 확인할 수 있다.

기준은 백엔드 레포 `.ai/STATE.md` "[2][3] 임베딩 30 + 조건점수 70" 표다. 그 표의 검산 예시를
그대로 테스트로 옮겨서, 표와 코드가 어긋나면 여기서 걸리게 한다.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.domains.matching.scoring import (
    CONDITION_WEIGHT,
    SIMILARITY_WEIGHT,
    CandidateCondition,
    PositionCondition,
    condition_score,
    score_candidates,
)

# 조건점수 원점수 만점. 25+15+15+8+7+5+5.
_CONDITION_MAX = 80.0


def _position(**overrides) -> PositionCondition:
    defaults = {
        "required_skills": ("SPRING_BOOT", "POSTGRESQL", "JAVA", "REDIS", "DOCKER"),
        "min_career_years": 3,
        "work_style": "REMOTE",
        "work_form": "FULL_TIME",
        "start_desired_date": date(2026, 9, 1),
        "start_negotiable": False,
        "period_value": 6,
        "period_unit": "MONTH",
        "budget_cap": 9_000_000,
    }
    return PositionCondition(**{**defaults, **overrides})


def _candidate(freelancer_id: int = 1, **overrides) -> CandidateCondition:
    """모든 항목 만점인 후보. 한 항목씩 덮어써서 그 항목만 검증한다."""
    defaults = {
        "freelancer_id": freelancer_id,
        "similarity": 0.8,
        "matched_skill_levels": ("ADVANCED",) * 5,
        "career_years": 5,
        "pay_unit": "MONTHLY",
        "pay_amount": Decimal("8000000"),
        "work_style": "REMOTE",
        "work_form": "FULL_TIME",
        "available_from": date(2026, 8, 1),
        "start_negotiable": False,
        "period_value": 6,
        "period_unit": "MONTH",
    }
    return CandidateCondition(**{**defaults, **overrides})


def _raw(position: PositionCondition, candidate: CandidateCondition) -> float:
    """0~1 로 나온 조건점수를 원점수(0~80)로 되돌린다. 표와 직접 비교하려고."""
    return condition_score(position, candidate) * _CONDITION_MAX


def test_all_conditions_met_scores_full_marks():
    assert _raw(_position(), _candidate()) == pytest.approx(_CONDITION_MAX)


# --- 스킬 25점 -------------------------------------------------------------


def test_skill_count_beats_proficiency():
    """STATE.md 검산: 5/5 전부 초급(20점) > 3/5 전부 고급(15점).

    숙련도는 보너스이지 필수 조건이 아니다 — position_skill 에 요구 숙련도 컬럼이 없어서
    프로젝트가 "Java 고급 필요"를 표현할 방법 자체가 없다. 그래서 개수가 이겨야 한다.
    """
    all_beginner = _candidate(matched_skill_levels=("BEGINNER",) * 5)
    three_advanced = _candidate(matched_skill_levels=("ADVANCED",) * 3)

    # 25 x 1.0 x (0.8 + 0.2x0.0) = 20
    assert _raw(_position(), all_beginner) - _raw(_position(), _candidate()) == pytest.approx(-5.0)
    # 25 x 0.6 x (0.8 + 0.2x1.0) = 15
    assert _raw(_position(), three_advanced) - _raw(_position(), _candidate()) == pytest.approx(-10.0)

    assert condition_score(_position(), all_beginner) > condition_score(_position(), three_advanced)


def test_no_matched_skill_scores_zero_for_skills():
    """스킬 0개는 하드필터에서 걸리지만, 완화 재검색으로 들어올 수 있다."""
    none_matched = _candidate(matched_skill_levels=())
    assert _raw(_position(), none_matched) == pytest.approx(_CONDITION_MAX - 25.0)


def test_position_without_required_skills_gives_everyone_full_skill_score():
    """요구 스킬이 없으면 우열이 없다. 0점을 주면 이 항목이 사라져 다른 항목 비중만 커진다."""
    position = _position(required_skills=())
    assert _raw(position, _candidate(matched_skill_levels=())) == pytest.approx(_CONDITION_MAX)


# --- 경력 15점 -------------------------------------------------------------


def test_career_shortfall_is_prorated():
    """STATE.md 검산: 요구 3년·보유 1년 = 5점."""
    assert _raw(_position(), _candidate(career_years=1)) == pytest.approx(_CONDITION_MAX - 10.0)


def test_extra_career_years_get_no_bonus():
    """요구 3년에 10년차나 3년차나 같은 15점이다.

    가점하면 예산만 비싼 고연차가 항상 앞에 선다. "3년이면 충분한데 10년차가 더 나은가"는
    프로젝트 성격에 달린 판단이라 LLM(Stage E)에 맡긴다.
    """
    assert _raw(_position(), _candidate(career_years=3)) == pytest.approx(
        _raw(_position(), _candidate(career_years=10))
    )


# --- 단가 15점 -------------------------------------------------------------


def test_pay_over_cap_is_prorated_by_excess_ratio():
    """STATE.md 검산: cap 900만·희망 1,100만 = 11.7점. 초과율 2/9 → 15 x (1-0.2222)."""
    over = _candidate(pay_amount=Decimal("11000000"))
    expected_pay = 15.0 * (1 - (11_000_000 - 9_000_000) / 9_000_000)
    assert expected_pay == pytest.approx(11.666, abs=0.01)
    assert _raw(_position(), over) == pytest.approx(_CONDITION_MAX - 15.0 + expected_pay)


def test_pay_far_over_cap_floors_at_zero():
    """초과율이 100%를 넘으면 음수가 되므로 0에서 멈춰야 한다."""
    assert _raw(_position(), _candidate(pay_amount=Decimal("99000000"))) == pytest.approx(
        _CONDITION_MAX - 15.0
    )


def test_hourly_and_daily_pay_are_converted_to_monthly():
    """시급/일급은 월단가로 환산해야 cap 과 단위가 맞는다."""
    hourly = _candidate(pay_unit="HOURLY", pay_amount=Decimal("40000"))  # x160 = 640만
    daily = _candidate(pay_unit="DAILY", pay_amount=Decimal("400000"))  # x20 = 800만
    assert _raw(_position(), hourly) == pytest.approx(_CONDITION_MAX)
    assert _raw(_position(), daily) == pytest.approx(_CONDITION_MAX)

    # 환산을 빠뜨리면 40000원이 상한 이내로 보여 똑같이 만점이 나온다. 넘는 값으로도 확인한다.
    expensive_hourly = _candidate(pay_unit="HOURLY", pay_amount=Decimal("100000"))  # 1600만
    assert _raw(_position(), expensive_hourly) < _CONDITION_MAX


def test_pay_conversion_matches_negotiation_domain():
    """일급 x20 / 시급 x160 은 정책 확정값이다.

    근거: `.ai/STATE.md` "확정된 설계 결정 3" + 협상 도메인
    `FreelancerConditionSnapshot.monthlyPay()`(DAYS_PER_MONTH=20, HOURS_PER_MONTH=160).
    어긋나면 매칭이 같은 사람을 협상보다 비싸게 봐서 단가 감점이 과하게 들어가고, 정작
    협상에 가면 예산 안에 들어온다. 양쪽 다 에러 없이 돌아가 발견이 어렵다.

    상한과 딱 맞아떨어지는 값으로 확인한다 — 배수가 달라지면 바로 넘어가서 만점이 깨진다.
    """
    position = _position(budget_cap=10_000_000)

    # 일급 50만 x 20일 = 정확히 1,000만. x21 이면 1,050만이라 초과한다.
    assert _raw(position, _candidate(pay_unit="DAILY", pay_amount=Decimal("500000"))) == (
        pytest.approx(_CONDITION_MAX)
    )
    assert _raw(position, _candidate(pay_unit="DAILY", pay_amount=Decimal("510000"))) < _CONDITION_MAX

    # 시급 62,500 x 160시간 = 정확히 1,000만. x209 면 1,306만이라 초과한다.
    assert _raw(position, _candidate(pay_unit="HOURLY", pay_amount=Decimal("62500"))) == (
        pytest.approx(_CONDITION_MAX)
    )
    assert _raw(position, _candidate(pay_unit="HOURLY", pay_amount=Decimal("63000"))) < _CONDITION_MAX


def test_pay_amount_is_stored_in_won_not_ten_thousand_won():
    """화면은 만원 단위로 받지만 DB는 원 단위 그대로다(스프링 requireInTenThousandUnit 주석).

    만원으로 착각해 10000을 곱하면 전원이 예산 초과로 0점이 된다.
    """
    # 800만원 = 8000000. cap 900만 이내라 만점이어야 한다.
    assert _raw(_position(), _candidate(pay_amount=Decimal("8000000"))) == pytest.approx(
        _CONDITION_MAX
    )


def test_missing_budget_cap_skips_pay_scoring():
    """budget_cap 이 없으면(옛 스프링 배포) 기준이 없다. 0점을 주면 조건 총합만 낮아져
    유사도 30의 비중이 커진다. 만점을 줘서 항목을 빼는 것과 같게 만든다."""
    position = _position(budget_cap=None)
    assert _raw(position, _candidate(pay_amount=Decimal("99000000"))) == pytest.approx(
        _CONDITION_MAX
    )


# --- 근무방식 8 / 근무형태 7 -------------------------------------------------


def test_work_style_mismatch_loses_all_eight_points():
    assert _raw(_position(), _candidate(work_style="ONSITE")) == pytest.approx(_CONDITION_MAX - 8.0)


def test_any_matches_everything_on_either_side():
    assert _raw(_position(), _candidate(work_style="ANY")) == pytest.approx(_CONDITION_MAX)
    assert _raw(_position(work_style="ANY"), _candidate(work_style="ONSITE")) == pytest.approx(
        _CONDITION_MAX
    )


# --- 시작일 5 --------------------------------------------------------------


def test_late_start_is_prorated_over_sixty_days():
    late = _candidate(available_from=date(2026, 10, 1))  # 30일 지연
    assert _raw(_position(), late) == pytest.approx(_CONDITION_MAX - 2.5)


def test_negotiable_start_date_is_never_penalized():
    late = _candidate(available_from=date(2027, 1, 1), start_negotiable=True)
    assert _raw(_position(), late) == pytest.approx(_CONDITION_MAX)

    # 프로젝트가 협의 가능이라고 했으면 후보가 늦어도 감점하지 않는다.
    position = _position(start_negotiable=True)
    assert _raw(position, _candidate(available_from=date(2027, 1, 1))) == pytest.approx(
        _CONDITION_MAX
    )


# --- 기간 5 ----------------------------------------------------------------


def test_shorter_period_is_prorated():
    assert _raw(_position(), _candidate(period_value=3)) == pytest.approx(_CONDITION_MAX - 2.5)


def test_weeks_are_converted_to_months_at_four_weeks_per_month():
    """스프링 BudgetCapCalculator 와 같은 값(4주=1개월)이어야 한다."""
    twenty_four_weeks = _candidate(period_value=24, period_unit="WEEK")  # = 6개월
    assert _raw(_position(), twenty_four_weeks) == pytest.approx(_CONDITION_MAX)


def test_missing_period_is_treated_as_negotiable():
    """기간은 선택 입력이라 안 적었다고 '짧다'로 볼 근거가 없다."""
    assert _raw(_position(), _candidate(period_value=None)) == pytest.approx(_CONDITION_MAX)


# --- 합산 30:70 ------------------------------------------------------------


def test_similarity_uses_full_thirty_point_range_regardless_of_distribution():
    """코사인 유사도는 0.55~0.85 같은 좁은 구간에 몰린다. 그대로 x30 하면 변동폭이 9점뿐이라
    조건점수가 순위를 100% 결정한다. 순위 기반이면 분포와 무관하게 0~30을 다 쓴다."""
    candidates = [
        _candidate(1, similarity=0.55),
        _candidate(2, similarity=0.70),
        _candidate(3, similarity=0.85),
    ]

    scored = {item.freelancer_id: item for item in score_candidates(_position(), candidates)}

    assert scored[3].similarity_score == pytest.approx(SIMILARITY_WEIGHT)
    assert scored[2].similarity_score == pytest.approx(SIMILARITY_WEIGHT / 2)
    assert scored[1].similarity_score == pytest.approx(0.0)


def test_good_conditions_beat_high_similarity():
    """이번 재설계의 핵심. 유사도로 먼저 자르면 이 사람이 아예 안 나온다.

    자기소개가 짧아 유사도 꼴찌(임베딩 0점)지만 조건이 완벽한 사람 vs 유사도 1등(30점)인데
    조건이 나쁜 사람. 조건이 70점을 쥐고 있으므로 전자가 이겨야 한다.
    """
    perfect_conditions = _candidate(1, similarity=0.50)
    high_similarity_only = _candidate(
        2,
        similarity=0.95,
        matched_skill_levels=("BEGINNER",),
        career_years=1,
        work_style="ONSITE",
        pay_amount=Decimal("20000000"),
    )

    ranked = score_candidates(_position(), [perfect_conditions, high_similarity_only])

    assert [item.freelancer_id for item in ranked] == [1, 2]


def test_similarity_can_outweigh_a_moderate_condition_gap():
    """30:70 의 실제 경계. 유사도 30점은 **조건 원점수 34점(=70x34/80)** 까지 뒤집는다.

    스킬 1/5·경력 1년(요구 3년)이면 조건 원점수가 80 → 49 로 31점 깎이는데, 그래도 유사도
    1등이면 이긴다. "조건 70이니 조건이 항상 이긴다"가 아니라는 뜻이다 — 설계상 의도된
    동작이지만 직관과 어긋나므로 숫자로 박아둔다. 이게 과하다고 판단되면 30:70 비중이나
    순위 기반 정규화를 손봐야 하고, 그때 이 테스트가 먼저 깨진다.
    """
    good_conditions = _candidate(1, similarity=0.50)
    high_similarity = _candidate(
        2, similarity=0.95, matched_skill_levels=("BEGINNER",), career_years=1
    )

    ranked = score_candidates(_position(), [good_conditions, high_similarity])

    assert [item.freelancer_id for item in ranked] == [2, 1]
    # 뒤집히는 지점: 조건 격차 x (70/80) < 유사도 격차 30
    assert ranked[0].total_score - ranked[1].total_score == pytest.approx(
        30.0 - (80.0 - 49.0) * CONDITION_WEIGHT / _CONDITION_MAX
    )


def test_total_is_similarity_plus_condition_on_a_hundred_point_scale():
    single = _candidate(1)
    [scored] = score_candidates(_position(), [single])

    # 후보가 1명이면 순위를 매길 대상이 없어 유사도는 0점이다.
    assert scored.similarity_score == pytest.approx(0.0)
    assert scored.condition_score == pytest.approx(CONDITION_WEIGHT)
    assert scored.total_score == pytest.approx(CONDITION_WEIGHT)


def test_ties_are_broken_by_freelancer_id_for_stable_output():
    """동점 순서가 매번 달라지면 같은 입력에 다른 추천이 나간다."""
    candidates = [_candidate(9, similarity=0.7), _candidate(3, similarity=0.7)]

    ranked = score_candidates(_position(), candidates)

    assert [item.freelancer_id for item in ranked] == [3, 9]
