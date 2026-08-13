"""매칭 추천.

세 단계로 나눈다.
  1) 하드필터로 자격 없는 사람을 뺀다 (여기서만 후보가 배제된다)
  2) 임베딩 유사도 25 + DB 조건점수 75 를 **합산**해 상위 (모집 인원 x 3)명으로 추린다
  3) LLM 이 그 풀만 보고 순위와 사유를 만든다

전체 프리랜서를 LLM 에 넣지 않는 이유는 비용·지연·컨텍스트 한계 셋 다다.
2)를 합산으로 하는 이유와 채점식은 `app/domains/matching/scoring.py` 참고.
"""

import json
import logging

from app.clients.gemini import GeminiClient, GeminiTask, GeminiUsage
from app.core.errors import AiErrorCode, AiException
from app.domains.ai_log.repository import (
    AgentType,
    AiAgentLogRepository,
    AiCallRecord,
    LogStatus,
    RefType,
)
from app.domains.embedding.repository import CandidateConditionRow
from app.domains.embedding.service import EmbeddingService
from app.domains.matching.repository import (
    DirectoryRepository,
    FreelancerProfile,
    PositionRequirement,
)
from app.domains.matching.schemas import MatchingResponse, RankedCandidate
from app.domains.matching.scoring import (
    CandidateCondition,
    PositionCondition,
    score_breakdown,
    score_candidates,
)

logger = logging.getLogger(__name__)

_PERIOD_LABELS = {"MONTH": "개월", "WEEK": "주"}
_PAY_UNIT_LABELS = {"HOURLY": "시급", "DAILY": "일급", "MONTHLY": "월급"}


def _period_label(unit: str | None) -> str:
    return _PERIOD_LABELS.get(unit, unit or "")


def _pay_unit_label(unit: str | None) -> str:
    return _PAY_UNIT_LABELS.get(unit, unit or "")


def _to_position_condition(
    position: PositionRequirement, budget_cap: int | None
) -> PositionCondition:
    return PositionCondition(
        required_skills=tuple(position.skills),
        min_career_years=position.min_career_years,
        work_style=position.work_style,
        work_form=position.work_form,
        start_desired_date=position.start_desired_date,
        start_negotiable=position.start_negotiable,
        period_value=position.period_value,
        period_unit=position.period_unit,
        budget_cap=budget_cap,
    )


def _to_candidate_condition(row: CandidateConditionRow) -> CandidateCondition:
    return CandidateCondition(
        freelancer_id=row.freelancer_id,
        similarity=row.similarity,
        matched_skill_levels=tuple(row.matched_skill_levels),
        career_years=row.career_years,
        pay_unit=row.pay_unit,
        pay_amount=row.pay_amount,
        work_style=row.work_style,
        work_form=row.work_form,
        available_from=row.available_from,
        start_negotiable=row.start_negotiable,
        period_value=row.period_value,
        period_unit=row.period_unit,
    )


def _row_debug(row: CandidateConditionRow) -> dict[str, object]:
    return {
        "freelancer_id": row.freelancer_id,
        "similarity": round(row.similarity, 6),
        "matched_skill_levels": row.matched_skill_levels,
        "career_years": row.career_years,
        "pay_unit": row.pay_unit,
        "pay_amount": str(row.pay_amount) if row.pay_amount is not None else None,
        "work_style": row.work_style,
        "work_form": row.work_form,
        "available_from": str(row.available_from) if row.available_from is not None else None,
        "start_negotiable": row.start_negotiable,
        "period_value": row.period_value,
        "period_unit": row.period_unit,
    }


def _score_debug(value: float) -> float:
    return round(value, 4)


def _budget_guidance(budget_cap: int | None) -> str:
    """단가 판단 지시문. budget_cap 유무로 문구가 완전히 달라진다.

    budget_cap 은 스프링이 계산해 넘겨주는 '1인 월단가 상한'이다. 이게 있으면 후보의 희망
    급여와 같은 단위라 바로 비교할 수 있다. 없으면 프롬프트에 있는 건 프로젝트 총예산뿐인데,
    그건 전체 인원 x 전체 기간 합계라 1인 월급과 자릿수가 다르다 — 그대로 비교하면 멀쩡한
    후보가 전부 "예산 초과"로 감점된다. 그래서 없을 때는 비교하지 말라고 명시적으로 막는다.
    """
    if budget_cap is None:
        return (
            "단가 주의: 이 요청에는 1인 단가 상한이 없다. 프로젝트 총예산은 전체 인원·전체 기간을 "
            "합친 금액이라 후보의 1인 희망 급여와 자릿수가 다르다. **두 숫자를 직접 비교하지 마라.** "
            "단가는 판단에서 빼고 경력·스킬·업무 적합도로만 순위를 매겨라.\n"
        )
    return (
        f"1인 월단가 상한: {budget_cap}원 (수수료를 뺀 순예산을 인원수와 개월 수로 나눈 값). "
        "후보의 희망 급여를 월단가로 환산해서 이 값과 비교해라 — 시급이면 x 160시간, "
        "일급이면 x 20일이 월 환산 기준이다. 상한을 넘으면 넘는 폭에 비례해 감점하고 reason에 "
        "적어라. 협상 여지가 있으므로 조금 넘는 정도로는 크게 감점하지 마라. "
        "프로젝트 총예산은 참고용이니 후보 급여와 직접 비교하지 마라.\n"
    )


# 이 값 이하만 나오면 LLM 이 0~100 이 아닌 다른 스케일(0~1, 0~10)로 답했다고 본다.
# 후보들은 이미 하드필터(직군/직무 일치)와 벡터 유사도 상위를 통과한 사람들이라, 전원이
# 100점 만점에 10점 이하인 상황은 사실상 나오지 않는다.
_SUSPICIOUS_MAX_SCORE = 10


def _assert_score_scale(candidates: list[RankedCandidate]) -> None:
    """점수 스케일이 0~100 인지 검증한다.

    `RankedCandidate.score` 의 ge/le 는 범위 밖(음수·100 초과)만 막는다. 정작 실제로 났던 사고는
    LLM 이 10점 만점으로 답한 것(`score=9.5`)이라 범위 검증에 안 걸린다 — 스프링은 이 값을
    base_score(0~100)에 그대로 저장하고 50점 미만을 '적합도 낮음'으로 보기 때문에, 모든 후보가
    조용히 저품질로 찍힌다. 에러가 안 나서 발견이 어려운 종류의 사고라 여기서 명시적으로 막는다.

    값을 추측해서 보정(예: ×10)하지 않고 실패시킨다. 잘못 보정하면 결국 또 조용히 틀린 점수가
    쌓이고, 그건 지금 막으려는 문제와 같기 때문이다. 순위 자체는 스케일과 무관하게 유지되지만
    lowScoreWarned 판정이 망가지므로 그냥 넘기지도 않는다.
    """
    if not candidates:
        return
    highest = max(candidate.score for candidate in candidates)
    if highest <= _SUSPICIOUS_MAX_SCORE:
        logger.warning(
            "LLM 점수 스케일 이상: 최고점이 %s 다. 0~100 이 아닌 다른 스케일로 답한 것으로 보인다. "
            "프롬프트의 점수 범위 지시를 확인할 것.",
            highest,
        )
        raise AiException(
            AiErrorCode.LLM_RESPONSE_INVALID,
            f"추천 점수가 0~100 스케일이 아닙니다. (최고점 {highest})",
        )


_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "freelancer_id": {"type": "integer"},
                    # 범위를 스키마에도 박아둔다. 프롬프트 지시는 강제력이 없다.
                    "score": {"type": "number", "minimum": 0, "maximum": 100},
                    "reason": {"type": "string"},
                },
                "required": ["freelancer_id", "score", "reason"],
            },
        }
    },
    "required": ["candidates"],
}


class MatchingService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        gemini: GeminiClient,
        directory_repository: DirectoryRepository,
        ai_log_repository: AiAgentLogRepository | None = None,
    ):
        self._embedding_service = embedding_service
        self._gemini = gemini
        self._directory_repository = directory_repository
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def recommend(
        self,
        position_id: int,
        recruit_count: int,
        pool_multiplier: int,
        excluded_freelancer_ids: list[int] | None = None,
        budget_cap: int | None = None,
    ) -> MatchingResponse:
        position = await self._directory_repository.find_position_requirement(position_id)
        if position is None:
            raise AiException(AiErrorCode.NOT_FOUND, "포지션을 찾을 수 없습니다.")

        logger.info(
            "MATCHING_DEBUG python.recommend.start position_id=%s recruit_count=%s "
            "pool_multiplier=%s budget_cap=%s excluded_count=%s excluded_ids=%s "
            "job_category=%s job_role=%s skills=%s min_career_years=%s work_style=%s "
            "work_form=%s start_desired_date=%s start_negotiable=%s period_value=%s period_unit=%s",
            position_id,
            recruit_count,
            pool_multiplier,
            budget_cap,
            len(excluded_freelancer_ids or []),
            excluded_freelancer_ids or [],
            position.job_category,
            position.job_role,
            position.skills,
            position.min_career_years,
            position.work_style,
            position.work_form,
            position.start_desired_date,
            position.start_negotiable,
            position.period_value,
            position.period_unit,
        )

        # 하드필터: AI매칭 동의 + 매칭 일시중지 아님 + 계정 ACTIVE + 직군/직무 일치
        # + 요구 스킬 1개 이상 + 이미 노출된 후보 제외. **후보를 배제하는 건 여기뿐이다.**
        # 유사도로는 자르지 않는다 — 조건이 좋은데 자기소개가 짧아 유사도가 낮은 사람이
        # 잘려나가면 재설계 전 문제(스킬·연차가 1차 추림에 반영 안 됨)로 그대로 돌아간다.
        rows = await self._embedding_service.search_scored_candidates(
            position_id, position.job_category, position.job_role,
            position.skills, excluded_freelancer_ids,
        )
        logger.info(
            "MATCHING_DEBUG python.search.strict_result position_id=%s row_count=%s",
            position_id,
            len(rows),
        )
        for row in rows:
            logger.info(
                "MATCHING_DEBUG python.search.strict_row position_id=%s row=%s",
                position_id,
                _row_debug(row),
            )

        if not rows and position.skills:
            # 조건 완화 후 재검색(명세: "조건 충족 후보 0명 → 조건 완화 후 재검색").
            # **직군/직무는 그대로 둔다** — 백엔드 자리에 디자이너를 넣는 건 완화가 아니라
            # 오추천이다. AI매칭 동의·일시중지·계정 상태도 사용자 의사라 못 푼다. 그래서
            # 풀 수 있는 건 스킬 조건 하나뿐이다. 스킬 0개인 사람까지 넣는 건 분명 느슨하지만,
            # 대안이 "아무도 못 보여줌"이라 P09("조건에 맞는 후보가 부족하여 적합도가 낮은
            # 후보가 포함될 수 있습니다")가 이미 허용하는 상황이다. 어차피 조건점수 스킬 30점이
            # 0점이라 순위 맨 뒤로 간다.
            logger.info("후보 0명 → 스킬 조건을 풀어 재검색한다. position_id=%s", position_id)
            logger.info(
                "MATCHING_DEBUG python.search.relax position_id=%s reason=strict_empty "
                "relaxed_skills=[] original_skills=%s",
                position_id,
                position.skills,
            )
            rows = await self._embedding_service.search_scored_candidates(
                position_id, position.job_category, position.job_role,
                [], excluded_freelancer_ids,
            )
            logger.info(
                "MATCHING_DEBUG python.search.relaxed_result position_id=%s row_count=%s",
                position_id,
                len(rows),
            )
            for row in rows:
                logger.info(
                    "MATCHING_DEBUG python.search.relaxed_row position_id=%s row=%s",
                    position_id,
                    _row_debug(row),
                )

        if not rows:
            # 재검색도 0명 → 스프링이 MT_009로 받아 "재추천 안내"를 띄운다.
            logger.info(
                "MATCHING_DEBUG python.recommend.empty position_id=%s reason=no_rows_after_relax",
                position_id,
            )
            raise AiException(AiErrorCode.CANDIDATE_POOL_EMPTY)

        # 임베딩 25 + 조건점수 75 를 **합산한 뒤** 자른다. 순차로 하면(유사도로 먼저 N명을 뽑고
        # 그 안에서 조건 정렬) 누가 후보가 되는지를 유사도가 100% 정하게 되어 75의 비중이 사라진다.
        position_condition = _to_position_condition(position, budget_cap)
        candidate_conditions = [_to_candidate_condition(row) for row in rows]
        ranked = score_candidates(position_condition, candidate_conditions)
        ranked_by_id = {candidate.freelancer_id: candidate for candidate in ranked}
        for condition in candidate_conditions:
            breakdown = score_breakdown(position_condition, condition)
            scored = ranked_by_id[condition.freelancer_id]
            logger.info(
                "MATCHING_DEBUG python.score.detail position_id=%s freelancer_id=%s "
                "similarity=%s similarity_score=%s condition_score=%s total_score=%s "
                "skill=%s career=%s pay=%s work_style=%s work_form=%s start_date=%s "
                "period=%s raw_condition=%s normalized_condition=%s",
                position_id,
                condition.freelancer_id,
                _score_debug(scored.similarity),
                _score_debug(scored.similarity_score),
                _score_debug(scored.condition_score),
                _score_debug(scored.total_score),
                _score_debug(breakdown.skill),
                _score_debug(breakdown.career),
                _score_debug(breakdown.pay),
                _score_debug(breakdown.work_style),
                _score_debug(breakdown.work_form),
                _score_debug(breakdown.start_date),
                _score_debug(breakdown.period),
                _score_debug(breakdown.raw_condition),
                _score_debug(breakdown.normalized_condition),
            )
        top = ranked[: recruit_count * pool_multiplier]
        logger.info(
            "MATCHING_DEBUG python.rank.top position_id=%s hard_filter_count=%s top_count=%s top=%s",
            position_id,
            len(ranked),
            len(top),
            [
                {
                    "freelancer_id": candidate.freelancer_id,
                    "similarity": _score_debug(candidate.similarity),
                    "similarity_score": _score_debug(candidate.similarity_score),
                    "condition_score": _score_debug(candidate.condition_score),
                    "total_score": _score_debug(candidate.total_score),
                }
                for candidate in top
            ],
        )

        freelancer_ids = [candidate.freelancer_id for candidate in top]
        profiles = await self._directory_repository.find_freelancer_profiles(freelancer_ids)

        model = self._gemini.model_for(GeminiTask.MATCHING)
        prompt = self._build_prompt(position, freelancer_ids, profiles, recruit_count, budget_cap)
        logger.info(
            "MATCHING_DEBUG python.llm.request position_id=%s model=%s top_ids=%s "
            "profile_count=%s prompt_chars=%s prompt_preview=%s",
            position_id,
            model,
            freelancer_ids,
            len(profiles),
            len(prompt),
            prompt[:1000],
        )
        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.MATCHING, prompt, _RANKING_SCHEMA
            )
        except AiException as exc:
            logger.info(
                "MATCHING_DEBUG python.llm.failed position_id=%s model=%s error=%s",
                position_id,
                model,
                exc,
            )
            await self._record_call(position_id, model, prompt, None, None, str(exc))
            raise
        await self._record_call(position_id, model, prompt, raw, usage, None)
        logger.info(
            "MATCHING_DEBUG python.llm.response position_id=%s model=%s latency_ms=%s "
            "prompt_tokens=%s output_tokens=%s retry_count=%s raw=%s",
            position_id,
            model,
            usage.latency_ms,
            usage.prompt_tokens,
            usage.output_tokens,
            usage.retry_count,
            raw,
        )

        try:
            parsed = json.loads(raw)
            candidates = [RankedCandidate(**item) for item in parsed["candidates"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 후보 풀에 없는 ID 를 지어낼 수 있다. 풀 밖의 값은 버린다.
        # 남은 후보에는 1차 추림에서 계산한 코사인 유사도를 붙여 보낸다 — LLM 이 만든 값이 아니라
        # 우리가 계산한 값이라 여기서 채워야 한다(프롬프트에는 안 넣는다: 넣으면 LLM 이 원문을
        # 읽는 대신 그 숫자를 베낀다). 스프링이 matching_candidate.similarity 에 그대로 저장한다.
        similarity_by_id = {item.freelancer_id: item.similarity for item in top}
        filtered = [
            candidate.model_copy(update={"similarity": similarity_by_id[candidate.freelancer_id]})
            for candidate in candidates
            if candidate.freelancer_id in similarity_by_id
        ]
        logger.info(
            "MATCHING_DEBUG python.llm.filtered position_id=%s before_count=%s after_count=%s "
            "dropped_ids=%s candidates=%s",
            position_id,
            len(candidates),
            len(filtered),
            [
                candidate.freelancer_id
                for candidate in candidates
                if candidate.freelancer_id not in similarity_by_id
            ],
            [
                {
                    "freelancer_id": candidate.freelancer_id,
                    "score": candidate.score,
                    "similarity": candidate.similarity,
                    "reason": candidate.reason,
                }
                for candidate in filtered
            ],
        )

        # 스케일 검증은 **풀 밖 후보를 버린 뒤에** 한다. 지어낸 후보가 정상 점수(95)를 달고 오면
        # 최고점이 그 값으로 잡혀서, 정작 실제로 넘어갈 후보가 0.95 여도 검증을 통과해 버린다.
        _assert_score_scale(filtered)

        return MatchingResponse(position_id=position_id, model=model, candidates=filtered)

    async def _record_call(
        self,
        position_id: int,
        model: str,
        prompt: str,
        raw: str | None,
        usage: GeminiUsage | None,
        error_message: str | None,
    ) -> None:
        """추천 LLM 호출 1건 = ai_agent_log 1행.

        프롬프트/응답을 통째로 남긴다 — "이 추천이 왜 이렇게 나왔나"를 나중에 되짚으려면
        그때 실제로 무엇을 보냈는지가 있어야 한다(관리자 원본 로그 화면의 목적).
        """
        if self._ai_log_repository is None:
            return
        await self._ai_log_repository.record(
            AiCallRecord(
                agent_type=AgentType.MATCHER,
                status=LogStatus.FAILED if error_message else LogStatus.SUCCESS,
                ref_type=RefType.POSITION,
                ref_id=position_id,
                model=model,
                request_json={"prompt": prompt},
                response_json={"raw": raw} if raw is not None else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                latency_ms=usage.latency_ms if usage else None,
                retry_count=usage.retry_count if usage else 0,
                error_message=error_message,
            )
        )

    def _build_prompt(
        self,
        position: PositionRequirement,
        freelancer_ids: list[int],
        profiles: dict[int, FreelancerProfile],
        recruit_count: int,
        budget_cap: int | None = None,
    ) -> str:
        # 유사도도 조건점수도 프롬프트에 넣지 않는다. 넣으면 LLM 이 원문을 읽는 대신 그 숫자를
        # 베껴 쓴다 — 1차 추림 점수를 다시 확인하는 셈이라 새로 알아내는 게 없다. LLM 은 원문을
        # 직접 읽고 자기 판단으로 점수를 새로 매긴다(기존 원칙 유지).
        # 프로필이 없는 후보(레이스 컨디션 등)는 판단 근거가 없으니 통째로 뺀다.
        candidate_blocks = [
            self._describe_candidate(profiles[freelancer_id])
            for freelancer_id in freelancer_ids
            if freelancer_id in profiles
        ]

        return (
            "너는 프리랜서 매칭 심사자다. 아래 포지션 요구조건에 맞춰 후보를 적합한 순서로 정렬하고 "
            f"상위 {recruit_count}명을 골라라.\n"
            "직군/직무/AI매칭 동의는 이미 걸러진 후보들이다. 경력·스킬·자기소개·경력사항의 적합도를 "
            "우선 보고 판단해라.\n"
            "희망 급여·근무 방식·근무 형태·시작 가능일·희망 기간이 포지션 조건과 어긋나는 후보는 "
            "**후보에서 제외하지 말고 감점만 하고, 어긋난 내용을 reason에 함께 적어라.** "
            "조건이 맞는 후보가 부족하면 어긋난 후보도 노출돼야 하기 때문이다. "
            "감점 폭은 어긋난 정도에 비례해서 네가 판단해라 — 정해진 공식은 없다. "
            "'협의 가능'으로 표시된 항목은 어긋나도 감점하지 마라.\n"
            + _budget_guidance(budget_cap)
            + "score는 **0~100 사이의 숫자**로 매겨라(10점 만점이나 0~1 소수가 아니다). 스프링이 이 값을 "
            "그대로 적합도 점수로 저장하고 50점을 '적합도 낮음' 기준선으로 쓴다 — 요구조건을 대체로 "
            "충족하는 후보는 50점 이상, 핵심 요건이 어긋나 추천이 망설여지는 후보는 50점 미만으로 매겨라.\n"
            'reason은 짧은 근거 여러 개를 "|"로 이어붙인 하나의 문자열로 써라 '
            '(예: "백엔드 경력 5년 이상 충족|Spring 스킬 보유|희망 단가가 예산 상한을 웃돎"). '
            "문장으로 쓰지 말고 근거 단위로 끊어라.\n\n"
            f"{self._describe_position(position)}\n\n"
            "후보 목록:\n" + "\n\n".join(candidate_blocks)
        )

    @staticmethod
    def _describe_position(position: PositionRequirement) -> str:
        lines = [
            f"프로젝트: {position.project_title}",
            f"직군/직무: {position.job_category}/{position.job_role}",
            f"필요 경력: {position.min_career_years}년 이상",
            f"필요 스킬: {', '.join(position.skills) if position.skills else '명시 없음'}",
            f"근무 형태: {position.work_style}/{position.work_form}",
        ]
        if position.budget_amount is not None:
            lines.append(f"프로젝트 총예산: {int(position.budget_amount)}원 (기간 전체 총액, 부가세 별도)")
        if position.period_value is not None:
            lines.append(f"예상 기간: {position.period_value}{_period_label(position.period_unit)}")
        if position.start_desired_date is not None:
            start = f"시작 희망일: {position.start_desired_date}"
            lines.append(start + " (협의 가능)" if position.start_negotiable else start)
        for label, value in (
            ("현재 상황", position.current_situation),
            ("담당 업무", position.main_task),
            ("업무 범위", position.detail_scope),
            ("우대사항", position.extra_note),
        ):
            if value:
                lines.append(f"{label}: {value}")
        return "포지션 요구조건:\n" + "\n".join(lines)

    @staticmethod
    def _describe_candidate(profile: FreelancerProfile) -> str:
        lines = [
            f"- freelancer_id={profile.freelancer_id}",
            f"  직군/직무: {profile.job_category}/{profile.job_role}",
            f"  경력: {profile.career_years}년 이상"
            + ("(프리랜서 경험 있음)" if profile.has_freelance_exp else ""),
            f"  스킬: {', '.join(profile.skills) if profile.skills else '명시 없음'}",
        ]
        if profile.pay_amount is not None:
            lines.append(
                f"  희망 급여: {_pay_unit_label(profile.pay_unit)} {int(profile.pay_amount)}원"
            )
        if profile.work_style or profile.work_form:
            lines.append(f"  희망 근무: {profile.work_style}/{profile.work_form}")
        if profile.available_from is not None:
            available = f"  시작 가능일: {profile.available_from}"
            lines.append(available + " (협의 가능)" if profile.start_negotiable else available)
        if profile.period_value is not None:
            lines.append(f"  희망 기간: {profile.period_value}{_period_label(profile.period_unit)}")
        if profile.self_introduction:
            lines.append(f"  자기소개: {profile.self_introduction}")
        if profile.career_summary:
            lines.append(f"  경력사항: {profile.career_summary}")
        return "\n".join(lines)
