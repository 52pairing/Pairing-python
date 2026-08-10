"""매칭 추천.

두 단계로 나눈다.
  1) 벡터 검색으로 후보 풀을 좁힌다 (모집 인원 x 3)
  2) LLM 이 그 풀만 보고 순위와 사유를 만든다

전체 프리랜서를 LLM 에 넣지 않는 이유는 비용·지연·컨텍스트 한계 셋 다다.
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
from app.domains.embedding.schemas import SimilaritySearchResponse
from app.domains.embedding.service import EmbeddingService
from app.domains.matching.repository import (
    DirectoryRepository,
    FreelancerProfile,
    PositionRequirement,
)
from app.domains.matching.schemas import MatchingResponse, RankedCandidate

logger = logging.getLogger(__name__)

_PERIOD_LABELS = {"MONTH": "개월", "WEEK": "주"}
_PAY_UNIT_LABELS = {"HOURLY": "시급", "DAILY": "일급", "MONTHLY": "월급"}


def _period_label(unit: str | None) -> str:
    return _PERIOD_LABELS.get(unit, unit or "")


def _pay_unit_label(unit: str | None) -> str:
    return _PAY_UNIT_LABELS.get(unit, unit or "")


_RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "freelancer_id": {"type": "integer"},
                    "score": {"type": "number"},
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
    ) -> MatchingResponse:
        position = await self._directory_repository.find_position_requirement(position_id)
        if position is None:
            raise AiException(AiErrorCode.NOT_FOUND, "포지션을 찾을 수 없습니다.")

        # 하드필터(Stage B): AI매칭 동의 + 매칭 일시중지 아님 + 직군/직무 일치. 벡터 검색 전에 미리
        # 걸러서 후보 풀 자체를 줄인다. 일정/근무조건/단가는 여기서 안 본다 — 후보를 배제하지 않고
        # Stage E(_build_prompt)에서 감점 요인으로만 반영한다.
        pool_size = recruit_count * pool_multiplier
        pool = await self._embedding_service.search_candidates(
            position_id, pool_size, position.job_category, position.job_role, excluded_freelancer_ids
        )

        if not pool.candidates:
            raise AiException(AiErrorCode.CANDIDATE_POOL_EMPTY)

        freelancer_ids = [candidate.freelancer_id for candidate in pool.candidates]
        profiles = await self._directory_repository.find_freelancer_profiles(freelancer_ids)

        model = self._gemini.model_for(GeminiTask.MATCHING)
        prompt = self._build_prompt(position, pool, profiles, recruit_count)
        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.MATCHING, prompt, _RANKING_SCHEMA
            )
        except AiException as exc:
            await self._record_call(position_id, model, prompt, None, None, str(exc))
            raise
        await self._record_call(position_id, model, prompt, raw, usage, None)

        try:
            parsed = json.loads(raw)
            candidates = [RankedCandidate(**item) for item in parsed["candidates"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 후보 풀에 없는 ID 를 지어낼 수 있다. 풀 밖의 값은 버린다.
        allowed = {candidate.freelancer_id for candidate in pool.candidates}
        filtered = [candidate for candidate in candidates if candidate.freelancer_id in allowed]

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
        pool: SimilaritySearchResponse,
        profiles: dict[int, FreelancerProfile],
        recruit_count: int,
    ) -> str:
        # 벡터 유사도는 풀을 좁히는 데만 쓴다. 프로필이 없는 후보(레이스 컨디션 등)는 판단 근거가
        # 없으니 통째로 뺀다 — 어차피 최종 필터(allowed)에서도 걸러지지 않는다.
        candidate_blocks = [
            self._describe_candidate(profiles[candidate.freelancer_id])
            for candidate in pool.candidates
            if candidate.freelancer_id in profiles
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
            "단가를 볼 때 주의: 포지션의 총예산은 프로젝트 전체 인원·전체 기간을 합친 금액이고 "
            "후보의 희망 급여는 1인 단위 금액이다. 두 숫자를 그대로 비교하지 말고 기간과 인원을 "
            "감안해서 판단해라. 총예산은 참고치이고 확정된 1인 상한이 아니므로, 조금 넘는 정도로는 "
            "크게 감점하지 마라.\n"
            "score는 **0~100 사이의 숫자**로 매겨라(10점 만점이나 0~1 소수가 아니다). 스프링이 이 값을 "
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
