"""A2A 협상 제안 생성.

백엔드(스프링)가 심판(라운드/락/타결)을 맡고, 여기서는 **두 대리인**(클라이언트 대리 /
프리랜서 대리)이 서로 제안·역제안·수락을 주고받는 A2A 대화를 생성한다.

- 클라 대리: 예산 상한·클라 마지노선 안에서 클라에게 유리하게.
- 프리 대리: 프리 마지노선 안에서 프리에게 유리하게.
- 마지노선은 가드로만 쓰고 발언 텍스트에 그대로 노출하지 않는다.
- 모든 발언에 근거(reason)를 붙인다. (요구사항 P13)

한 번 호출로 대화 전체(트랜스크립트)와 쟁점별 최종 결과를 돌려준다. 스프링이 이를 라운드로
기록하고, agreed 조건은 자동 락, 나머지는 사람 승인/재지시로 넘긴다.
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
from app.domains.negotiation.schemas import (
    AgentMessage,
    ConditionOutcome,
    ProposeRequest,
    ProposeResponse,
)

logger = logging.getLogger(__name__)

# 선택형 값의 **문장용** 한글 표기.
#
# 말풍선에 코드가 그대로 나가고 있었다 — "PART_TIME 형태의 근무를 희망합니다",
# "상주 근무는 불가능하여 REMOTE 방식을 유지해야 합니다". 사람에게 보이는 문장이라 코드가 아니라
# 라벨이어야 한다. 금액은 규칙 6-1 로 이미 처리했는데 선택형만 빠져 있었다.
#
# **표기는 스프링/관리자(ConditionValueLabels)와 같은 것을 쓴다.** 같은 코드가 화면마다 다르게
# 번역되면 사용자가 다른 조건으로 읽는다. 특히 ANY 는 근무 방식에서 "혼합", 근무 형태에서
# "모두 가능"이라 **쟁점 타입별로 갈라야 한다** — 하나의 표로 합치면 한쪽이 틀린다.
#
# proposed_value 에는 절대 쓰지 않는다(규칙 5). 값 필드에 라벨이 들어가면 스프링이 해석하지 못해
# 조건이 미합의로 강등된다.
_VALUE_LABELS = {
    "WORK_STYLE": {"REMOTE": "재택", "ONSITE": "상주", "ANY": "혼합"},
    "WORK_FORM": {"FULL_TIME": "풀타임", "PART_TIME": "파트타임", "ANY": "모두 가능"},
}

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string", "enum": ["CLIENT_AGENT", "FREELANCER_AGENT"]},
                    "condition_id": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["PROPOSAL", "COUNTER", "ACCEPT"]},
                    "proposed_value": {"type": "string"},
                    "content": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["sender", "condition_id", "kind", "proposed_value", "content", "reason"],
            },
        },
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition_id": {"type": "integer"},
                    "proposed_value": {"type": "string"},
                    "agreed": {"type": "boolean"},
                },
                "required": ["condition_id", "proposed_value", "agreed"],
            },
        },
    },
    "required": ["messages", "outcomes"],
}


class NegotiationService:
    def __init__(
        self,
        gemini: GeminiClient,
        ai_log_repository: AiAgentLogRepository | None = None,
    ):
        self._gemini = gemini
        # 없으면 로그만 안 남기고 그대로 동작한다(테스트에서 굳이 안 넣어도 되게).
        self._ai_log_repository = ai_log_repository

    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        model = self._gemini.model_for(GeminiTask.NEGOTIATION)
        prompt = self._build_prompt(request)
        try:
            raw, usage = await self._gemini.generate_json_with_usage(
                GeminiTask.NEGOTIATION, prompt, _RESPONSE_SCHEMA
            )
        except AiException as exc:
            await self._record_call(request, model, prompt, None, None, str(exc))
            raise

        try:
            messages, outcomes = self._parse(raw, request)
        except AiException as exc:
            # 파싱·검증 실패는 호출 실패보다 원인 찾기가 어렵다. 응답 원문을 남겨야
            # "LLM 이 무엇을 돌려줘서 못 썼는지"를 나중에 되짚을 수 있다.
            await self._record_call(request, model, prompt, raw, usage, str(exc))
            raise

        await self._record_call(request, model, prompt, raw, usage, None)
        return ProposeResponse(
            negotiation_id=request.negotiation_id, model=model, messages=messages, outcomes=outcomes
        )

    def _parse(
        self, raw: str, request: ProposeRequest
    ) -> tuple[list[AgentMessage], list[ConditionOutcome]]:
        try:
            parsed = json.loads(raw)
            messages = [AgentMessage(**m) for m in parsed["messages"]]
            outcomes = [ConditionOutcome(**o) for o in parsed["outcomes"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("A2A 협상 응답 파싱 실패: %s", exc)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID) from exc

        # LLM 이 요청에 없던 condition_id 를 지어낼 수 있다. 요청한 조건만 남긴다.
        allowed = {c.condition_id for c in request.conditions}
        messages = [m for m in messages if m.condition_id in allowed]
        outcomes = [o for o in outcomes if o.condition_id in allowed]

        # 모든 쟁점에 결과가 있어야 스프링이 라운드를 확정할 수 있다.
        covered = {o.condition_id for o in outcomes}
        missing = allowed - covered
        if missing:
            logger.warning("A2A 결과 누락 쟁점: %s", missing)
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "일부 쟁점 결과가 비어 있습니다.")
        if not messages:
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID, "대화가 비어 있습니다.")

        return messages, outcomes

    async def _record_call(
        self,
        request: ProposeRequest,
        model: str,
        prompt: str,
        raw: str | None,
        usage: GeminiUsage | None,
        error_message: str | None,
    ) -> None:
        """A2A 제안 LLM 호출 1건 = ai_agent_log 1행.

        협상은 stub 폴백이 있어서 파이썬이 실패해도 스프링 쪽은 계속 돈다. 그래서 실패가
        화면에 드러나지 않는다 — 여기 남기지 않으면 "왜 stub 대화가 나왔는지"를 알 수 없다.
        """
        if self._ai_log_repository is None:
            return
        await self._ai_log_repository.record(
            AiCallRecord(
                agent_type=AgentType.NEGOTIATOR,
                status=LogStatus.FAILED if error_message else LogStatus.SUCCESS,
                ref_type=RefType.NEGOTIATION,
                ref_id=request.negotiation_id,
                model=model,
                request_json={"prompt": prompt, "round": request.round},
                response_json={"raw": raw} if raw is not None else None,
                prompt_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                latency_ms=usage.latency_ms if usage else None,
                retry_count=usage.retry_count if usage else 0,
                error_message=error_message,
            )
        )

    def _build_prompt(self, request: ProposeRequest) -> str:
        lines = []
        for c in request.conditions:
            # 마지노선은 방향을 함께 적는다. "클라마지노선=3300000" 만 주면 LLM 이 그게 상한인지
            # 하한인지 몰라 그 값을 넘겨 합의해 버린다(실제로 프리 하한 480만인데 330만을 수락했다).
            line = (
                f"- condition_id={c.condition_id}, 쟁점={c.type}, "
                f"클라희망={c.client_value}, 프리희망={c.freelancer_value}, "
                f"클라상한(이_값을_초과하면_클라가_거절)={c.client_floor}, "
                f"프리하한(이_값에_못_미치면_프리가_거절)={c.freelancer_floor}"
            )
            # 직전 라운드 각 측 마지막 제시값(= 현재 협상 위치). 있으면 오프닝을 희망값이 아니라
            # 여기서 잡는다(규칙 1-2). 사람이 재지시로 좁혀 온 진행을 매 라운드 리셋하지 않기 위해서다.
            if c.client_last_value is not None or c.freelancer_last_value is not None:
                line += (
                    f", 클라직전제시={c.client_last_value}, 프리직전제시={c.freelancer_last_value}"
                )
            # 선택형은 후보를, 형식이 정해진 쟁점은 표기법을 함께 준다(없는 값·형식 이탈 방지).
            if c.allowed_values:
                line += f", 허용값={'|'.join(c.allowed_values)}"
            if c.value_format:
                line += f", 값형식={c.value_format}"
            # 문장에 쓸 한글 표기를 쟁점마다 같이 준다. 규칙으로만 "한글로 써라" 하면 LLM 이
            # 제멋대로 번역해(REMOTE → "원격") 화면 용어와 어긋난다.
            labels = _VALUE_LABELS.get(c.type)
            if labels:
                pairs = [f"{code}→{labels[code]}" for code in c.allowed_values or []
                         if code in labels]
                if pairs:
                    line += f", 문장표기={'|'.join(pairs)}"
            lines.append(line)
        budget = f"{request.budget_cap}" if request.budget_cap is not None else "미지정"
        # 선공을 라운드마다 바꾼다. 지정하지 않으면 LLM 이 먼저 등장한 CLIENT_AGENT 를 개시자로
        # 잡아, 매 라운드 클라가 앵커를 쥔다. 협상에서 첫 제안은 기준점이라 그 자체로 유불리다.
        opener = "FREELANCER_AGENT" if request.round % 2 == 1 else "CLIENT_AGENT"
        return (
            "너는 프리랜서-클라이언트 채용 조건 협상을 **두 AI 대리인의 대화**로 시뮬레이션한다.\n"
            "- CLIENT_AGENT(클라이언트 대리): 예산 상한과 클라 상한 안에서 클라에게 유리하게 협상한다.\n"
            "- FREELANCER_AGENT(프리랜서 대리): 프리 하한 위에서 프리에게 유리하게 협상한다.\n"
            "\n"
            "**가장 중요한 제약 — 마지노선은 넘을 수 없는 선이다.**\n"
            "- 클라상한을 초과하는 값에 CLIENT_AGENT 가 동의해서는 안 된다.\n"
            "- 프리하한에 못 미치는 값에 FREELANCER_AGENT 가 동의해서는 안 된다.\n"
            "- 합의를 성사시키는 것보다 이 선을 지키는 것이 우선이다. "
            "선을 지키면서 합의할 수 없으면 **합의하지 말고 agreed=false 로 남긴다.** "
            "결렬은 실패가 아니라 정상적인 결과다.\n"
            "- 클라상한 < 프리하한 이면 접점이 없다. 이때는 절대 합의하지 말고 agreed=false 로 남긴다.\n"
            "\n"
            # 이 블록은 원래 규칙 1-3 이었다. 규칙 목록 안에 두었더니 지켜지지 않아 앞으로 끌어올렸다
            # (2026-08-12 실측: 규칙이 들어간 배포 뒤에도 프리 대리인이 550만 → 상대 375만을
            # 역제안 없이 그대로 수락했다). 함께 고친 것은 §후속 3가지 — 아래 주석 참고.
            "**두 번째 제약 — 한 번의 발언으로 다 양보하지 않는다.**\n"
            "(마지노선 규칙이 우선한다. 둘이 부딪히면 마지노선을 따른다.)\n"
            "- **상대가 처음 제시한 값을 그대로 수락하지 않는다.** "
            "그것이 제안(PROPOSAL)이든 역제안(COUNTER)이든 똑같이 적용된다.\n"
            "- **자기 희망값에서 상대 값까지 한 번에 이동하지 않는다.** "
            "최소 한 번은 두 값 사이의 중간값을 역제안(COUNTER)한 뒤 절충한다. "
            "550만을 부른 쪽이 상대의 375만을 곧바로 받으면 양보할 수 있었던 폭을 통째로 내주는 것이다.\n"
            "- **예외는 다음 두 가지뿐이다.** "
            "① 상대가 제시한 값이 이미 자기 희망값과 같거나 그보다 유리하다. "
            "② 클라상한과 프리하한이 정확히 같아 고를 수 있는 값이 하나뿐이다. "
            "**그 밖에는 접점이 좁더라도 중간값을 한 번 제시한다** — "
            "'여지가 없어 보인다'는 판단으로 이 규칙을 건너뛰지 않는다.\n"
            "\n"
            # 실측(2026-08-16): 재지시로 라운드가 진행돼도 대리인이 매 라운드 희망값으로 리셋해 다시
            # 열었다(금액 590만 재등장). 날짜에선 10/1→10/21 처럼 상대에게서 멀어지고 "선행 프로젝트"
            # 같은 없는 사실까지 지어냈다. '직전제시' 오프닝(규칙 1-2)과 함께, 뒤로 가기·작화를 막는다.
            # 첫 라운드(직전제시 없음)에는 적용하지 않는다 — 그때는 희망값에서 정상적으로 연다.
            "**세 번째 제약 — 뒤로 가지 않는다. 없는 사실을 지어내지 않는다.**\n"
            "(마지노선·양보폭 규칙과 함께 지킨다. '직전제시'가 주어진 라운드에만 적용한다.)\n"
            "- **직전제시가 있으면, 그보다 상대에게서 더 먼 값(= 뒤로 가는 값)을 새로 내지 않는다.** "
            "협상은 상대 쪽으로 다가가거나 그 자리를 지킬 뿐, 되돌아가지 않는다. "
            "(예: 프리가 지난 라운드 430만까지 왔는데 이번에 590만을 다시 부르면 안 된다. "
            "날짜·기간·금액 모두 같다.)\n"
            "- **주어진 값(희망·마지노선·직전제시) 밖의 사실·제약을 지어내지 않는다.** "
            "'선행 프로젝트 때문에 늦어진다' 같은 근거를 만들어 자기 값을 뒤로 물리지 않는다. "
            "reason 은 자기가 앞서 한 발언과 모순되지 않아야 한다.\n"
            "\n"
            "규칙:\n"
            # "2~4개 발언으로 간결하게"가 위 블록과 정면으로 부딪혔다. 중간 역제안을 한 번 넣으면
            # 4발언인데 '간결하게'가 3발언으로 끌어내린다. 하한을 3으로 올려 여지를 만든다.
            "1) 각 쟁점마다 두 대리인이 번갈아 제안(PROPOSAL)·역제안(COUNTER)·수락(ACCEPT)을 주고받되, "
            "쟁점당 3~5개 발언으로 수렴시킨다. "
            "접점이 없어 곧바로 결렬되는 쟁점만 2개 발언으로 끝낼 수 있다.\n"
            "1-1) **각 쟁점의 첫 발언은 아래에 지정된 '이번 라운드 선공' 대리인이 한다.**\n"
            "1-2) **각 대리인의 이번 라운드 오프닝은 '직전제시'가 주어졌으면 그 값에서 이어간다. "
            "직전제시가 없으면(첫 라운드) 자기 측 희망값(클라희망/프리희망)에서 시작한다.** "
            "직전제시는 지난 라운드까지 좁혀 온 현재 위치다 — 매 라운드 희망값으로 되돌아가면 "
            "사람이 재지시로 양보해 온 진행이 통째로 사라진다. "
            "마지노선은 물러설 수 없는 방어선이지 시작점이 아니다. "
            "첫 발언에 자기 마지노선을 그대로 부르면 남은 협상 여지가 사라지고, "
            "대화 로그는 상대에게도 그대로 보이므로 내 선을 알려주는 것과 같다.\n"
            "1-3) 양보 폭에 대한 규칙은 위의 **'두 번째 제약'** 블록을 따른다.\n"
            "2) 금액(AMOUNT)은 프리하한 이상 클라상한 이하에서, 예산 상한을 절대 넘지 않게 합의한다. "
            "그 구간이 비어 있으면 합의하지 않는다.\n"
            "3) 근무형태/방식 등 선택형은 양측 수용 가능한 값으로 합의한다.\n"
            "3-1) 쟁점에 '허용값'이 주어지면 proposed_value 는 **반드시 그 목록 안의 값 그대로**만 쓴다. "
            "목록에 없는 값을 새로 만들지 않는다"
            "(예: 허용값이 REMOTE|ONSITE|ANY 인데 HYBRID 를 쓰면 안 된다). "
            "양측이 겹치는 허용값을 못 찾으면 억지로 만들지 말고 agreed=false 로 남긴다.\n"
            "3-2) 쟁점에 '값형식'이 주어지면 proposed_value 는 그 형식을 정확히 지킨다"
            "(예: 값형식=<숫자> MONTH 이면 '3' 이 아니라 '3 MONTH').\n"
            "4) 두 마지노선이 겹쳐 합의 가능한 쟁점은 "
            "outcomes.agreed=true 와 최종 proposed_value 로 마무리한다. "
            "겹치지 않아 합의 불가한 쟁점은 마지막 역제안 값을 "
            "proposed_value 로 두고 agreed=false 로 남긴다.\n"
            "4-1) outcomes 의 proposed_value 는 "
            "**그 쟁점 대화의 마지막 발언 proposed_value 와 반드시 같아야 한다.** "
            "대화는 330만원으로 끝났는데 outcomes 에 480만원을 적는 식으로 어긋나면 안 된다. "
            "사람이 대화를 읽고 이해한 값이 곧 계약 값이 된다.\n"
            "5) proposed_value 는 설명 없이 값만 담는다(금액은 단위·콤마 없는 숫자 문자열). "
            "'값형식'·'허용값'이 주어진 쟁점은 그 지시가 우선한다.\n"
            "6) content 는 사람에게 보일 한국어 한 문장, reason 은 한국어 한 문장 근거. 모든 발언에 필수.\n"
            "6-1) content·reason 에서 금액을 말할 때는 만원 단위로 쓴다"
            "(3300000 → \"330만 원\"). **이 규칙은 문장에만 적용한다.** "
            "proposed_value 는 5) 의 형식(단위·콤마 없는 숫자)을 그대로 지킨다 — "
            "값 필드에 만원 표기가 들어오면 스프링이 해석하지 못해 조건이 미합의로 강등된다.\n"
            "6-2) **content·reason 에 코드값을 그대로 쓰지 않는다.** "
            "쟁점에 '문장표기'가 주어지면 문장에서는 반드시 그 한글 표기를 쓴다"
            "(예: \"REMOTE 방식을 유지해야 합니다\" 가 아니라 \"재택 근무를 유지해야 합니다\"). "
            "'문장표기'에 없는 값은 임의로 번역하지 말고 원문을 쓴다.\n"
            "6-3) 기간도 문장에서는 한국어로 쓴다(\"4 MONTH\" → \"4개월\", \"2 WEEK\" → \"2주\"). "
            "**6-1~6-3 은 전부 문장에만 적용한다** — proposed_value 는 5) 의 형식을 그대로 지킨다.\n"
            "7) 마지노선 숫자를 발언 텍스트에 그대로 노출하지 않는다(가드로만 사용).\n"
            "8) messages 는 모든 쟁점의 대화를 시간순으로, outcomes 는 쟁점별 최종 결과를 담는다.\n"
            f"\n예산 상한(원): {budget}\n"
            f"현재 라운드: {request.round}\n"
            f"이번 라운드 선공: {opener}\n"
            "쟁점 목록:\n" + "\n".join(lines)
        )
