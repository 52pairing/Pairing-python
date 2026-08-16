from pydantic import BaseModel, Field


class ConditionContext(BaseModel):
    """협상 조건 1건의 현재 상태. 스프링(심판)이 넘겨준다."""

    condition_id: int = Field(..., ge=1, description="negotiation_condition.id")
    type: str = Field(..., description="AMOUNT/PERIOD/START_DATE/WORK_STYLE/WORK_FORM/SCOPE/OTHER")
    client_value: str | None = Field(default=None, description="클라 희망값(공개)")
    freelancer_value: str | None = Field(default=None, description="프리 희망값(공개)")
    # 마지노선(비공개)은 내부 호출이라 심판이 함께 넘긴다.
    # 대리인은 가드로만 쓰고 응답 텍스트에 노출하지 않는다.
    client_floor: str | None = Field(default=None, description="클라 마지노선(가드)")
    freelancer_floor: str | None = Field(default=None, description="프리 마지노선(가드)")
    # 직전 라운드에서 각 대리인이 마지막으로 낸 제시값(= 현재 협상 위치). 라운드 1 이거나 그 측
    # 제안이 아직 없으면 None → 그때는 희망값에서 시작한다. 값이 있으면 매 라운드 희망값으로
    # 리셋하지 말고 여기서 이어 협상한다(사람이 재지시로 좁혀 온 진행을 보존).
    client_last_value: str | None = Field(default=None, description="클라 대리인 직전 제시값(현재 위치)")
    freelancer_last_value: str | None = Field(
        default=None, description="프리 대리인 직전 제시값(현재 위치)"
    )
    # 마지노선 방향(MAX=상한/MIN=하한/CHOICE=허용값/NONE). 스프링(ConditionType)이 정한 값을 그대로
    # 받는다 — 프롬프트가 타입으로 다시 추론하지 않기 위해서다(방향 단일 진실 원본). None 이면(구 백엔드)
    # 기존 가정(클라=MAX, 프리=MIN)으로 폴백한다.
    client_floor_direction: str | None = Field(default=None, description="클라 마지노선 방향")
    freelancer_floor_direction: str | None = Field(default=None, description="프리 마지노선 방향")
    # 선택형(enum) 쟁점은 후보를 안 주면 LLM 이 없는 값을 지어낸다(예: WORK_STYLE 에 HYBRID).
    allowed_values: list[str] | None = Field(
        default=None,
        description="선택형 쟁점의 허용값 목록. 있으면 proposed_value 는 반드시 이 안에서 고른다",
    )
    value_format: str | None = Field(
        default=None, description="proposed_value 표기 형식(예: '숫자만', '<숫자> MONTH', 'YYYY-MM-DD')"
    )


class ProposeRequest(BaseModel):
    negotiation_id: int = Field(..., ge=1)
    round: int = Field(..., ge=1, description="현재 라운드")
    budget_cap: int | None = Field(default=None, description="예산 상한(원). AMOUNT 가드")
    conditions: list[ConditionContext] = Field(..., min_length=1)


class AgentMessage(BaseModel):
    """A2A 대화 로그 한 줄. 두 대리인이 번갈아 제안·역제안·수락한다."""

    sender: str = Field(..., description="CLIENT_AGENT | FREELANCER_AGENT")
    condition_id: int = Field(..., description="이 발언이 다루는 쟁점. 요청 conditions 중 하나")
    kind: str = Field(..., description="PROPOSAL(제안) | COUNTER(역제안) | ACCEPT(수락)")
    proposed_value: str = Field(..., description="이 발언이 제시/수락하는 값(금액은 숫자 문자열)")
    content: str = Field(..., description="사람에게 보일 한국어 한 문장")
    reason: str = Field(..., description="한국어 한 문장 근거. 모든 발언에 필수")


class ConditionOutcome(BaseModel):
    """쟁점별 대리인 협상 결과. agreed=True 면 양 대리인이 이 값에 합의(자동 락 후보)."""

    condition_id: int
    proposed_value: str = Field(..., description="대리인들이 도달한 최종 제안값")
    agreed: bool = Field(..., description="두 대리인이 이 값에 합의했는지(사람 승인 전 단계)")


class ProposeResponse(BaseModel):
    negotiation_id: int
    model: str = Field(..., description="제안에 쓴 LLM 모델명. 결과 재현·감사용")
    messages: list[AgentMessage] = Field(..., description="A2A 대화 로그(시간순)")
    outcomes: list[ConditionOutcome] = Field(..., description="쟁점별 최종 제안·합의 여부")
