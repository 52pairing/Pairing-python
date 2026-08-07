from pydantic import BaseModel, Field


class ConditionContext(BaseModel):
    """협상 조건 1건의 현재 상태. 스프링(심판)이 넘겨준다."""

    condition_id: int = Field(..., ge=1, description="negotiation_condition.id")
    type: str = Field(..., description="AMOUNT/PERIOD/START_DATE/WORK_STYLE/WORK_FORM/SCOPE/OTHER")
    client_value: str | None = Field(default=None, description="클라 희망값(공개)")
    freelancer_value: str | None = Field(default=None, description="프리 희망값(공개)")
    # 마지노선(비공개)은 내부 호출이라 심판이 함께 넘긴다. AI 는 중재를 위해 참고만 하고 응답에 담지 않는다.
    client_floor: str | None = Field(default=None, description="클라 마지노선(가드)")
    freelancer_floor: str | None = Field(default=None, description="프리 마지노선(가드)")


class ProposeRequest(BaseModel):
    negotiation_id: int = Field(..., ge=1)
    round: int = Field(..., ge=1, description="현재 라운드")
    budget_cap: int | None = Field(default=None, description="예산 상한(원). AMOUNT 가드")
    conditions: list[ConditionContext] = Field(..., min_length=1)


class ConditionProposal(BaseModel):
    condition_id: int
    proposed_value: str = Field(..., description="제안값(금액은 숫자 문자열, 그 외 enum/날짜)")
    content: str = Field(..., description="사람에게 보일 제안 문장(한국어)")
    reason: str = Field(..., description="제안 근거(한국어). 모든 제안엔 근거가 붙는다")


class ProposeResponse(BaseModel):
    negotiation_id: int
    model: str = Field(..., description="제안에 쓴 LLM 모델명. 결과 재현·감사용")
    proposals: list[ConditionProposal]
