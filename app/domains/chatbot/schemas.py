from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """스프링이 세션/사용량을 이미 확인한 뒤 순수 질의만 넘긴다."""

    question: str = Field(..., min_length=1, max_length=500, description="사용자 질문")


class AskResponse(BaseModel):
    answer: str = Field(..., description="답변(한국어)")
    intent: str = Field(
        "NONE",
        description=(
            "답변과 이어지는 화면 코드. 닫힌 목록 중 하나이며 해당 없으면 NONE. "
            "실제 이동 경로는 스프링이 이 코드로 결정한다 — LLM 은 URL 을 만들지 않는다."
        ),
    )
    model: str = Field(..., description="답변에 쓴 LLM 모델명. 결과 재현·감사용")
    out_of_scope: bool = Field(
        False,
        description="페어링과 무관한 질문이어서 답하지 않았다는 표시. 관찰용이며 차감 판단은 charge_quota 가 한다.",
    )
    charge_quota: bool = Field(
        True,
        description=(
            "하루 사용량을 깎아야 하는지. 스프링이 이 값만 보고 판단한다. "
            "범위 밖 질문(답을 못 줌)과 단순 인사(질문이 아님)는 false 다 — "
            "답을 받지 못했는데 횟수만 빠지면 오타 한 번에 1회가 날아간다."
        ),
    )


class KnowledgeReindexResponse(BaseModel):
    """관련성 판정용 지식 청크 재색인 결과."""

    total: int = Field(..., description="전체 청크 수")
    embedded: int = Field(..., description="이번에 임베딩한 수(신규·문구 변경)")
    skipped: int = Field(..., description="문구가 그대로여서 건너뛴 수")
