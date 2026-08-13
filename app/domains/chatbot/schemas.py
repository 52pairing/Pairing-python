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
        description=(
            "페어링과 무관한 질문이어서 답하지 않았다는 표시. "
            "스프링은 이 값이 true 면 사용량을 차감하지 않고 고정 안내 문구로 응답한다."
        ),
    )


class KnowledgeReindexResponse(BaseModel):
    """관련성 판정용 지식 청크 재색인 결과."""

    total: int = Field(..., description="전체 청크 수")
    embedded: int = Field(..., description="이번에 임베딩한 수(신규·문구 변경)")
    skipped: int = Field(..., description="문구가 그대로여서 건너뛴 수")
