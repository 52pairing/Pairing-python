from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """스프링이 세션/사용량을 이미 확인한 뒤 순수 질의만 넘긴다."""

    question: str = Field(..., min_length=1, max_length=500, description="사용자 질문")


class AskResponse(BaseModel):
    answer: str = Field(..., description="답변(한국어)")
    model: str = Field(..., description="답변에 쓴 LLM 모델명. 결과 재현·감사용")
