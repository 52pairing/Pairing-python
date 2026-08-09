from pydantic import BaseModel, Field


class DraftTextRequest(BaseModel):
    """계약서의 자유 텍스트 칸만 넘긴다. 금액·기간·당사자는 스프링이 직접 채운다."""

    contract_id: int = Field(..., ge=1)
    main_task: str = Field(..., min_length=1, max_length=1500, description="주요 담당 업무 원문")
    detail_scope: str | None = Field(default=None, max_length=1500, description="세부 업무 범위 원문")
    agreed_notes: list[str] = Field(
        default_factory=list,
        description="협상 SCOPE/OTHER 합의값. 특약사항 후보. 비어 있으면 특약 없음으로 확정된다",
    )


class DraftTextResponse(BaseModel):
    contract_id: int
    model: str = Field(..., description="사용한 LLM 모델명. 결과 재현·감사용")
    main_task_summary: str = Field(..., description="제2조 담당 업무. 한 줄")
    detail_scope_summary: str = Field(..., description="제2조 세부 업무 범위. 1~2줄. 원문이 없으면 빈 문자열")
    special_terms: str = Field(..., description="제15조 특약사항. 없으면 '별도의 특약사항 없음'")