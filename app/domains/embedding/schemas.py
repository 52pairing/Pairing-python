"""요청/응답 모델. 스프링의 request/response record 와 같은 자리다."""

from pydantic import BaseModel, Field


class FreelancerEmbeddingRequest(BaseModel):
    freelancer_id: int = Field(..., ge=1, description="freelancer_profile.id")
    text: str = Field(..., min_length=1, max_length=20000, description="이력서/조건을 합친 원문")


class PositionEmbeddingRequest(BaseModel):
    position_id: int = Field(..., ge=1, description="project_position.id")
    text: str = Field(..., min_length=1, max_length=20000)


class EmbeddingResponse(BaseModel):
    target_id: int
    model: str
    dimension: int
    skipped: bool = Field(default=False, description="내용이 그대로라 재생성하지 않았으면 true")


class SimilarFreelancer(BaseModel):
    freelancer_id: int
    score: float = Field(..., description="코사인 유사도 0~1. 높을수록 가깝다")


class SimilaritySearchResponse(BaseModel):
    position_id: int
    candidates: list[SimilarFreelancer]
