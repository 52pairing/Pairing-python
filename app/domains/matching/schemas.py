from pydantic import BaseModel, Field


class MatchingRequest(BaseModel):
    position_id: int = Field(..., ge=1, description="project_position.id")
    recruit_count: int = Field(..., ge=1, le=50, description="모집 인원")
    # 1차 후보 풀은 모집 인원 x 3 이다. (요구사항 기준값)
    pool_multiplier: int = Field(default=3, ge=1, le=10)


class RankedCandidate(BaseModel):
    freelancer_id: int
    score: float = Field(..., description="0~1. 벡터 유사도와 LLM 판단을 합친 최종 점수")
    reason: str = Field(..., description="추천 사유. 클라이언트 화면에 노출된다")


class MatchingResponse(BaseModel):
    position_id: int
    model: str = Field(..., description="재랭킹에 쓴 LLM 모델명. 결과 재현에 필요하다")
    candidates: list[RankedCandidate]
