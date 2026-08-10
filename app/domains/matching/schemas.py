from pydantic import BaseModel, Field


class MatchingRequest(BaseModel):
    position_id: int = Field(..., ge=1, description="project_position.id")
    recruit_count: int = Field(..., ge=1, le=50, description="모집 인원")
    # 1차 후보 풀은 모집 인원 x 3 이다. (요구사항 기준값)
    pool_multiplier: int = Field(default=3, ge=1, le=10)
    excluded_freelancer_ids: list[int] = Field(
        default_factory=list,
        description="같은 프로젝트에서 이미 후보로 노출됐던 freelancer_id 목록. 벡터 검색 전에 제외한다",
    )


class RankedCandidate(BaseModel):
    freelancer_id: int
    # 스프링의 matching_candidate.base_score(NUMERIC(5,2), 0~100)에 그대로 저장되고,
    # 50점 미만이 lowScoreWarned(적합도 낮음 경고) 기준이다. 범위를 바꾸면 양쪽을 함께 바꿔야 한다.
    # ge/le로 범위를 강제한다 — 프롬프트 지시만으로는 LLM이 다른 스케일로 답하는 걸 못 막는다.
    score: float = Field(..., ge=0, le=100, description="0~100. LLM이 판단한 원점수(등급 가중치 반영 전)")
    reason: str = Field(
        ..., description='추천 사유. "|"로 이어붙인 문자열 — 스프링이 이 구분자로 다시 나눠 노출한다'
    )


class MatchingResponse(BaseModel):
    position_id: int
    model: str = Field(..., description="재랭킹에 쓴 LLM 모델명. 결과 재현에 필요하다")
    candidates: list[RankedCandidate]
