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
    # AI 서버가 스스로 못 구하는 값이라 스프링이 넘겨준다 — 순예산을 알려면 수수료율이 필요하고,
    # 수수료율은 클라이언트 등급(account 도메인)에 걸려 있다. 여기서 계산하려 들면 등급 테이블까지
    # 읽어야 하고, 스프링(BudgetCapCalculator)과 두 벌이 되어 조용히 어긋난다.
    # None이면 단가 비교를 생략한다(옛 스프링 배포와 섞여 도는 동안).
    budget_cap: int | None = Field(
        default=None, ge=0, description="1인 월단가 상한(원). 순예산 ÷ 프로젝트 전체 인원 ÷ 개월 수"
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
    # LLM 이 만드는 값이 아니라 **서버가 1차 추림에서 계산해 채워 넣는 값**이다(_RANKING_SCHEMA 에
    # 넣지 않는 이유). LLM 응답을 이 모델로 파싱하는 단계에서는 비어 있으므로 기본값이 None 이고,
    # 풀 밖 후보를 걸러낸 뒤 실제 값으로 덮어쓴다.
    #
    # 스프링 matching_candidate.similarity(numeric(6,4)) 에 그대로 저장된다. 지금까지 그 컬럼엔
    # 0.0 이 박혀 있어서 "이 후보가 왜 뽑혔나"를 나중에 되짚을 수 없었다.
    similarity: float | None = Field(
        default=None,
        description="코사인 유사도(-1~1, 텍스트 임베딩이라 실제로는 0~1 부근). 순위 환산 전 원본값",
    )


class MatchingResponse(BaseModel):
    position_id: int
    model: str = Field(..., description="재랭킹에 쓴 LLM 모델명. 결과 재현에 필요하다")
    candidates: list[RankedCandidate]
