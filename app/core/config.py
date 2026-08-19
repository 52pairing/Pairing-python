"""환경 설정.

값이 없으면 기동 시점에 실패한다. LLM 키나 DB 주소가 빠진 채로 떠서
첫 요청에서야 터지는 것보다 못 뜨는 게 낫다. (스프링 쪽 규칙과 같다)
"""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- 실행 ----------
    app_env: str = Field(default="local", description="local / dev / prod")
    app_port: int = 8000
    log_level: str = "INFO"

    # ---------- DB (스프링과 같은 PostgreSQL) ----------
    # 스프링은 JDBC URL, 여기는 asyncpg URL 이라 값이 다르다. 호스트/DB명은 같아야 한다.
    ai_db_url: str = Field(..., description="postgresql+asyncpg://user:pass@host:5432/pairing")
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # ---------- 내부 통신 ----------
    # 스프링이 이 키를 헤더에 실어 보낸다. 외부에 열지 않는다.
    internal_api_key: str = Field(..., description="스프링과 공유하는 내부 호출 키")
    spring_base_url: str = "http://localhost:8080"
    spring_timeout_seconds: float = 5.0

    # ---------- Gemini ----------
    # 키는 여러 개를 쉼표로 넣을 수 있고, 용도별로 모델만 바꾼다.
    # 무료 티어는 키당 일/분 한도가 있어서 한 키가 429 를 내면 다음 키로 넘어간다.
    # (전환 로직은 app/clients/gemini.py)
    gemini_api_key: str = Field(
        ...,
        description="Google AI Studio API 키. 쉼표로 여러 개 지정하면 한도 초과 시 순서대로 전환한다.",
    )
    # 한도 초과로 걸러낸 키를 다시 후보에 넣기까지 기다리는 시간.
    # 분당 한도는 60초면 풀리고 일일 한도는 더 길어서, 그 사이값을 기본으로 둔다.
    gemini_key_cooldown_seconds: int = 300
    # 모델은 별칭(-latest)이 아니라 버전을 고정한다. 별칭은 구글이 최신을 갈아끼우면
    # 코드 변경 없이 동작이 바뀌어, 그 모델에 맞춰 튜닝한 프롬프트가 예고 없이 무효가 된다.
    # 2.5 계열과 text-embedding-004 는 신규 키에서 404(no longer available to new users)라 쓸 수 없다.
    gemini_model_embedding: str = "gemini-embedding-001"
    gemini_model_matching: str = "gemini-3.5-flash"
    gemini_model_review: str = "gemini-3.5-flash-lite"
    gemini_model_negotiation: str = "gemini-3.6-flash"
    gemini_model_contract: str = "gemini-3.5-flash"
    gemini_timeout_seconds: float = 30.0
    gemini_max_retries: int = 2

    # 저장 차원. gemini-embedding-001 은 기본 3072 이라 embed 호출에서 이 값으로 축소해 받는다
    # (DB 컬럼·기존 벡터와 맞추기 위함).
    # 이 값을 바꾸면 pgvector 컬럼과 기존 임베딩 전량 재생성이 함께 필요하다.
    embedding_dimension: int = 768

    # 챗봇 관련성 임계값. 질문과 가장 가까운 정책 청크의 코사인 유사도가 이 값보다 낮으면
    # LLM 을 호출하지 않고 거절한다. 0~1 이고 클수록 엄격하다.
    #
    # 느슨하게(낮게) 시작해서 조인다. 정상 질문을 막는 쪽이 무관한 질문에 답하는 것보다 훨씬
    # 나쁘다 — 사용자는 "챗봇이 고장났다"고 느낀다. 차단 로그를 보고 조정한다.
    #
    # 실측(2026-08-14, 청크 14개 시딩 후 질문 24개):
    #   범위 안  0.653 ~ 0.803 (평균 0.731)
    #   범위 밖  0.483 ~ 0.559 (평균 0.527)
    # 두 무리 사이가 비어 있어 그 안이면 어디를 잘라도 된다. 정중앙은 0.61 이지만 한 칸 낮게
    # 잡는다 — 무관한 질문이 새면 LLM 이 2차로 거르지만(토큰만 손해), 정상 질문이 막히면
    # 사용자는 답을 못 받는다. 새 청크를 추가하면 이 분포가 달라지니 다시 재고 정한다.
    chatbot_relevance_threshold: float = 0.60

    # 음성(차단 본보기) 청크가 양성 청크를 이만큼 넘게 앞서야 차단한다.
    #
    # 0 으로 두면(단순 최근접 비교) 정상 질문이 막힌다. 음성 청크는 주제어를 나열하게 되어서
    # 짧고 일반적인 질문이 거기에 더 붙는다. 실측에서 "수수료 얼마야?"가 0.750(음성) 대
    # 0.740(양성)으로 0.01 차이로 막혔다.
    #
    # 실측(2026-08-14, 질문 34개)으로 안전 구간을 구했다:
    #   정상 질문이 음성에 끌린 최대치            +0.047
    #   여유 규칙으로 차단되는 질문의 최소 차이    +0.120
    #     (그보다 작은 차이로 무관한 질문은 유사도 하한이 이미 처리한다)
    # 즉 0.047 ~ 0.120 사이면 되고, 한쪽에 붙지 않게 가운데를 잡았다.
    #
    # 이 값은 "정상 질문을 막는 쪽이 더 나쁘다"를 숫자로 표현한 것이다. 애매하면 통과시킨다.
    chatbot_negative_margin: float = 0.08

    # 답변 생성 temperature. 기본값(1.0)을 그대로 쓰면 같은 질문에도 정책 문구를
    # 매번 다르게 바꿔 말할 위험이 있다. 이 챗봇은 답변을 _POLICY_CONTEXT 안에서만
    # 만들고 메뉴 경로도 원문 그대로 써야 하는, 창의성보다 일관성이 중요한 용도라 낮춘다.
    # 0으로 두면 문장이 기계적으로 반복돼 0.2~0.3 사이로 잡는다.
    chatbot_temperature: float = 0.2
    # ---------- AI 스텁 (부하 테스트 전용) ----------
    # 켜면 Gemini 를 실제로 부르지 않고 스키마에 맞는 더미를 만들어 돌려준다.
    # (app/clients/gemini_stub.py)
    #
    # 왜 필요한가
    #   k6 로 AI 경로에 부하를 걸면 실제 호출이 그만큼 나가서 (1) 무료 티어 쿼터가 소진되고
    #   (2) 키가 전부 쿨다운에 들어가 그 시점부터 전부 실패하며 (3) 돈이 든다. 그러면 측정하려던
    #   "부하가 늘 때 응답이 어떻게 변하는가" 대신 "쿼터가 언제 떨어지는가"를 재게 된다.
    #
    #   더미는 지연을 파라미터로 주므로 매 실행이 같은 조건이 된다. 스프링 스레드가 AI 응답을
    #   기다리며 점유되는 문제(부하 테스트의 주 관심사)는 더미로도 그대로 재현된다.
    #
    # ★ 켜 둔 채로 잊으면 서비스가 조용히 가짜 응답을 준다. 그래서 세 겹으로 막는다.
    #   1) prod 에서는 아래 검증이 기동을 거부한다
    #   2) 기동 로그에 매번 경고를 남긴다
    #   3) gemini_stub_mode 게이지가 1 이 되어 대시보드에 드러난다
    ai_stub_mode: bool = False

    # 더미가 응답하기까지 흉내낼 지연(ms). 0 이면 즉시 응답한다.
    #   0        — 순수 인프라 한계 측정 (AI 대기가 없을 때의 상한)
    #   3000     — 빠른 LLM 응답
    #   20000    — 느린 응답. 스프링 스레드 점유 문제가 이 근처에서 드러난다
    #   125000   — 스프링 AI_TIMEOUT_MS(120초) 초과. 타임아웃 경로와 취소 처리를 검증한다
    ai_stub_delay_ms: int = 0

    # 위 지연에 더할 흔들림(± ms). 전부 같은 값이면 지연 히스토그램이 한 버킷에만 쌓여서
    # p50/p95/p99 가 구분되지 않는다. 실제 LLM 처럼 퍼뜨려야 백분위 그래프가 의미를 갖는다.
    ai_stub_jitter_ms: int = 0

    # 더미가 실패를 섞는 비율(0.0~1.0). 서킷브레이커 동작과 outcome 분리(error/timeout)를
    # 확인할 때만 쓴다. 기본은 전부 성공이다.
    ai_stub_fail_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("gemini_api_key")
    @classmethod
    def _require_one_gemini_key(cls, value: str) -> str:
        """쉼표만 있거나 공백뿐이면 기동 시점에 막는다. 첫 호출에서 터지는 것보다 낫다."""
        if not [key for key in (value or "").split(",") if key.strip()]:
            raise ValueError("GEMINI_API_KEY 에 유효한 키가 없습니다. (쉼표로 여러 개 지정 가능)")
        return value

    @model_validator(mode="after")
    def _forbid_stub_in_prod(self) -> "Settings":
        """운영에서는 스텁을 켤 수 없다.

        환경변수 하나가 남아 있으면 서비스가 조용히 가짜 응답을 준다. 로그 경고만으로는
        막을 수 없다 — 아무도 기동 로그를 계속 보지 않는다. 그래서 기동 자체를 거부한다.
        """
        if self.ai_stub_mode and self.app_env == "prod":
            raise ValueError(
                "AI_STUB_MODE 는 prod 에서 쓸 수 없습니다. 부하 테스트용 더미 응답 설정입니다."
            )
        return self

    @property
    def gemini_api_keys(self) -> list[str]:
        """쉼표로 구분된 키 목록. 입력 순서를 유지하고 중복은 제거한다.

        같은 키를 두 번 적으면 한도도 함께 소진되므로 후보를 늘리는 효과가 없다. 그래서 지운다.
        """
        unique: dict[str, None] = {}
        for raw in self.gemini_api_key.split(","):
            key = raw.strip()
            if key:
                unique.setdefault(key, None)
        return list(unique)


@lru_cache
def get_settings() -> Settings:
    """설정은 프로세스당 한 번만 읽는다. 테스트에서는 캐시를 비우고 갈아끼운다."""
    return Settings()
