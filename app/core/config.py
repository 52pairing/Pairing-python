"""환경 설정.

값이 없으면 기동 시점에 실패한다. LLM 키나 DB 주소가 빠진 채로 떠서
첫 요청에서야 터지는 것보다 못 뜨는 게 낫다. (스프링 쪽 규칙과 같다)
"""

from functools import lru_cache

from pydantic import Field, field_validator
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

    @field_validator("gemini_api_key")
    @classmethod
    def _require_one_gemini_key(cls, value: str) -> str:
        """쉼표만 있거나 공백뿐이면 기동 시점에 막는다. 첫 호출에서 터지는 것보다 낫다."""
        if not [key for key in (value or "").split(",") if key.strip()]:
            raise ValueError("GEMINI_API_KEY 에 유효한 키가 없습니다. (쉼표로 여러 개 지정 가능)")
        return value

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
