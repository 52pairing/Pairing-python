"""환경 설정.

값이 없으면 기동 시점에 실패한다. LLM 키나 DB 주소가 빠진 채로 떠서
첫 요청에서야 터지는 것보다 못 뜨는 게 낫다. (스프링 쪽 규칙과 같다)
"""

from functools import lru_cache

from pydantic import Field
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
    # 키는 하나를 공유하고 용도별로 모델만 바꾼다.
    gemini_api_key: str = Field(..., description="Google AI Studio API 키")
    gemini_model_embedding: str = "text-embedding-004"
    gemini_model_matching: str = "gemini-2.0-flash"
    gemini_model_review: str = "gemini-2.0-flash"
    gemini_model_negotiation: str = "gemini-flash-latest"
    gemini_timeout_seconds: float = 30.0
    gemini_max_retries: int = 2

    # text-embedding-004 의 차원. 모델을 바꾸면 이 값과 DB 컬럼을 함께 바꿔야 한다.
    embedding_dimension: int = 768


@lru_cache
def get_settings() -> Settings:
    """설정은 프로세스당 한 번만 읽는다. 테스트에서는 캐시를 비우고 갈아끼운다."""
    return Settings()
