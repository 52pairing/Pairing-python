"""Gemini 클라이언트.

키는 하나를 공유하고 **용도별로 모델만 바꾼다.** 모델명을 코드에 흩뿌리면
"어디를 바꿔야 임베딩 모델이 바뀌는지" 아무도 모르게 되므로 설정에서만 읽는다.

    임베딩  -> settings.gemini_model_embedding
    매칭    -> settings.gemini_model_matching
    검수    -> settings.gemini_model_review
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings
from app.core.errors import AiErrorCode, AiException

logger = logging.getLogger(__name__)


@dataclass
class GeminiUsage:
    """호출 1건의 통계. `ai_agent_log` 에 그대로 들어간다.

    토큰 수는 None 일 수 있다 — **임베딩 호출(embed_content)은 usage_metadata 를 주지 않는다**
    (2026-08-10 실제 호출로 확인). 토큰이 없다고 로그 자체를 포기하지는 않는다. 지연시간·모델·
    재시도 횟수는 임베딩에서도 정상적으로 채워진다.
    """

    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    retry_count: int


def _usage(model: str, response: object, started: float, attempts: int) -> GeminiUsage:
    metadata = getattr(response, "usage_metadata", None)
    return GeminiUsage(
        model=model,
        prompt_tokens=getattr(metadata, "prompt_token_count", None),
        output_tokens=getattr(metadata, "candidates_token_count", None),
        latency_ms=int((time.perf_counter() - started) * 1000),
        # 첫 시도는 재시도가 아니다. 3번 시도했으면 재시도 2회.
        retry_count=max(0, attempts - 1),
    )


class GeminiTask(Enum):
    """용도. 새 용도가 생기면 여기와 설정에 함께 추가한다."""

    EMBEDDING = "embedding"
    MATCHING = "matching"
    REVIEW = "review"
    NEGOTIATION = "negotiation"
    CONTRACT = "contract"


class GeminiClient:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def model_for(self, task: GeminiTask) -> str:
        mapping = {
            GeminiTask.EMBEDDING: self._settings.gemini_model_embedding,
            GeminiTask.MATCHING: self._settings.gemini_model_matching,
            GeminiTask.REVIEW: self._settings.gemini_model_review,
            GeminiTask.NEGOTIATION: self._settings.gemini_model_negotiation,
            GeminiTask.CONTRACT: self._settings.gemini_model_contract,
        }
        return mapping[task]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트를 벡터로 바꾼다. 저장 차원은 settings.embedding_dimension 과 맞아야 한다."""
        vectors, _ = await self.embed_with_usage(texts)
        return vectors

    async def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], GeminiUsage]:
        """{@link embed} 와 같지만 호출 통계를 함께 준다. ai_agent_log 를 남기는 쪽에서 쓴다."""
        model = self.model_for(GeminiTask.EMBEDDING)
        # 재시도 횟수를 ai_agent_log 에 남겨야 해서 시도 수를 직접 센다. tenacity 의 통계는
        # 함수 객체에 붙어 호출 간 공유되므로(동시 요청이 섞인다) 여기서 세는 게 맞다.
        attempts = 0
        started = time.perf_counter()

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception_type(AiException),
            reraise=True,
        )
        async def _call():
            nonlocal attempts
            attempts += 1
            try:
                # gemini-embedding-001 은 기본 3072 차원이라 축소를 명시해야 pgvector 컬럼(768)과 맞는다.
                response = await self._client.aio.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self._settings.embedding_dimension
                    ),
                )
            except Exception as exc:
                logger.warning("임베딩 호출 실패: model=%s, cause=%s", model, exc)
                raise AiException(AiErrorCode.EMBEDDING_FAILED) from exc

            vectors = [item.values for item in response.embeddings]
            expected = self._settings.embedding_dimension
            if any(len(vector) != expected for vector in vectors):
                # 모델을 바꿨는데 DB 컬럼 차원을 안 바꾼 상황. 저장 시점에 터지면 원인 찾기 어렵다.
                raise AiException(
                    AiErrorCode.EMBEDDING_FAILED,
                    f"임베딩 차원이 설정과 다릅니다. (기대 {expected})",
                )
            return vectors, response

        vectors, response = await _call()
        return vectors, _usage(model, response, started, attempts)

    async def generate_json(self, task: GeminiTask, prompt: str, response_schema: dict) -> str:
        """구조화된 응답을 받는다. 자유 텍스트를 파싱하면 프롬프트가 바뀔 때마다 깨진다."""
        text, _ = await self.generate_json_with_usage(task, prompt, response_schema)
        return text

    async def generate_json_with_usage(
        self, task: GeminiTask, prompt: str, response_schema: dict
    ) -> tuple[str, GeminiUsage]:
        """{@link generate_json} 과 같지만 호출 통계를 함께 준다. ai_agent_log 를 남기는 쪽에서 쓴다."""
        model = self.model_for(task)
        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
        except TimeoutError as exc:
            raise AiException(AiErrorCode.LLM_TIMEOUT) from exc
        except Exception as exc:
            logger.warning("LLM 호출 실패: model=%s, cause=%s", model, exc)
            raise AiException(AiErrorCode.LLM_CALL_FAILED) from exc

        if not response.text:
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID)
        return response.text, _usage(model, response, started, attempts=1)


_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """라우터 의존성. 클라이언트는 프로세스당 하나면 된다."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
