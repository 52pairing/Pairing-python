"""Gemini 클라이언트.

키는 하나를 공유하고 **용도별로 모델만 바꾼다.** 모델명을 코드에 흩뿌리면
"어디를 바꿔야 임베딩 모델이 바뀌는지" 아무도 모르게 되므로 설정에서만 읽는다.

    임베딩  -> settings.gemini_model_embedding
    매칭    -> settings.gemini_model_matching
    검수    -> settings.gemini_model_review
"""

import logging
from enum import Enum

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings
from app.core.errors import AiErrorCode, AiException

logger = logging.getLogger(__name__)


class GeminiTask(Enum):
    """용도. 새 용도가 생기면 여기와 설정에 함께 추가한다."""

    EMBEDDING = "embedding"
    MATCHING = "matching"
    REVIEW = "review"


class GeminiClient:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def model_for(self, task: GeminiTask) -> str:
        mapping = {
            GeminiTask.EMBEDDING: self._settings.gemini_model_embedding,
            GeminiTask.MATCHING: self._settings.gemini_model_matching,
            GeminiTask.REVIEW: self._settings.gemini_model_review,
        }
        return mapping[task]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(AiException),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트를 벡터로 바꾼다. 저장 차원은 settings.embedding_dimension 과 맞아야 한다."""
        model = self.model_for(GeminiTask.EMBEDDING)
        try:
            response = await self._client.aio.models.embed_content(model=model, contents=texts)
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
        return vectors

    async def generate_json(self, task: GeminiTask, prompt: str, response_schema: dict) -> str:
        """구조화된 응답을 받는다. 자유 텍스트를 파싱하면 프롬프트가 바뀔 때마다 깨진다."""
        model = self.model_for(task)
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
        return response.text


_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """라우터 의존성. 클라이언트는 프로세스당 하나면 된다."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
