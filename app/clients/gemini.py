"""Gemini 클라이언트.

키는 **여러 개를 쉼표로** 받고(`GEMINI_API_KEY`), 용도별로 모델만 바꾼다. 모델명을 코드에
흩뿌리면 "어디를 바꿔야 임베딩 모델이 바뀌는지" 아무도 모르게 되므로 설정에서만 읽는다.

    임베딩  -> settings.gemini_model_embedding
    매칭    -> settings.gemini_model_matching
    검수    -> settings.gemini_model_review

무료 티어는 키마다 분/일 한도가 있다. 한 키가 429(RESOURCE_EXHAUSTED)를 내면 그 키를
쿨다운에 넣고 **다음 키로 이어서** 호출한다. 일시적 오류(타임아웃·5xx)는 키를 바꾸지 않고
같은 키로 잠깐 쉬었다 다시 시도한다 — 키를 바꿔도 해결되지 않는 종류라 후보만 낭비된다.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types

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
        # 첫 시도는 재시도가 아니다. 3번 시도했으면 재시도 2회. (키 전환도 시도로 센다)
        retry_count=max(0, attempts - 1),
    )


class GeminiTask(Enum):
    """용도. 새 용도가 생기면 여기와 설정에 함께 추가한다."""

    EMBEDDING = "embedding"
    MATCHING = "matching"
    REVIEW = "review"
    NEGOTIATION = "negotiation"
    CONTRACT = "contract"


class _Action(Enum):
    """실패를 어떻게 다룰지."""

    ROTATE = "rotate"  # 이 키로는 안 된다 -> 쿨다운에 넣고 다음 키로
    RETRY = "retry"  # 일시적이다 -> 같은 키로 잠깐 뒤에 다시
    FAIL = "fail"  # 우리 요청이 틀렸거나 회복 불가 -> 그대로 올린다


# google-genai 의 예외는 버전에 따라 속성명(code/status_code)과 메시지 형식이 달라진다.
# 그래서 상태코드와 메시지 문자열을 함께 본다.
_ROTATE_MARKERS = (
    "resource_exhausted",
    "quota",
    "rate limit",
    "too many requests",
    "api key not valid",
    "api_key_invalid",
    "permission_denied",
)
_RETRY_MARKERS = ("unavailable", "overloaded", "deadline", "timed out", "internal error")


def _status_of(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _brief(exc: BaseException) -> str:
    """로그용 요약. 예외 본문이 길어 로그를 덮는 것을 막는다."""
    return str(exc).replace("\n", " ")[:200]


def _classify(exc: BaseException) -> _Action:
    if isinstance(exc, AiException):
        # 응답이 비었거나 형식이 깨진 건 같은 키로 다시 부르면 성공할 수 있다. 명세가
        # "LLM 응답 오류 시 재시도(3회), 3회 실패하면 실패 표시"를 요구하므로 재시도 대상이다.
        if exc.error_code is AiErrorCode.LLM_RESPONSE_INVALID:
            return _Action.RETRY
        # 그 밖에 우리가 던진 예외(임베딩 차원 불일치 등)는 다시 시도해도 같은 결과다.
        return _Action.FAIL

    status = _status_of(exc)
    text = str(exc).lower()

    # 429 = 한도 초과, 401/403 = 죽었거나 권한 없는 키. 둘 다 이 키로는 답이 없다.
    if status == 429 or status in (401, 403) or any(m in text for m in _ROTATE_MARKERS):
        return _Action.ROTATE

    if isinstance(exc, TimeoutError | ConnectionError):
        return _Action.RETRY
    if status in (500, 502, 503, 504) or any(m in text for m in _RETRY_MARKERS):
        return _Action.RETRY

    return _Action.FAIL


class GeminiClient:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        keys = self._settings.gemini_api_keys

        # 키마다 클라이언트를 미리 만든다. 전환 시점에 만들면 그 생성 비용이 요청 지연에 실린다.
        self._clients = [genai.Client(api_key=key) for key in keys]
        self._key_count = len(self._clients)

        # 한도에 걸린 키를 언제까지 건너뛸지(monotonic 초). 단일 이벤트 루프에서만 갱신되므로
        # 락이 필요 없다. await 를 끼고 갱신하지 않도록 주의한다.
        self._cooldown_until = [0.0] * self._key_count
        self._index = 0

        logger.info("Gemini 키 %d개 로드 (한도 초과 시 순서대로 전환)", self._key_count)

    def model_for(self, task: GeminiTask) -> str:
        mapping = {
            GeminiTask.EMBEDDING: self._settings.gemini_model_embedding,
            GeminiTask.MATCHING: self._settings.gemini_model_matching,
            GeminiTask.REVIEW: self._settings.gemini_model_review,
            GeminiTask.NEGOTIATION: self._settings.gemini_model_negotiation,
            GeminiTask.CONTRACT: self._settings.gemini_model_contract,
        }
        return mapping[task]

    # ------------------------------------------------------------------
    # 키 선택
    # ------------------------------------------------------------------

    def _pick(self) -> int:
        """지금 쓸 키 인덱스. 쿨다운 중인 키는 건너뛴다."""
        now = time.monotonic()
        for offset in range(self._key_count):
            index = (self._index + offset) % self._key_count
            if self._cooldown_until[index] <= now:
                self._index = index
                return index

        # 전부 쿨다운이면 가장 먼저 풀리는 키로 그냥 시도한다. 한도가 예상보다 일찍 풀릴 수 있어서,
        # 아무 호출도 하지 않고 실패시키는 것보다 한 번 찔러 보는 편이 낫다.
        index = min(range(self._key_count), key=lambda i: self._cooldown_until[i])
        self._index = index
        return index

    def _disable(self, index: int, exc: BaseException) -> None:
        seconds = self._settings.gemini_key_cooldown_seconds
        self._cooldown_until[index] = time.monotonic() + seconds
        # 다음 요청은 그다음 키에서 시작한다.
        self._index = (index + 1) % self._key_count
        # 키 값은 절대 로그에 남기지 않는다. 몇 번째 키인지만 남긴다.
        logger.warning(
            "Gemini 키 %d/%d 를 %d초간 제외한다: %s",
            index + 1,
            self._key_count,
            seconds,
            _brief(exc),
        )

    async def _invoke(
        self,
        call,
        fallback: AiErrorCode,
        timeout_code: AiErrorCode | None = None,
    ) -> tuple[object, int]:
        """`call(client)` 을 성공할 때까지 키를 돌려 가며 수행한다. (결과, 총 시도 횟수) 를 준다.

        키 전환은 최대 키 개수만큼, 같은 키 재시도는 `gemini_max_retries` 회까지 한다.
        """
        attempts = 0
        retries_left = self._settings.gemini_max_retries
        rotations_left = self._key_count
        last_exc: BaseException | None = None

        while True:
            index = self._pick()
            attempts += 1
            try:
                return await call(self._clients[index]), attempts
            except Exception as exc:
                last_exc = exc
                action = _classify(exc)

                if action is _Action.FAIL:
                    if isinstance(exc, AiException):
                        raise
                    logger.warning("Gemini 호출 실패(재시도 안 함): %s", _brief(exc))
                    raise AiException(fallback) from exc

                if action is _Action.ROTATE:
                    self._disable(index, exc)
                    rotations_left -= 1
                    if rotations_left <= 0:
                        break
                    continue

                # RETRY: 같은 키로 잠깐 쉬었다 다시.
                if retries_left <= 0:
                    break
                retries_left -= 1
                backoff = min(4.0, 0.5 * 2 ** (self._settings.gemini_max_retries - retries_left))
                logger.warning("Gemini 일시 오류, %.1f초 뒤 재시도: %s", backoff, _brief(exc))
                await asyncio.sleep(backoff)

        if timeout_code is not None and isinstance(last_exc, TimeoutError):
            code = timeout_code
        elif isinstance(last_exc, AiException):
            # 응답 오류로 재시도하다 소진된 경우. fallback(호출 실패)로 덮으면 "왜 실패했나"가
            # 뒤바뀐다 — 호출은 됐고 응답이 계속 이상했던 것이다.
            code = last_exc.error_code
        else:
            code = fallback
        raise AiException(
            code,
            f"Gemini 호출이 실패했습니다. (키 {self._key_count}개, 시도 {attempts}회)",
        ) from last_exc

    # ------------------------------------------------------------------
    # 호출
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트를 벡터로 바꾼다. 저장 차원은 settings.embedding_dimension 과 맞아야 한다."""
        vectors, _ = await self.embed_with_usage(texts)
        return vectors

    async def embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], GeminiUsage]:
        """{@link embed} 와 같지만 호출 통계를 함께 준다. ai_agent_log 를 남기는 쪽에서 쓴다."""
        model = self.model_for(GeminiTask.EMBEDDING)
        started = time.perf_counter()

        async def _call(client: genai.Client):
            # gemini-embedding-001 은 기본 3072 차원이라 축소를 명시해야 pgvector 컬럼(768)과 맞는다.
            response = await client.aio.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._settings.embedding_dimension
                ),
            )
            vectors = [item.values for item in response.embeddings]
            expected = self._settings.embedding_dimension
            if any(len(vector) != expected for vector in vectors):
                # 모델을 바꿨는데 DB 컬럼 차원을 안 바꾼 상황. 저장 시점에 터지면 원인 찾기 어렵다.
                raise AiException(
                    AiErrorCode.EMBEDDING_FAILED,
                    f"임베딩 차원이 설정과 다릅니다. (기대 {expected})",
                )
            return vectors, response

        result, attempts = await self._invoke(_call, AiErrorCode.EMBEDDING_FAILED)
        vectors, response = result
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

        async def _call(client: genai.Client):
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            if not response.text:
                raise AiException(AiErrorCode.LLM_RESPONSE_INVALID)
            return response

        response, attempts = await self._invoke(
            _call, AiErrorCode.LLM_CALL_FAILED, timeout_code=AiErrorCode.LLM_TIMEOUT
        )
        return response.text, _usage(model, response, started, attempts)


_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """라우터 의존성. 클라이언트는 프로세스당 하나면 된다.

    키 쿨다운 상태를 이 인스턴스가 들고 있으므로, 요청마다 새로 만들면 전환 이력이 사라진다.
    """
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
