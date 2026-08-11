"""GEMINI_API_KEY 다중 키 전환 테스트.

실제 Gemini 를 부르지 않는다. genai.Client 를 더미로 갈아끼우고, 호출 함수가 429 를 던지게 해서
"다음 키로 넘어가는지" 만 본다.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.clients import gemini as gemini_module
from app.core.config import Settings
from app.core.errors import AiErrorCode, AiException


def _settings(keys: str, **overrides) -> Settings:
    return Settings(
        ai_db_url="postgresql+asyncpg://pairing:pairing@localhost:5432/pairing",
        internal_api_key="test-internal-key",
        gemini_api_key=keys,
        **overrides,
    )


class _QuotaError(Exception):
    """google-genai 가 429 를 낼 때와 비슷한 모양."""

    def __init__(self):
        super().__init__("429 RESOURCE_EXHAUSTED: quota exceeded for this key")
        self.code = 429


class _BadRequest(Exception):
    """우리 요청이 잘못된 경우. 키를 바꿔도 소용없다."""

    def __init__(self):
        super().__init__("400 INVALID_ARGUMENT: bad request")
        self.code = 400


@pytest.fixture
def dummy_genai(monkeypatch):
    """genai.Client(api_key=...) 를 api_key 만 들고 있는 더미로 바꾼다."""
    monkeypatch.setattr(
        gemini_module.genai, "Client", lambda api_key: SimpleNamespace(api_key=api_key)
    )


# ---------------------------------------------------------------- 설정 파싱


def test_keys_are_split_trimmed_and_deduped():
    settings = _settings(" k1 , k2 ,k1, , k3 ")
    assert settings.gemini_api_keys == ["k1", "k2", "k3"]


def test_single_key_still_works():
    assert _settings("k1").gemini_api_keys == ["k1"]


def test_blank_key_is_rejected_at_startup():
    with pytest.raises(ValidationError):
        _settings(" , , ")


# ---------------------------------------------------------------- 키 전환


async def test_rotates_to_next_key_on_quota_error(dummy_genai):
    client = gemini_module.GeminiClient(_settings("k1,k2,k3"))
    used: list[str] = []

    async def call(inner):
        used.append(inner.api_key)
        if inner.api_key == "k1":
            raise _QuotaError()
        return "ok"

    result, attempts = await client._invoke(call, AiErrorCode.LLM_CALL_FAILED)

    assert result == "ok"
    assert used == ["k1", "k2"]  # k1 한 번 실패 후 k2 로 넘어갔다
    assert attempts == 2


async def test_exhausted_key_is_skipped_on_next_call(dummy_genai):
    """한 번 429 를 낸 키는 쿨다운 동안 후보에서 빠진다."""
    client = gemini_module.GeminiClient(_settings("k1,k2"))

    async def fail_k1(inner):
        if inner.api_key == "k1":
            raise _QuotaError()
        return "ok"

    await client._invoke(fail_k1, AiErrorCode.LLM_CALL_FAILED)

    used: list[str] = []

    async def record(inner):
        used.append(inner.api_key)
        return "ok"

    _, attempts = await client._invoke(record, AiErrorCode.LLM_CALL_FAILED)

    assert used == ["k2"]  # k1 은 시도조차 하지 않는다
    assert attempts == 1


async def test_all_keys_exhausted_raises_ai_exception(dummy_genai):
    client = gemini_module.GeminiClient(_settings("k1,k2,k3"))
    used: list[str] = []

    async def always_quota(inner):
        used.append(inner.api_key)
        raise _QuotaError()

    with pytest.raises(AiException) as exc_info:
        await client._invoke(always_quota, AiErrorCode.EMBEDDING_FAILED)

    assert exc_info.value.error_code is AiErrorCode.EMBEDDING_FAILED
    assert used == ["k1", "k2", "k3"]  # 키를 한 번씩만 쓴다


async def test_bad_request_does_not_burn_keys(dummy_genai):
    """400 은 우리 요청 문제라 키를 소모하지 않고 바로 올린다."""
    client = gemini_module.GeminiClient(_settings("k1,k2,k3"))
    used: list[str] = []

    async def bad(inner):
        used.append(inner.api_key)
        raise _BadRequest()

    with pytest.raises(AiException):
        await client._invoke(bad, AiErrorCode.LLM_CALL_FAILED)

    assert used == ["k1"]
    assert client._cooldown_until == [0.0, 0.0, 0.0]  # 쿨다운에 들어간 키가 없다


async def test_our_own_exception_is_not_retried(dummy_genai):
    """차원 불일치처럼 우리가 던진 AiException 은 그대로 올라간다."""
    client = gemini_module.GeminiClient(_settings("k1,k2"))
    calls = 0

    async def ours(_inner):
        nonlocal calls
        calls += 1
        raise AiException(AiErrorCode.EMBEDDING_FAILED, "임베딩 차원이 설정과 다릅니다.")

    with pytest.raises(AiException) as exc_info:
        await client._invoke(ours, AiErrorCode.EMBEDDING_FAILED)

    assert calls == 1
    assert "차원" in exc_info.value.message


async def test_invalid_response_is_retried_up_to_three_attempts(dummy_genai):
    """명세: "LLM 응답 오류 시 재시도(3회) → 3회 실패하면 응답에 실패했음 표시".

    응답이 비었거나 형식이 깨진 건 같은 키로 다시 부르면 성공할 수 있어서 재시도 대상이다.
    (차원 불일치 같은 우리 쪽 오류는 위 테스트처럼 재시도하지 않는다 — 구분이 핵심)
    """
    client = gemini_module.GeminiClient(_settings("k1,k2"))
    calls = 0

    async def invalid(_inner):
        nonlocal calls
        calls += 1
        raise AiException(AiErrorCode.LLM_RESPONSE_INVALID)

    with pytest.raises(AiException) as exc_info:
        await client._invoke(invalid, AiErrorCode.LLM_CALL_FAILED)

    # 최초 1회 + 재시도 2회(gemini_max_retries 기본값) = 총 3회
    assert calls == 3
    # 소진 후에도 "호출 실패"가 아니라 "응답 오류"로 남아야 원인이 안 뒤바뀐다.
    assert exc_info.value.error_code == AiErrorCode.LLM_RESPONSE_INVALID
    # 응답 오류는 키 문제가 아니므로 키를 쿨다운에 넣지 않는다.
    assert client._cooldown_until == [0.0, 0.0]


async def test_invalid_response_succeeds_on_retry(dummy_genai):
    """첫 응답이 비어도 재시도에서 정상 응답이 오면 성공으로 끝난다."""
    client = gemini_module.GeminiClient(_settings("k1"))
    calls = 0

    async def flaky(_inner):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AiException(AiErrorCode.LLM_RESPONSE_INVALID)
        return "ok"

    result, attempts = await client._invoke(flaky, AiErrorCode.LLM_CALL_FAILED)

    assert result == "ok"
    assert attempts == 2
