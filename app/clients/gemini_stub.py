"""Gemini 더미 클라이언트 — 부하 테스트 전용.

`genai.Client` 의 **표면만** 흉내낸다. 그래서 `GeminiClient` 의 재시도·키 전환·메트릭 기록
로직이 하나도 우회되지 않고 그대로 실행된다. 스텁 모드에서도 `gemini_requests_total`,
`gemini_attempts_total`, `gemini_request_duration_seconds` 가 실제 호출과 같은 모양으로
채워지므로 Grafana 대시보드를 그대로 쓸 수 있다.

스텁을 여기(전송 계층)에 둔 이유
    도메인 서비스나 `GeminiClient` 메서드 안에서 분기하면 재시도·키 전환 경로가 통째로
    빠진다. 그러면 "재시도가 지연을 만들고 있는가"를 부하 테스트로 볼 수 없다 —
    측정하려던 대상이 스텁 때문에 사라진다.

응답을 어떻게 만드는가
    호출자가 넘긴 `response_schema`(JSON Schema) 를 읽어서 그 스키마를 만족하는 값을
    조립한다. 도메인마다 더미를 따로 쓰지 않아도 되고, 스키마가 바뀌면 더미도 따라간다.

    프롬프트에 들어 있는 `condition_id=11` 같은 식별자는 정규식으로 뽑아 배열 길이와 값에
    쓴다. 협상 도메인은 "요청한 모든 쟁점에 결과가 있어야" 통과하므로(negotiation/service.py
    의 `_parse`) 이게 없으면 더미가 검증에서 막혀 500 이 된다. 500 이 나면 스프링의
    resilience4j 서킷이 열리고, 그 뒤 호출은 AI 를 타지 않고 즉시 차단되어 **부하 테스트가
    아무것도 측정하지 못한다.**

★ 더미 문자열에는 일부러 "더미"라고 적는다. 스텁을 켠 채 잊었을 때 화면에서 바로 드러나야
  한다. 그럴듯한 가짜 문장을 돌려주면 아무도 눈치채지 못한다.
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import re

from app.core.config import Settings

logger = logging.getLogger(__name__)

# 더미임이 드러나야 하는 자리에 넣는 문장.
STUB_TEXT = "[더미] 부하 테스트용 응답입니다. AI_STUB_MODE 가 켜져 있습니다."

# 프롬프트에서 식별자를 뽑는다. `condition_id=11`, `"freelancer_id": 5` 둘 다 잡는다.
_ID_PATTERN = re.compile(r"([a-z][a-z0-9_]*_id)[\"']?\s*[:=]\s*(\d+)")

# 스키마 재귀 깊이 상한. 스키마가 자기 자신을 참조하면 무한히 내려간다.
_MAX_DEPTH = 12


def _extract_id_hints(prompt: str) -> dict[str, list[int]]:
    """프롬프트에서 `<이름>_id = <숫자>` 를 모은다. 등장 순서를 유지하고 중복은 지운다."""
    hints: dict[str, list[int]] = {}
    for name, raw in _ID_PATTERN.findall(prompt):
        seen = hints.setdefault(name, [])
        value = int(raw)
        if value not in seen:
            seen.append(value)
    return hints


def _pick_array_ids(
    items_schema: dict, hints: dict[str, list[int]]
) -> tuple[str | None, list[int]]:
    """배열 항목이 식별자 필드를 가지면 (필드명, 식별자 목록) 을 준다.

    이게 배열 길이를 정한다. 쟁점이 3개면 결과도 3개여야 도메인 검증을 통과한다.
    """
    properties = (items_schema or {}).get("properties") or {}
    for name in properties:
        if hints.get(name):
            return name, hints[name]
    return None, []


def _dummy_for(
    schema: dict,
    hints: dict[str, list[int]],
    *,
    fixed: dict | None = None,
    depth: int = 0,
    field: str | None = None,
):
    """JSON Schema 를 만족하는 값을 만든다.

    field 는 이 값이 들어갈 속성 이름이다. 불리언을 정할 때만 쓴다 — 스키마만 봐서는
    True 와 False 중 어느 쪽이 '정상'인지 알 수 없기 때문이다(_dummy_bool 참고).
    """
    if depth > _MAX_DEPTH or not isinstance(schema, dict):
        return None

    if "enum" in schema and schema["enum"]:
        # 첫 값을 쓴다. enum 은 도메인이 받아들이는 값 목록이라 아무 것이나 골라도 통과한다.
        return schema["enum"][0]

    kind = schema.get("type")

    if kind == "object":
        properties = schema.get("properties") or {}
        result = {}
        for name, sub in properties.items():
            if fixed and name in fixed:
                result[name] = fixed[name]
            else:
                result[name] = _dummy_for(sub, hints, depth=depth + 1, field=name)
        return result

    if kind == "array":
        items = schema.get("items") or {}
        name, ids = _pick_array_ids(items, hints)
        if ids:
            return [
                _dummy_for(items, hints, fixed={name: value}, depth=depth + 1) for value in ids
            ]
        # 힌트가 없으면 1개만 만든다. 빈 배열은 "결과 없음"으로 해석되는 도메인이 있다.
        return [_dummy_for(items, hints, depth=depth + 1)]

    if kind == "string":
        return STUB_TEXT

    if kind in ("integer", "number"):
        low = schema.get("minimum", 1)
        high = schema.get("maximum", low + 100)
        if high < low:
            high = low
        # 값을 고정하지 않고 범위 안에서 흔든다. 전부 같은 값이면 점수 분포 같은 후속 처리가
        # 실제와 다르게 동작한다(정렬이 무의미해지는 등).
        value = random.uniform(low, high)
        return int(value) if kind == "integer" else round(value, 2)

    if kind == "boolean":
        return _dummy_bool(field)

    # type 이 없는 스키마(anyOf 등)는 문자열로 둔다. 지금 쓰는 스키마에는 없다.
    return STUB_TEXT


def _dummy_bool(field: str | None) -> bool:
    """불리언 하나를 정한다. 기본은 True 이고, '예외 상황' 이름만 False 로 둔다.

    왜 이름을 보는가
        JSON Schema 는 True 와 False 중 어느 쪽이 정상 흐름인지 말해 주지 않는다. 그런데
        도메인은 그 값으로 흐름을 가른다.

            negotiation.agreed       True  여야 쟁점이 타결된다
            chatbot.out_of_scope     False 여야 답변이 나온다

        전부 True 로 두면 챗봇이 매번 "범위 밖입니다"를 돌려주고 LLM 호출 경로를 타지 않는다.
        그러면 부하 테스트가 재는 것은 챗봇 응답 생성이 아니라 거절 처리다 — 실제로 그렇게
        동작했고(2026-08-14, develop 이 out_of_scope 를 추가한 뒤) 그래서 이 함수가 생겼다.

        '실패·예외를 뜻하는 이름은 False' 라는 규칙이 도메인 의미와 대체로 맞는다. 새 불리언
        필드가 이 규칙과 반대라면 아래 목록에 이름을 추가한다.
    """
    if not field:
        return True
    name = field.lower()
    return not any(hint in name for hint in _FALSE_BOOL_HINTS)


# 이 조각이 이름에 들어 있으면 False 로 만든다. 전부 "정상이 아님"을 뜻하는 말이다.
_FALSE_BOOL_HINTS = (
    "out_of_scope",
    "error",
    "fail",
    "invalid",
    "reject",
    "block",
    "expired",
    "deleted",
    "cancel",
)


class _StubUsage:
    """`response.usage_metadata` 자리. 토큰 그래프가 움직이도록 대략적인 값을 준다."""

    def __init__(self, prompt_chars: int, output_chars: int):
        # 한국어는 토큰당 대략 2~3자, 영어는 4자다. 정확할 필요는 없고 부하에 비례해서
        # 늘어나는 것이 목적이다.
        self.prompt_token_count = max(1, prompt_chars // 3)
        self.candidates_token_count = max(1, output_chars // 3)


class _StubEmbedding:
    def __init__(self, values: list[float]):
        self.values = values


class _StubEmbedResponse:
    def __init__(self, embeddings: list[_StubEmbedding]):
        self.embeddings = embeddings
        # 임베딩 호출은 실제로도 usage_metadata 를 주지 않는다(gemini.py 의 주석 참고).
        # 그 성질까지 같게 두어야 토큰 그래프의 "임베딩은 안 잡힌다"가 스텁에서도 재현된다.
        self.usage_metadata = None


class _StubGenerateResponse:
    def __init__(self, text: str, prompt_chars: int):
        self.text = text
        self.usage_metadata = _StubUsage(prompt_chars, len(text))


class _StubTransientError(RuntimeError):
    """일시 오류로 분류되게 만든 예외.

    `gemini.py` 의 `_classify` 가 메시지에서 "overloaded" 를 보고 RETRY 로 판정한다.
    그래서 주입한 실패가 재시도 경로를 실제로 통과하고, 시도 횟수(`gemini_attempts_total`)와
    백오프 지연이 그래프에 나타난다.
    """

    def __init__(self):
        super().__init__("stub injected failure: model is overloaded")


def _deterministic_vector(text: str, dimension: int) -> list[float]:
    """같은 텍스트에 같은 벡터를 준다.

    매번 다른 벡터를 주면 같은 이력서를 두 번 임베딩했을 때 유사도가 달라져서, 매칭 결과가
    실행마다 흔들린다. 부하 테스트는 같은 조건을 반복해야 비교가 되므로 텍스트에서 시드를 만든다.
    """
    seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    # 실제 임베딩은 대개 단위 벡터다. 코사인 유사도를 쓰는 쪽과 성질을 맞춘다.
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class _StubModels:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def _pause(self) -> None:
        """흉내낼 지연. 여기서 실제로 기다려야 스프링 스레드 점유가 재현된다."""
        base = self._settings.ai_stub_delay_ms
        jitter = self._settings.ai_stub_jitter_ms
        millis = base + (random.uniform(-jitter, jitter) if jitter else 0.0)
        if millis > 0:
            await asyncio.sleep(millis / 1000.0)

    def _maybe_fail(self) -> None:
        """설정된 비율로 실패를 주입한다. **시도마다** 판정한다.

        호출 단위가 아니라 시도 단위로 판정해야 재시도가 실제로 일어나고, "재시도가 지연을
        만들고 있는가"(호출당 평균 시도 횟수)를 부하 테스트로 확인할 수 있다.
        """
        rate = self._settings.ai_stub_fail_rate
        if rate > 0 and random.random() < rate:
            raise _StubTransientError()

    async def embed_content(self, *, model: str, contents, config=None):
        await self._pause()
        self._maybe_fail()

        texts = contents if isinstance(contents, list) else [contents]
        # 차원은 호출자가 config 로 넘긴 값을 그대로 쓴다. 설정에서 다시 읽으면 둘이
        # 어긋났을 때 스텁만 통과하고 실제 호출은 실패하는 상황이 생긴다.
        dimension = getattr(config, "output_dimensionality", None) or (
            self._settings.embedding_dimension
        )
        return _StubEmbedResponse(
            [_StubEmbedding(_deterministic_vector(str(text), dimension)) for text in texts]
        )

    async def generate_content(self, *, model: str, contents, config=None):
        await self._pause()
        self._maybe_fail()

        prompt = contents if isinstance(contents, str) else str(contents)
        schema = getattr(config, "response_schema", None)
        if not isinstance(schema, dict):
            # 스키마 없이 부르는 경로는 지금 없다. 생기면 더미가 형식을 맞출 수 없으므로
            # 조용히 이상한 값을 주지 않고 드러나게 둔다.
            logger.warning("스텁: response_schema 가 없어 더미 문자열만 돌려준다")
            return _StubGenerateResponse(json.dumps({"stub": STUB_TEXT}), len(prompt))

        payload = _dummy_for(schema, _extract_id_hints(prompt))
        return _StubGenerateResponse(
            json.dumps(payload, ensure_ascii=False),
            len(prompt),
        )


class _StubAio:
    def __init__(self, settings: Settings):
        self.models = _StubModels(settings)


class StubGenaiClient:
    """`genai.Client` 대신 들어간다. 쓰이는 표면은 `client.aio.models.*` 뿐이다."""

    def __init__(self, settings: Settings):
        self.aio = _StubAio(settings)
