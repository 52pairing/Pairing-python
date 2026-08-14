"""Prometheus 메트릭.

모니터링 서버의 Prometheus 가 30초마다 `/metrics` 를 긁어간다. ECS 태스크가 여러 개일 때
각 태스크는 Prometheus 쪽에서 instance/task 라벨로 구분되므로, 여기서 태스크 식별자를
라벨로 넣지 않는다. (넣으면 태스크가 교체될 때마다 시리즈가 새로 생겨 쌓인다)

주의 — 워커를 늘릴 때
    지금은 uvicorn 단일 프로세스라 기본 레지스트리로 충분하다. Dockerfile 의 CMD 에
    `--workers N` 을 붙이면 프로세스마다 별도 레지스트리를 갖게 되고, 스크레이프가 그중
    임의의 워커에 붙어서 값이 1/N 로 보인다. 그때는 prometheus_client 의 멀티프로세스
    모드로 바꿔야 한다 (환경변수 PROMETHEUS_MULTIPROC_DIR + MultiProcessCollector).
    LLM 호출은 I/O 대기라 async 동시성으로 먼저 해결하는 편이 낫다(Dockerfile 주석 참고).
"""

import logging

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP
#
# path 라벨에는 '라우트 템플릿'을 넣는다(/api/v1/embeddings/{id}). 실제 URL 을 그대로 넣으면
# ID 마다 시리즈가 하나씩 생겨서 메모리가 무한히 늘어난다. 모니터링 서버가 t3.small 이라
# 특히 조심해야 한다. 매칭되지 않은 요청(404, 스캐너)은 전부 "unmatched" 로 묶는다.
# ---------------------------------------------------------------------------
http_requests_total = Counter(
    "http_requests_total",
    "HTTP 요청 수",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP 요청 처리 시간",
    ["method", "path"],
    # 이 서버는 요청 안에서 Gemini 를 호출해서 꼬리가 아주 길다. 상한을 120초까지 두는 이유는
    # application.yaml(스프링)의 AI_TIMEOUT_MS 가 120초이기 때문이다. 그보다 짧게 끊으면
    # 타임아웃 직전 구간이 +Inf 로 뭉쳐서 "얼마나 걸려서 잘렸는지"를 못 본다.
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "지금 처리 중인 HTTP 요청 수",
)


# ---------------------------------------------------------------------------
# Gemini 호출
#
# 스프링 쪽 application.yaml 에 AI 호출 타임아웃을 20초 -> 33초 -> 60초 -> 120초로 계속
# 밀어올린 이력과 "40초는 원인 미상"이라는 주석이 남아 있다. 그 원인을 보려면 모델 호출
# 자체의 지연 분포와, 재시도·키 전환이 얼마나 끼어들었는지를 나눠서 봐야 한다.
# 그래서 duration(전체)과 attempts(시도 횟수)를 따로 센다.
# ---------------------------------------------------------------------------
gemini_requests_total = Counter(
    "gemini_requests_total",
    "Gemini 호출 수",
    # outcome:
    #   success   정상 응답
    #   timeout   Gemini 응답이 늦어 우리가 끊었다
    #   cancelled 호출자(스프링)가 먼저 끊었다 — Gemini 잘못이 아니다
    #   error     그 외 실패
    ["task", "model", "outcome"],
)

gemini_request_duration_seconds = Histogram(
    "gemini_request_duration_seconds",
    "Gemini 호출 소요 시간 (재시도·키 전환 포함, 최종 결과까지)",
    ["task", "model"],
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180),
)

gemini_attempts_total = Counter(
    "gemini_attempts_total",
    "Gemini 실제 API 시도 횟수 (재시도와 키 전환을 포함하므로 호출 수보다 크다)",
    ["task", "model"],
)

gemini_tokens_total = Counter(
    "gemini_tokens_total",
    "Gemini 토큰 사용량. 임베딩 호출은 토큰을 주지 않아 집계되지 않는다.",
    # kind: prompt / output
    ["task", "model", "kind"],
)

gemini_key_disabled_total = Counter(
    "gemini_key_disabled_total",
    "쿼터 소진 등으로 키를 쿨다운에 넣은 횟수",
)

gemini_keys_in_cooldown = Gauge(
    "gemini_keys_in_cooldown",
    "지금 쿨다운 중인 Gemini 키 수. 전체 키 수와 같아지면 호출이 모두 실패한다.",
)

gemini_keys_total = Gauge(
    "gemini_keys_total",
    "설정된 Gemini 키 수",
)

gemini_stub_mode = Gauge(
    "gemini_stub_mode",
    "1 이면 Gemini 를 부르지 않고 더미로 응답하는 부하 테스트 모드다. 운영에서 1 이면 사고다.",
)


# ---------------------------------------------------------------------------
# DB 커넥션 풀
#
# 값을 이벤트로 갱신하지 않고 스크레이프 시점에 풀에 직접 물어본다(set_function).
# 이벤트 방식은 훅을 하나 놓치면 그대로 값이 어긋난 채로 남는다.
# ---------------------------------------------------------------------------
db_pool_size = Gauge("db_pool_size", "커넥션 풀이 들고 있는 커넥션 수")
db_pool_checked_out = Gauge("db_pool_checked_out", "지금 대출 중인 커넥션 수")


def bind_db_pool_metrics(engine) -> None:
    """SQLAlchemy 엔진의 풀 상태를 게이지에 연결한다.

    풀 구현체에 따라 메서드가 없을 수 있다. 메트릭 때문에 앱이 죽으면 안 되므로 전부 감싼다.
    """

    def _safe(fn):
        def _get() -> float:
            try:
                return float(fn())
            except Exception:  # noqa: BLE001 - 메트릭은 실패해도 무해해야 한다
                return 0.0

        return _get

    try:
        pool = engine.pool
        db_pool_size.set_function(_safe(pool.size))
        db_pool_checked_out.set_function(_safe(pool.checkedout))
    except AttributeError:
        logger.info("커넥션 풀이 크기 조회를 지원하지 않아 db_pool_* 메트릭을 생략한다")


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------
router = APIRouter(tags=["00. Health"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus 스크레이프 대상.

    내부 키를 요구하지 않는다. Prometheus 에 앱 키를 심고 싶지 않고, 이 서버 자체가 VPC
    내부 전용이라 네트워크로 이미 막혀 있다(app/core/security.py 의 전제와 같다).
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def route_template(request: Request) -> str:
    """라벨에 쓸 경로. 실제 URL 이 아니라 라우트 템플릿을 준다.

    `/api/v1/embeddings/positions/777/candidates` -> `/api/v1/embeddings/positions/{position_id}/candidates`

    실제 URL 을 그대로 넣으면 ID 마다 시리즈가 생겨 메모리가 무한히 늘어난다. 템플릿으로
    묶으면 시리즈 수가 '라우트 개수' 로 고정된다.

    라우팅이 끝난 뒤에 scope["route"] 가 채워지므로 응답을 받은 다음에 불러야 한다.
    매칭 실패(404)면 템플릿이 없다 — 스캐너가 만드는 무한한 경로를 "unmatched" 하나로 묶는다.

    prefix 를 되붙이는 이유
        이 프로젝트의 FastAPI(0.141) 는 include_router 로 붙인 하위 라우터를 펼치지 않고
        그대로 들고 있어서, scope["route"].path 가 '하위 라우터 기준 경로' 만 준다.
        위 예시에서 `/embeddings/positions/{position_id}/candidates` 만 나오고 `/api/v1` 이
        빠진다. 그래서 실제 URL 의 앞쪽 세그먼트를 세어 붙여준다.
        (scope 의 root_path / route_root_path 로는 이 값을 얻을 수 없다 — 확인함)
    """
    route = request.scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if not template:
        return "unmatched"

    actual_segments = request.url.path.strip("/").split("/")
    template_segments = template.strip("/").split("/")

    # 세그먼트 수가 맞아야 앞쪽을 prefix 로 떼어낼 수 있다. `path:` 변환자처럼 파라미터
    # 하나가 여러 세그먼트를 먹는 경우는 수가 어긋나므로, 그때는 템플릿을 그대로 쓴다.
    missing = len(actual_segments) - len(template_segments)
    if missing <= 0:
        return template

    return "/" + "/".join(actual_segments[:missing] + template_segments)
