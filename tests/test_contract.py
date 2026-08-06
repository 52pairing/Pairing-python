"""스프링과의 계약을 지키는지 확인한다.

응답 형식과 인증 규칙이 깨지면 백엔드/프론트가 동시에 깨지므로 여기서 막는다.
"""

from app.core.context import TRACE_ID_HEADER
from app.core.security import INTERNAL_API_KEY_HEADER


def test_health_is_open_and_returns_common_success_shape(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    # 스프링 ApiResponse 와 같은 필드여야 한다.
    assert set(body) == {"timestamp", "status", "code", "message", "data"}
    assert body["code"] == "HEALTHY"


def test_internal_api_requires_key(client):
    response = client.post("/api/v1/matchings/recommendations", json={})

    assert response.status_code == 401
    body = response.json()
    # 스프링 ErrorResponse 와 같은 필드여야 한다.
    assert set(body) == {"timestamp", "status", "errorCode", "message", "traceId"}
    assert body["errorCode"] == "AI_003"


def test_validation_error_returns_field_name(client):
    response = client.post(
        "/api/v1/matchings/recommendations",
        json={"position_id": 0, "recruit_count": 1},
        headers={INTERNAL_API_KEY_HEADER: "test-internal-key"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["errorCode"] == "AI_002"
    assert "position_id" in body["message"]


def test_trace_id_from_spring_is_echoed_back(client):
    response = client.get("/health", headers={TRACE_ID_HEADER: "abc12345"})

    # 스프링 로그와 AI 서버 로그를 같은 ID 로 맞출 수 있어야 한다.
    assert response.headers[TRACE_ID_HEADER] == "abc12345"


def test_trace_id_is_generated_when_absent(client):
    response = client.get("/health")

    assert response.headers[TRACE_ID_HEADER]
