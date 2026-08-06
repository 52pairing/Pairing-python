"""테스트 공통 설정.

실제 DB/Gemini 없이도 앱이 뜨도록 환경변수를 먼저 주입한다.
"""

import os

import pytest

os.environ.setdefault("AI_DB_URL", "postgresql+asyncpg://pairing:pairing@localhost:5432/pairing")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
