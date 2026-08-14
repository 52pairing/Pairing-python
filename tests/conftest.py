"""테스트 공통 설정.

실제 DB/Gemini 없이도 앱이 뜨도록 환경변수를 먼저 주입한다.
"""

import os

import pytest

os.environ.setdefault("AI_DB_URL", "postgresql+asyncpg://pairing:pairing@localhost:5432/pairing")
os.environ.setdefault("INTERNAL_API_KEY", "test-internal-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

# 스텁 모드는 테스트에서 **항상 끈다.** setdefault 가 아니라 강제로 덮어쓴다.
#
# 부하 테스트를 하려고 .env 에 AI_STUB_MODE=true 를 넣어 두면, Settings 가 그 파일을 읽어서
# 테스트에서도 스텁이 켜진다. 그러면 키 전환·재시도 테스트가 진짜 클라이언트 대신 더미를
# 상대하게 되어 4개가 깨진다(실측 확인). 원인이 .env 라는 걸 알아채기 어려운 실패다.
#
# 스텁 자체를 시험할 때는 Settings(ai_stub_mode=True) 로 직접 넘긴다 — 생성자 인자가
# 환경변수보다 우선하므로 이 줄에 막히지 않는다.
os.environ["AI_STUB_MODE"] = "false"


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
