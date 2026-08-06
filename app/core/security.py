"""내부 호출 인증.

이 서버는 사용자 토큰을 검증하지 않는다. 사용자 인증은 스프링이 이미 끝냈고,
여기로는 서버가 서버를 부르는 호출만 들어온다. 공유 키 하나로 "우리 서버가 맞는지"만 본다.

전제: 이 서버는 외부에 노출하지 않는다. (VPC 내부 / 방화벽)
키가 유일한 방어선이 되지 않게 네트워크 차단을 함께 건다.
"""

import secrets

from fastapi import Header

from app.core.config import get_settings
from app.core.errors import AiErrorCode, AiException

INTERNAL_API_KEY_HEADER = "X-Internal-Api-Key"


async def verify_internal_caller(
    x_internal_api_key: str | None = Header(default=None, alias=INTERNAL_API_KEY_HEADER),
) -> None:
    expected = get_settings().internal_api_key

    # 문자열 == 비교는 앞자리부터 달라지는 시점에 끝나서 타이밍으로 값을 좁힐 수 있다.
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, expected):
        raise AiException(AiErrorCode.UNAUTHORIZED)
