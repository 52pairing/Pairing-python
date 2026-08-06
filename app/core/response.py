"""응답 형식.

스프링의 ApiResponse / ErrorResponse 와 필드를 똑같이 맞춘다.
프론트와 스프링이 두 서버의 응답을 같은 코드로 파싱할 수 있어야 한다.

    성공: {timestamp, status, code, message, data}
    실패: {timestamp, status, errorCode, message, traceId}
"""

from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(UTC)


class ApiResponse(BaseModel, Generic[T]):
    timestamp: datetime = Field(default_factory=_now)
    status: int = 200
    code: str
    message: str
    data: T | None = None

    @classmethod
    def success(cls, code: str, message: str, data: T | None = None) -> "ApiResponse[T]":
        return cls(status=200, code=code, message=message, data=data)

    @classmethod
    def created(cls, code: str, message: str, data: T | None = None) -> "ApiResponse[T]":
        return cls(status=201, code=code, message=message, data=data)


class ErrorResponse(BaseModel):
    timestamp: datetime = Field(default_factory=_now)
    status: int
    errorCode: str  # noqa: N815 - 스프링 응답 필드명과 맞춘다
    message: str
    traceId: str  # noqa: N815
