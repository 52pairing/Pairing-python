"""에러 코드와 예외 처리.

스프링이 도메인별 접두사(AU_, AC_, TM_)를 쓰는 것과 같은 규칙으로 AI_ 를 쓴다.
프론트/스프링은 errorCode 로 분기하고 message 는 그대로 노출해도 되게 쓴다.
"""

import logging
from enum import Enum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.context import get_trace_id
from app.core.response import ErrorResponse

logger = logging.getLogger(__name__)


class AiErrorCode(Enum):
    """(HTTP 상태, 코드, 기본 메시지)"""

    SERVER_ERROR = (500, "AI_001", "AI 서버 내부 오류가 발생했습니다.")
    INVALID_REQUEST = (400, "AI_002", "잘못된 요청입니다.")
    UNAUTHORIZED = (401, "AI_003", "내부 호출 인증에 실패했습니다.")
    NOT_FOUND = (404, "AI_004", "대상을 찾을 수 없습니다.")

    EMBEDDING_FAILED = (502, "AI_010", "임베딩 생성에 실패했습니다.")
    EMBEDDING_NOT_FOUND = (404, "AI_011", "임베딩이 아직 생성되지 않았습니다.")
    LLM_CALL_FAILED = (502, "AI_012", "LLM 호출에 실패했습니다.")
    LLM_TIMEOUT = (504, "AI_013", "LLM 응답이 지연되었습니다.")
    LLM_RESPONSE_INVALID = (502, "AI_014", "LLM 응답 형식이 올바르지 않습니다.")

    CANDIDATE_POOL_EMPTY = (404, "AI_020", "추천할 후보가 없습니다.")
    SPRING_CALL_FAILED = (502, "AI_030", "백엔드 서버 호출에 실패했습니다.")

    @property
    def status(self) -> int:
        return self.value[0]

    @property
    def code(self) -> str:
        return self.value[1]

    @property
    def message(self) -> str:
        return self.value[2]


class AiException(Exception):
    """비즈니스 예외. 스프링의 BusinessException 과 같은 자리다."""

    def __init__(self, error_code: AiErrorCode, message: str | None = None):
        self.error_code = error_code
        self.message = message or error_code.message
        super().__init__(self.message)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        status=status_code, errorCode=code, message=message, traceId=get_trace_id()
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AiException)
    async def handle_ai_exception(_: Request, exc: AiException) -> JSONResponse:
        logger.warning("[AiException] %s - %s", exc.error_code.code, exc.message)
        return _error_response(exc.error_code.status, exc.error_code.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # 어떤 필드가 왜 틀렸는지 남긴다. 스프링의 "phone: 형식이 올바르지 않습니다" 와 같은 모양.
        details = ", ".join(
            f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in exc.errors()
        )
        logger.warning("[ValidationError] %s", details)
        return _error_response(
            AiErrorCode.INVALID_REQUEST.status,
            AiErrorCode.INVALID_REQUEST.code,
            details or AiErrorCode.INVALID_REQUEST.message,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # 내부 예외 상세는 응답에 담지 않는다. 추적은 traceId 로 한다.
        logger.exception("[UnexpectedError] %s", exc)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            AiErrorCode.SERVER_ERROR.code,
            AiErrorCode.SERVER_ERROR.message,
        )
