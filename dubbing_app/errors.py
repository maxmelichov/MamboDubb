"""The one error shape the UI ever sees.

    {"error": {"code": "invalid_request|not_found|busy|internal_error",
               "message": "human readable"}}

After MamboRambo's `write_error`: one envelope, four codes, no exceptions to the
rule — an unhandled Python exception is turned into `internal_error` rather than
leaking a stack trace or FastAPI's own `{"detail": ...}` shape.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

CODES = ("invalid_request", "not_found", "busy", "internal_error")

STATUS = {
    "invalid_request": 400,
    "not_found": 404,
    "busy": 409,
    "internal_error": 500,
}


class ApiError(Exception):
    """Raised anywhere in the server; rendered as the envelope by the handler."""

    def __init__(self, code: str, message: str):
        if code not in STATUS:
            code = "internal_error"
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status(self) -> int:
        return STATUS[self.code]

    def envelope(self) -> dict[str, Any]:
        return envelope(self.code, self.message)


def invalid(message: str) -> ApiError:
    return ApiError("invalid_request", message)


def not_found(message: str) -> ApiError:
    return ApiError("not_found", message)


def busy(message: str) -> ApiError:
    return ApiError("busy", message)


def envelope(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def install(app) -> None:
    """Attach the handlers that make every failure path use the envelope."""
    from fastapi import HTTPException
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    async def api_error(_request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=exc.envelope())

    async def validation_error(_request, exc: RequestValidationError):
        # FastAPI's own 422 body is a list of pydantic dicts; the UI only ever
        # parses one shape, so flatten it into the envelope.
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return JSONResponse(status_code=400,
                            content=envelope("invalid_request", "; ".join(parts) or "bad request"))

    async def http_error(_request, exc: StarletteHTTPException):
        code = {400: "invalid_request", 404: "not_found", 409: "busy"}.get(
            exc.status_code, "internal_error")
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse(status_code=exc.status_code, content=envelope(code, detail))

    async def unhandled(_request, exc: Exception):
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return JSONResponse(status_code=500,
                            content=envelope("internal_error", f"{type(exc).__name__}: {exc}"))

    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(Exception, unhandled)
