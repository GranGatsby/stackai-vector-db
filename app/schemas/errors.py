"""Error response schemas for the API."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Individual error detail."""

    model_config = ConfigDict(strict=True, extra="forbid")

    code: str
    message: str
    field: str | None = None
    context: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Standard error response format."""

    model_config = ConfigDict(strict=True, extra="forbid")

    error: ErrorDetail

    @classmethod
    def from_domain_error(cls, error) -> "ErrorResponse":
        return cls(
            error=ErrorDetail(
                code=getattr(error, "code", error.__class__.__name__),
                message=str(error),
                context=getattr(error, "context", None),
            )
        )

    @classmethod
    def validation_error(
        cls, message: str, field: str | None = None
    ) -> "ErrorResponse":
        return cls(
            error=ErrorDetail(code="VALIDATION_ERROR", message=message, field=field)
        )

    @classmethod
    def not_found(cls, resource: str, identifier: str) -> "ErrorResponse":
        return cls(
            error=ErrorDetail(
                code="NOT_FOUND",
                message=f"{resource} with identifier '{identifier}' not found",
            )
        )

    @classmethod
    def conflict(cls, message: str) -> "ErrorResponse":
        return cls(error=ErrorDetail(code="CONFLICT", message=message))

    @classmethod
    def internal_error(cls, message: str = "Internal server error") -> "ErrorResponse":
        return cls(error=ErrorDetail(code="INTERNAL_ERROR", message=message))
