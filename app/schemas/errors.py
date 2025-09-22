"""Error response schemas for the API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Individual error detail."""

    model_config = ConfigDict(strict=True, extra="forbid")

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: str | None = Field(None, description="Field name if error is field-specific")
    context: dict[str, Any] | None = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response format."""

    model_config = ConfigDict(strict=True, extra="forbid")

    error: ErrorDetail = Field(..., description="Error details")

    @classmethod
    def from_domain_error(cls, error) -> "ErrorResponse":
        """Create an ErrorResponse from a domain error.

        Args:
            error: The domain error exception

        Returns:
            An ErrorResponse schema instance
        """
        return cls(
            error=ErrorDetail(
                code=getattr(error, "code", error.__class__.__name__),
                message=str(error),
                context=getattr(error, "context", None),
            )
        )

    @classmethod
    def validation_error(cls, message: str, field: str = None) -> "ErrorResponse":
        """Create a validation error response.

        Args:
            message: The validation error message
            field: The field that failed validation (optional)

        Returns:
            An ErrorResponse for validation errors
        """
        return cls(
            error=ErrorDetail(code="VALIDATION_ERROR", message=message, field=field)
        )

    @classmethod
    def not_found(cls, resource: str, identifier: str) -> "ErrorResponse":
        """Create a not found error response.

        Args:
            resource: The type of resource (e.g., "Library")
            identifier: The identifier that wasn't found

        Returns:
            An ErrorResponse for not found errors
        """
        return cls(
            error=ErrorDetail(
                code="NOT_FOUND",
                message=f"{resource} with identifier '{identifier}' not found",
            )
        )

    @classmethod
    def conflict(cls, message: str) -> "ErrorResponse":
        """Create a conflict error response.

        Args:
            message: The conflict error message

        Returns:
            An ErrorResponse for conflict errors
        """
        return cls(error=ErrorDetail(code="CONFLICT", message=message))

    @classmethod
    def internal_error(cls, message: str = "Internal server error") -> "ErrorResponse":
        """Create an internal server error response.

        Args:
            message: The error message

        Returns:
            An ErrorResponse for internal server errors
        """
        return cls(error=ErrorDetail(code="INTERNAL_ERROR", message=message))
