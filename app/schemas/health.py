"""Health check schemas."""

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health check response schema."""

    model_config = ConfigDict(strict=True, extra="forbid")

    status: str
    message: str
