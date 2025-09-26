"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.health import HealthResponse


class TestHealthResponse:
    """Test cases for HealthResponse schema."""

    def test_health_response_valid_data(self):
        """Test HealthResponse with valid data."""
        response = HealthResponse(
            status=settings.health_status, message=settings.health_message
        )

        assert response.status == settings.health_status
        assert response.message == settings.health_message

    def test_health_response_model_validation(self):
        """Test HealthResponse model validation."""
        # Valid data
        valid_data = {
            "status": settings.health_status,
            "message": settings.health_message,
        }
        response = HealthResponse(**valid_data)
        assert response.status == settings.health_status
        assert response.message == settings.health_message

    def test_health_response_missing_fields(self):
        """Test HealthResponse validation with missing fields."""
        # Missing status
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(message=settings.health_message)
        assert "status" in str(exc_info.value)

        # Missing message
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(status=settings.health_status)
        assert "message" in str(exc_info.value)

    def test_health_response_extra_fields_forbidden(self):
        """Test that extra fields are forbidden in HealthResponse."""
        # In Pydantic v2 with strict=True, extra fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse.model_validate(
                {
                    "status": settings.health_status,
                    "message": settings.health_message,
                    "extra_field": "not_allowed",
                }
            )
        # Should contain information about forbidden extra field
        error_str = str(exc_info.value)
        assert (
            "extra_field" in error_str or "Extra inputs are not permitted" in error_str
        )

    def test_health_response_empty_strings(self):
        """Test HealthResponse with empty strings."""
        response = HealthResponse(status="", message="")

        assert response.status == ""
        assert response.message == ""

    def test_health_response_serialization(self):
        """Test HealthResponse serialization to dict."""
        response = HealthResponse(
            status=settings.health_status, message=settings.health_message
        )

        data = response.model_dump()
        expected = {
            "status": settings.health_status,
            "message": settings.health_message,
        }

        assert data == expected

    def test_health_response_json_serialization(self):
        """Test HealthResponse JSON serialization."""
        response = HealthResponse(
            status=settings.health_status, message=settings.health_message
        )

        json_str = response.model_dump_json()
        assert f'"status":"{settings.health_status}"' in json_str
        assert f'"message":"{settings.health_message}"' in json_str

    def test_health_response_from_dict(self):
        """Test creating HealthResponse from dictionary."""
        data = {
            "status": "degraded",
            "message": "Some services are experiencing issues",
        }

        response = HealthResponse.model_validate(data)
        assert response.status == "degraded"
        assert response.message == "Some services are experiencing issues"
