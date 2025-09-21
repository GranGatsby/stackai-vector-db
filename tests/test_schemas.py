"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.core.constants import HEALTH_STATUS, HEALTH_MESSAGE
from app.schemas.health import HealthResponse


class TestHealthResponse:
    """Test cases for HealthResponse schema."""

    def test_health_response_valid_data(self):
        """Test HealthResponse with valid data."""
        response = HealthResponse(
            status=HEALTH_STATUS,
            message=HEALTH_MESSAGE
        )
        
        assert response.status == HEALTH_STATUS
        assert response.message == HEALTH_MESSAGE

    def test_health_response_model_validation(self):
        """Test HealthResponse model validation."""
        # Valid data
        valid_data = {
            "status": HEALTH_STATUS,
            "message": HEALTH_MESSAGE
        }
        response = HealthResponse(**valid_data)
        assert response.status == HEALTH_STATUS
        assert response.message == HEALTH_MESSAGE

    def test_health_response_missing_fields(self):
        """Test HealthResponse validation with missing fields."""
        # Missing status
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(message=HEALTH_MESSAGE)
        assert "status" in str(exc_info.value)
        
        # Missing message
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(status=HEALTH_STATUS)
        assert "message" in str(exc_info.value)

    def test_health_response_extra_fields_forbidden(self):
        """Test that extra fields are forbidden in HealthResponse."""
        # In Pydantic v2 with strict=True, extra fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse.model_validate({
                "status": HEALTH_STATUS,
                "message": HEALTH_MESSAGE,
                "extra_field": "not_allowed"
            })
        # Should contain information about forbidden extra field
        error_str = str(exc_info.value)
        assert "extra_field" in error_str or "Extra inputs are not permitted" in error_str

    def test_health_response_empty_strings(self):
        """Test HealthResponse with empty strings."""
        response = HealthResponse(
            status="",
            message=""
        )
        
        assert response.status == ""
        assert response.message == ""

    def test_health_response_serialization(self):
        """Test HealthResponse serialization to dict."""
        response = HealthResponse(
            status=HEALTH_STATUS,
            message=HEALTH_MESSAGE
        )
        
        data = response.model_dump()
        expected = {
            "status": HEALTH_STATUS,
            "message": HEALTH_MESSAGE
        }
        
        assert data == expected

    def test_health_response_json_serialization(self):
        """Test HealthResponse JSON serialization."""
        response = HealthResponse(
            status=HEALTH_STATUS,
            message=HEALTH_MESSAGE
        )
        
        json_str = response.model_dump_json()
        assert f'"status":"{HEALTH_STATUS}"' in json_str
        assert f'"message":"{HEALTH_MESSAGE}"' in json_str

    def test_health_response_from_dict(self):
        """Test creating HealthResponse from dictionary."""
        data = {
            "status": "degraded",
            "message": "Some services are experiencing issues"
        }
        
        response = HealthResponse.model_validate(data)
        assert response.status == "degraded"
        assert response.message == "Some services are experiencing issues"
