"""Tests for health check endpoints."""

import uuid

import pytest
from fastapi import status

from app.core.constants import HEALTH_MESSAGE, HEALTH_STATUS


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_endpoint_success(self, client):
        """Test health check endpoint returns correct response."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == HEALTH_STATUS
        assert data["message"] == HEALTH_MESSAGE

    def test_health_endpoint_has_request_id_header(self, client):
        """Test that health check includes request ID in response headers."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        assert "x-request-id" in response.headers
        # Should be a valid UUID format
        request_id = response.headers["x-request-id"]
        assert uuid.UUID(request_id)  # UUID format check

    def test_health_endpoint_with_custom_request_id(self, client):
        """Test health check with custom request ID in headers."""
        custom_request_id = "custom-test-id-123"
        response = client.get(
            "/api/v1/health", headers={"x-request-id": custom_request_id}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["x-request-id"] == custom_request_id

    def test_health_endpoint_response_model(self, client):
        """Test that health endpoint uses correct response model."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure matches HealthResponse schema
        assert "status" in data
        assert "message" in data
        assert len(data) == 2  # Only these two fields should be present

        # Verify field types
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)

    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    def test_health_methods_not_allowed(self, client, method):
        """Test that non-GET methods return 405 Method Not Allowed."""
        response = getattr(client, method)("/api/v1/health")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
