"""Tests for health check endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_success():
    """Test health check endpoint returns correct response."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "Vector DB API is running"


def test_health_endpoint_has_request_id_header():
    """Test that health check includes request ID in response headers."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    # Should be a valid UUID format
    request_id = response.headers["x-request-id"]
    assert len(request_id.split("-")) == 5  # UUID format check


def test_health_endpoint_with_custom_request_id():
    """Test health check with custom request ID in headers."""
    custom_request_id = "custom-test-id-123"
    response = client.get(
        "/api/v1/health",
        headers={"x-request-id": custom_request_id}
    )
    
    assert response.status_code == 200
    assert response.headers["x-request-id"] == custom_request_id
