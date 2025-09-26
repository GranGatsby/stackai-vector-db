"""Integration tests for Chunks API endpoints."""

import uuid

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app


class TestChunksAPI:
    """Test suite for Chunks API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_library(self, client: TestClient) -> dict:
        """Create a sample library for testing."""
        # Use unique name to avoid conflicts across tests
        library_name = f"Test Library Chunks {uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/v1/libraries/",
            json={"name": library_name, "description": "For chunk testing"},
        )
        assert response.status_code == status.HTTP_201_CREATED
        return response.json()

    @pytest.fixture
    def sample_document(self, client: TestClient, sample_library: dict) -> dict:
        """Create a sample document for testing."""
        response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents",
            json={
                "title": "Test Document",
                "content": "This is test content for chunks",
            },
        )
        assert response.status_code == status.HTTP_201_CREATED
        return response.json()

    def test_create_chunks_success(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ):
        """Test successful chunk creation in a document."""
        chunks_data = {
            "chunks": [
                {
                    "text": "First test chunk",
                    "start_index": 0,
                    "end_index": 16,
                    "metadata": {"chunk_type": "paragraph"},
                },
                {
                    "text": "Second test chunk",
                    "start_index": 17,
                    "end_index": 34,
                    "metadata": {"chunk_type": "paragraph"},
                },
            ],
            "compute_embedding": False,
        }

        response = client.post(
            f"/api/v1/documents/{sample_document['id']}/chunks", json=chunks_data
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["total_created"] == 2
        assert len(data["chunks"]) == 2

        # Verify first chunk
        chunk1 = data["chunks"][0]
        assert chunk1["text"] == "First test chunk"
        assert chunk1["document_id"] == sample_document["id"]
        assert chunk1["library_id"] == sample_library["id"]
        assert chunk1["start_index"] == 0
        assert chunk1["end_index"] == 16
        assert chunk1["metadata"]["chunk_type"] == "paragraph"

    def test_get_chunk_success(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ):
        """Test successful chunk retrieval by ID."""
        # Create chunk first
        create_response = client.post(
            f"/api/v1/documents/{sample_document['id']}/chunks",
            json={
                "chunks": [{"text": "Chunk to retrieve", "start_index": 0, "end_index": 17}],
                "compute_embedding": False,
            },
        )
        created_chunk = create_response.json()["chunks"][0]

        # Get chunk
        response = client.get(f"/api/v1/chunks/{created_chunk['id']}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_chunk["id"]
        assert data["text"] == "Chunk to retrieve"
        assert data["document_id"] == sample_document["id"]
        assert data["library_id"] == sample_library["id"]

    def test_get_chunk_not_found(self, client: TestClient):
        """Test chunk retrieval with non-existent ID."""
        non_existent_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/chunks/{non_existent_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["error"]["message"].lower()

    def test_update_chunk_success(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ):
        """Test successful chunk update."""
        # Create chunk first
        create_response = client.post(
            f"/api/v1/documents/{sample_document['id']}/chunks",
            json={
                "chunks": [{"text": "Original chunk text", "start_index": 0, "end_index": 19}],
                "compute_embedding": False,
            },
        )
        created_chunk = create_response.json()["chunks"][0]

        # Update chunk
        update_data = {
            "text": "Updated chunk text",
            "start_index": 5,
            "end_index": 23,
            "metadata": {"chunk_type": "updated"},
            "compute_embedding": False,
        }
        response = client.patch(f"/api/v1/chunks/{created_chunk['id']}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == created_chunk["id"]
        assert data["text"] == "Updated chunk text"
        assert data["start_index"] == 5
        assert data["end_index"] == 23
        assert data["metadata"]["chunk_type"] == "updated"

    def test_delete_chunk_success(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ):
        """Test successful chunk deletion."""
        # Create chunk first
        create_response = client.post(
            f"/api/v1/documents/{sample_document['id']}/chunks",
            json={
                "chunks": [{"text": "Chunk to delete", "start_index": 0, "end_index": 15}],
                "compute_embedding": False,
            },
        )
        created_chunk = create_response.json()["chunks"][0]

        # Delete chunk
        response = client.delete(f"/api/v1/chunks/{created_chunk['id']}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify chunk is gone
        get_response = client.get(f"/api/v1/chunks/{created_chunk['id']}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_chunks_by_document(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ):
        """Test listing chunks by document with pagination."""
        # Create multiple chunks
        chunks_data = {
            "chunks": [
                {"text": f"Chunk {i:02d}", "start_index": i * 10, "end_index": i * 10 + 8}
                for i in range(3)
            ],
            "compute_embedding": False,
        }
        client.post(f"/api/v1/documents/{sample_document['id']}/chunks", json=chunks_data)

        # List all chunks
        response = client.get(f"/api/v1/documents/{sample_document['id']}/chunks")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert len(data["chunks"]) == 3
        assert data["limit"] is None
        assert data["offset"] == 0

        # Test pagination
        paginated_response = client.get(
            f"/api/v1/documents/{sample_document['id']}/chunks?limit=2&offset=1"
        )
        assert paginated_response.status_code == status.HTTP_200_OK
        paginated_data = paginated_response.json()
        assert paginated_data["total"] == 3
        assert len(paginated_data["chunks"]) == 2
        assert paginated_data["limit"] == 2
        assert paginated_data["offset"] == 1
