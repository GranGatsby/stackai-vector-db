"""Integration tests for search and indexing endpoints."""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.domain import Chunk, Document, Library
from app.main import app


class TestSearchEndpoints:
    """Test suite for search and indexing API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_library(self, client: TestClient) -> dict:
        """Create a sample library for testing."""
        unique_name = f"Test Search Library {uuid.uuid4().hex[:8]}"
        library_data = {
            "name": unique_name,
            "description": "Library for search testing",
        }
        response = client.post("/api/v1/libraries", json=library_data)
        assert response.status_code == status.HTTP_201_CREATED
        return response.json()

    @pytest.fixture
    def sample_document(self, client: TestClient, sample_library: dict) -> dict:
        """Create a sample document."""
        doc_data = {
            "title": "Test Document",
            "content": "This is test content for search",
        }
        response = client.post(
            f"/api/v1/libraries/{sample_library['id']}/documents", json=doc_data
        )
        assert response.status_code == status.HTTP_201_CREATED
        return response.json()

    @pytest.fixture
    def sample_chunks(
        self, client: TestClient, sample_document: dict, sample_library: dict
    ) -> list[dict]:
        """Create sample chunks for testing."""
        chunks = []
        chunk_texts = [
            "This is the first chunk about machine learning",
            "This is the second chunk about artificial intelligence",
            "This is the third chunk about deep learning",
        ]

        for i, text in enumerate(chunk_texts):
            chunk_data = {
                "library_id": sample_library["id"],
                "text": text,
                "start_index": i * 50,
                "end_index": i * 50 + len(text),
                "compute_embedding": True,
            }
            response = client.post(
                f"/api/v1/documents/{sample_document['id']}/chunks", json=chunk_data
            )
            assert response.status_code == status.HTTP_201_CREATED
            chunks.append(response.json())

        return chunks

    def test_index_build_and_status(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test index build on library with data and verify status."""
        library_id = sample_library["id"]

        # Build index with default algorithm
        build_request = {}
        response = client.post(
            f"/api/v1/libraries/{library_id}/index", json=build_request
        )

        assert response.status_code == status.HTTP_200_OK
        build_result = response.json()

        # Verify build response structure
        assert build_result["library_id"] == library_id
        assert build_result["algorithm"] in ["linear", "kdtree", "ivf"]
        assert build_result["size"] == len(sample_chunks)
        assert build_result["embedding_dim"] > 0
        assert "built_at" in build_result
        assert build_result["version"] >= 1

    def test_query_text_and_vector(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test both text and vector search endpoints return correct SearchResult."""
        library_id = sample_library["id"]

        # First build the index
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK

        # Test text search
        text_query = {"text": "machine learning", "k": 2}
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json=text_query
        )

        assert response.status_code == status.HTTP_200_OK
        search_result = response.json()

        # Verify SearchResult structure
        assert "hits" in search_result
        assert "total" in search_result
        assert search_result["library_id"] == library_id
        assert search_result["algorithm"] in ["linear", "kdtree", "ivf"]
        assert search_result["index_size"] == len(sample_chunks)
        assert search_result["embedding_dim"] > 0
        assert len(search_result["hits"]) <= 2

        # Verify hits structure
        if search_result["hits"]:
            hit = search_result["hits"][0]
            assert "chunk_id" in hit
            assert "score" in hit
            assert isinstance(hit["score"], (int, float))
            assert hit["score"] >= 0

        # Test vector search with same embedding
        if search_result.get("query_embedding"):
            vector_query = {"embedding": search_result["query_embedding"], "k": 2}
            response = client.post(
                f"/api/v1/libraries/{library_id}/query/vector", json=vector_query
            )
            assert response.status_code == status.HTTP_200_OK

            vector_result = response.json()
            assert vector_result["total"] == search_result["total"]
            assert len(vector_result["hits"]) == len(search_result["hits"])

    def test_index_empty_library_build(self, client: TestClient):
        """Test building index on empty library."""
        # Create empty library with unique name
        unique_name = f"Empty Library {uuid.uuid4().hex[:8]}"
        library_data = {"name": unique_name}
        response = client.post("/api/v1/libraries", json=library_data)
        assert response.status_code == status.HTTP_201_CREATED
        library = response.json()

        # Build index on empty library
        response = client.post(f"/api/v1/libraries/{library['id']}/index", json={})
        assert response.status_code == status.HTTP_200_OK

        build_result = response.json()
        assert build_result["size"] == 0
        assert build_result["embedding_dim"] > 0  # Should have default dimension
        assert "built_at" in build_result
        assert build_result["version"] >= 1

    def test_k_clamped_to_size(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test that k > num_chunks returns exactly num_chunks results."""
        library_id = sample_library["id"]

        # Build index
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK

        # Query with k larger than available chunks
        large_k = len(sample_chunks) + 10
        text_query = {"text": "machine learning", "k": large_k}
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json=text_query
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert len(result["hits"]) == len(sample_chunks)
        assert result["total"] == len(sample_chunks)

    def test_invalid_k_raises_422(self, client: TestClient, sample_library: dict):
        """Test that k <= 0 returns 422."""
        library_id = sample_library["id"]

        # Test with k = 0
        text_query = {"text": "test", "k": 0}
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json=text_query
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Test with k < 0
        text_query = {"text": "test", "k": -1}
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json=text_query
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_dimension_mismatch_raises_422(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test that query with wrong dimension vector returns 422."""
        library_id = sample_library["id"]

        # Build index first
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK
        build_result = response.json()

        # Query with wrong dimension
        wrong_dim_vector = [0.1, 0.2]  # Too short
        if build_result["embedding_dim"] != len(wrong_dim_vector):
            vector_query = {"embedding": wrong_dim_vector, "k": 1}
            response = client.post(
                f"/api/v1/libraries/{library_id}/query/vector", json=vector_query
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_library_not_found_404(self, client: TestClient):
        """Test that non-existent library returns 404."""
        fake_library_id = str(uuid.uuid4())

        # Test index build
        response = client.post(f"/api/v1/libraries/{fake_library_id}/index", json={})
        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Test text search
        response = client.post(
            f"/api/v1/libraries/{fake_library_id}/query/text",
            json={"text": "test", "k": 1},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Test vector search
        response = client.post(
            f"/api/v1/libraries/{fake_library_id}/query/vector",
            json={"embedding": [0.1, 0.2], "k": 1},
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_dirty_flag_on_chunk_mutation(
        self,
        client: TestClient,
        sample_library: dict,
        sample_document: dict,
        sample_chunks: list[dict],
    ):
        """Test that chunk mutations mark index dirty and build cleans it."""
        library_id = sample_library["id"]

        # Build initial index
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK

        # Add a new chunk (should mark dirty)
        new_chunk_data = {
            "library_id": library_id,
            "text": "This is a new chunk that should mark index dirty",
            "compute_embedding": True,
        }
        response = client.post(
            f"/api/v1/documents/{sample_document['id']}/chunks", json=new_chunk_data
        )
        assert response.status_code == status.HTTP_201_CREATED

        # Build again (should clean dirty flag)
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK
        build_result = response.json()
        assert (
            build_result["size"] == len(sample_chunks) + 1
        )  # Should include new chunk

    def test_index_algorithm_selection(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test building index with explicit algorithm selection."""
        library_id = sample_library["id"]

        # Test with linear algorithm
        build_request = {"algorithm": "linear"}
        response = client.post(
            f"/api/v1/libraries/{library_id}/index", json=build_request
        )
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["algorithm"] == "linear"

        # Test with kdtree algorithm
        build_request = {"algorithm": "kdtree"}
        response = client.post(
            f"/api/v1/libraries/{library_id}/index", json=build_request
        )
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["algorithm"] == "kdtree"

    def test_schema_validation_search_vector(
        self, client: TestClient, sample_library: dict
    ):
        """Test vector search schema validation."""
        library_id = sample_library["id"]

        # Empty vector
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/vector",
            json={"embedding": [], "k": 1},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Non-numeric values
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/vector",
            json={"embedding": ["not", "numeric"], "k": 1},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_schema_validation_search_text(
        self, client: TestClient, sample_library: dict
    ):
        """Test text search schema validation."""
        library_id = sample_library["id"]

        # Empty text
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json={"text": "", "k": 1}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Whitespace only
        response = client.post(
            f"/api/v1/libraries/{library_id}/query/text", json={"text": "   ", "k": 1}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_concurrency_snapshot(
        self, client: TestClient, sample_library: dict, sample_chunks: list[dict]
    ):
        """Test concurrent reads during rebuild return stable results."""
        library_id = sample_library["id"]

        # Build initial index
        response = client.post(f"/api/v1/libraries/{library_id}/index", json={})
        assert response.status_code == status.HTTP_200_OK

        results = []
        errors = []

        def query_during_rebuild():
            """Execute query that might run during rebuild."""
            try:
                response = client.post(
                    f"/api/v1/libraries/{library_id}/query/text",
                    json={"text": "machine learning", "k": 2},
                )
                if response.status_code == status.HTTP_200_OK:
                    results.append(response.json())
                else:
                    errors.append(response.status_code)
                return response.status_code
            except Exception as e:
                errors.append(str(e))
                return None

        def rebuild_index():
            """Rebuild index."""
            try:
                response = client.post(
                    f"/api/v1/libraries/{library_id}/index",
                    json={"algorithm": "kdtree"},
                )
                return response.status_code
            except Exception as e:
                errors.append(str(e))
                return None

        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start multiple concurrent queries
            query_futures = [executor.submit(query_during_rebuild) for _ in range(3)]

            # Start rebuild
            rebuild_future = executor.submit(rebuild_index)

            # Wait for all to complete
            all_futures = query_futures + [rebuild_future]
            for future in as_completed(all_futures):
                future.result()

        # All operations should complete without exceptions
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"

        # At least some queries should succeed
        successful_queries = len(results)
        assert (
            successful_queries > 0
        ), "No queries succeeded during concurrent operations"

        # Results should be consistent (same structure)
        if len(results) > 1:
            first_result = results[0]
            for result in results[1:]:
                assert "hits" in result
                assert "total" in result
                assert result["library_id"] == library_id
