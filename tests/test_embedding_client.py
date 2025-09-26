"""Tests for embedding client functionality.

This module contains essential tests for the embedding client implementations,
focusing on the core functionality and error handling.
"""

import pytest

from app.clients.embedding import (
    EmbeddingError,
    FakeEmbeddingClient,
    create_embedding_client,
)


class TestFakeEmbeddingClient:
    """Test the fake embedding client."""

    def test_embed_text_returns_deterministic_embedding(self):
        """Test that fake client returns deterministic embeddings."""
        client = FakeEmbeddingClient(embedding_dim=10)

        text = "This is a test text"
        result1 = client.embed_text(text)
        result2 = client.embed_text(text)

        # Should be deterministic
        assert result1 == result2
        assert len(result1.single_embedding) == 10
        assert all(isinstance(x, float) for x in result1.single_embedding)
        assert result1.model_name == "fake-embedding-model"
        assert result1.embedding_dim == 10

    def test_embed_text_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        client = FakeEmbeddingClient(embedding_dim=10)

        result1 = client.embed_text("Hello world")
        result2 = client.embed_text("Goodbye world")

        # Should be different
        assert result1.single_embedding != result2.single_embedding

    def test_embed_text_empty_text_raises_error(self):
        """Test that empty text raises an error."""
        client = FakeEmbeddingClient()

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            client.embed_text("")

        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            client.embed_text("   ")

    def test_embed_texts_batch_processing(self):
        """Test batch embedding processing."""
        client = FakeEmbeddingClient(embedding_dim=5)

        texts = ["First text", "Second text", "Third text"]
        result = client.embed_texts(texts)

        assert len(result.embeddings) == 3
        assert result.model_name == "fake-embedding-model"
        assert result.embedding_dim == 5
        assert all(len(emb) == 5 for emb in result.embeddings)

        # Each should be different
        assert result.embeddings[0] != result.embeddings[1]
        assert result.embeddings[1] != result.embeddings[2]

    def test_embedding_dim_property(self):
        """Test embedding dimension property."""
        client = FakeEmbeddingClient(embedding_dim=128)
        assert client.embedding_dim == 128


class TestCreateEmbeddingClient:
    """Test the embedding client factory function."""

    def test_create_client_without_api_key_returns_fake(self):
        """Test that factory returns fake client when no API key is provided."""
        # Override settings temporarily by passing empty string
        client = create_embedding_client(api_key="")
        assert isinstance(client, FakeEmbeddingClient)

    def test_create_client_with_empty_api_key_returns_fake(self):
        """Test that factory returns fake client with empty API key."""
        client = create_embedding_client(api_key="")
        assert isinstance(client, FakeEmbeddingClient)

        client = create_embedding_client(api_key="   ")
        assert isinstance(client, FakeEmbeddingClient)

    def test_create_client_with_valid_api_key_returns_cohere(self):
        """Test that factory returns Cohere client with valid API key."""
        from app.clients.embedding import CohereEmbeddingClient

        client = create_embedding_client(api_key="test_api_key_123")
        assert isinstance(client, CohereEmbeddingClient)


class TestEmbeddingError:
    """Test the embedding error exception."""

    def test_embedding_error_with_message(self):
        """Test embedding error with just a message."""
        error = EmbeddingError("Test error message")
        assert str(error) == "Test error message"
        assert error.cause is None

    def test_embedding_error_with_cause(self):
        """Test embedding error with a cause exception."""
        cause = ValueError("Original error")
        error = EmbeddingError("Wrapped error", cause)

        assert str(error) == "Wrapped error"
        assert error.cause is cause
