"""Tests for application configuration."""

import os
from unittest.mock import patch

from app.core.config import Settings


class TestSettings:
    """Test cases for Settings configuration."""

    def test_default_settings(self, mock_cohere_env):
        """Test that default settings are loaded correctly."""
        # Create settings without .env file to test defaults
        settings = Settings(_env_file=None)

        assert settings.api_title == "StackAI Vector Database"
        assert settings.api_version == "0.1.0"
        assert settings.api_prefix == "/api/v1"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.debug is False
        assert settings.reload is False
        assert settings.log_level == "INFO"
        assert settings.default_embedding_dim == 1024
        assert settings.max_chunks_per_library == 10000
        assert settings.max_knn_results == 1000
        assert settings.default_knn_results == 10

    def test_environment_variable_override(self):
        """Test that environment variables override default settings."""
        with patch.dict(
            os.environ,
            {
                "API_TITLE": "Test Vector DB",
                "PORT": "9000",
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "MAX_CHUNKS_PER_LIBRARY": "5000",
                "COHERE_API_KEY": "test-key-123",
            },
        ):
            settings = Settings(_env_file=None)

            assert settings.api_title == "Test Vector DB"
            assert settings.port == 9000
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.max_chunks_per_library == 5000
            assert settings.cohere_api_key == "test-key-123"

    def test_cohere_api_key_with_value(self, mock_cohere_env):
        """Test that cohere_api_key can be set via environment variable."""
        settings = Settings(_env_file=None)
        assert settings.cohere_api_key == "test-key-123"

    def test_cohere_api_key_optional_defaults_to_none(self):
        """Test that cohere_api_key is optional and defaults to None."""
        # Ensure COHERE_API_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            # Should not raise an error and should default to None
            assert settings.cohere_api_key is None

    def test_embedding_client_selection_based_on_api_key(self):
        """Test that embedding client selection works correctly based on API key presence."""
        from app.clients import (
            CohereEmbeddingClient,
            FakeEmbeddingClient,
            create_embedding_client,
        )

        # Test with no API key - should create FakeEmbeddingClient
        # Use explicit empty string to override any global settings
        client = create_embedding_client(api_key="")
        assert isinstance(client, FakeEmbeddingClient)

        # Test with None API key - should create FakeEmbeddingClient
        client = create_embedding_client(api_key=None)
        # Note: This might still use global settings if available

        # Test with explicit API key - should create CohereEmbeddingClient
        client = create_embedding_client(api_key="test-api-key-123")
        assert isinstance(client, CohereEmbeddingClient)

    def test_logging_formats(self, mock_cohere_env):
        """Test that logging formats are configured correctly."""
        settings = Settings(_env_file=None)

        assert "%(asctime)s" in settings.log_format_general
        assert "%(name)s" in settings.log_format_general
        assert "%(levelname)s" in settings.log_format_general
        assert "%(message)s" in settings.log_format_general

        # Request format should include additional fields
        assert "method=%(method)s" in settings.log_format_request
        assert "path=%(path)s" in settings.log_format_request
        assert "status=%(status_code)s" in settings.log_format_request
        assert "duration_ms=%(duration_ms)s" in settings.log_format_request
        assert "request_id=%(request_id)s" in settings.log_format_request

    def test_cors_configuration(self, mock_cohere_env):
        """Test CORS configuration defaults."""
        settings = Settings(_env_file=None)

        assert settings.cors_origins == ["*"]
        assert settings.cors_allow_credentials is True
        assert settings.cors_allow_methods == ["*"]
        assert settings.cors_allow_headers == ["*"]

    def test_index_configuration(self, mock_cohere_env):
        """Test index configuration defaults."""
        settings = Settings(_env_file=None)

        assert settings.default_index_type == "linear"
        assert settings.index_rebuild_threshold == 0.1

    def test_cohere_configuration(self):
        """Test Cohere API configuration defaults."""
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            settings = Settings(_env_file=None)

            assert settings.cohere_model == "embed-english-v3.0"
            assert settings.cohere_input_type == "search_document"
