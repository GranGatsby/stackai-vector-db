"""Embedding clients for computing text embeddings.

This module defines the embedding client protocol and concrete implementations
for different embedding providers (Cohere API, fake client for testing).
It follows the strategy pattern to allow switching between different embedding
providers based on configuration.
"""

import logging
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients.

    This protocol defines the interface that all embedding clients must implement.
    It allows for easy substitution of different embedding providers.
    """

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this client.

        Returns:
            The embedding vector dimension
        """
        ...

    def embed_text(self, text: str) -> list[float]:
        """Compute embedding for the given text.

        Args:
            text: The text to embed

        Returns:
            The embedding vector as a list of floats

        Raises:
            EmbeddingError: If embedding computation fails
        """
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts (batch operation).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding computation fails
        """
        ...


class EmbeddingError(Exception):
    """Exception raised when embedding computation fails."""

    def __init__(self, message: str, cause: Exception = None) -> None:
        super().__init__(message)
        self.cause = cause


class FakeEmbeddingClient:
    """Fake embedding client for testing and development.

    This client generates deterministic fake embeddings based on text content.
    Useful when no API key is available or for testing purposes.
    """

    def __init__(self, embedding_dim: int = None) -> None:
        """Initialize the fake embedding client.

        Args:
            embedding_dim: Dimension of embeddings to generate
        """
        self._embedding_dim = embedding_dim or settings.default_embedding_dim
        logger.info(f"Initialized FakeEmbeddingClient with dim={self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this client."""
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic fake embedding for the given text.

        The embedding is generated based on text characteristics to ensure
        similar texts get similar embeddings for testing purposes.

        Args:
            text: The text to embed

        Returns:
            A deterministic fake embedding vector
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        # Generate deterministic embedding based on text characteristics
        text_clean = text.strip().lower()

        # Use text properties to generate embedding components
        embedding = []
        for i in range(self._embedding_dim):
            # Mix different text properties for pseudo-randomness
            char_sum = sum(ord(c) for c in text_clean)
            length_factor = len(text_clean)
            word_count = len(text_clean.split())

            # Create a deterministic but varied component
            component = (
                (char_sum * (i + 1) + length_factor * (i + 7) + word_count * (i + 13))
                % 1000
            ) / 1000.0 - 0.5

            embedding.append(component)

        logger.debug(f"Generated fake embedding for text length {len(text)}")
        return embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings for multiple texts."""
        if not texts:
            return []

        return [self.embed_text(text) for text in texts]


class CohereEmbeddingClient:
    """Cohere API embedding client.

    This client uses the Cohere API to generate real embeddings.
    It handles API authentication, rate limiting, and error handling.
    """

    def __init__(
        self,
        api_key: str,
        model: str = None,
        input_type: str = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Cohere embedding client.

        Args:
            api_key: Cohere API key
            model: Cohere model name to use
            input_type: Input type for embeddings
            timeout: Request timeout in seconds
        """
        if not api_key or not api_key.strip():
            raise ValueError("Cohere API key is required")

        self._api_key = api_key.strip()
        self._model = model or settings.cohere_model
        self._input_type = input_type or settings.cohere_input_type
        self._timeout = timeout
        self._base_url = "https://api.cohere.ai/v1"

        # We'll determine embedding dimension on first call
        self._embedding_dim = None

        logger.info(
            f"Initialized CohereEmbeddingClient with model={self._model}, "
            f"input_type={self._input_type}"
        )

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this client.

        Returns:
            The embedding dimension (determined after first API call)
        """
        if self._embedding_dim is None:
            # Use configured default until we make first API call
            return settings.default_embedding_dim
        return self._embedding_dim

    def embed_text(self, text: str) -> list[float]:
        """Compute embedding using Cohere API.

        Args:
            text: The text to embed

        Returns:
            The embedding vector from Cohere API

        Raises:
            EmbeddingError: If API call fails or returns invalid data
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            embeddings = self.embed_texts([text])
            return embeddings[0]
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed single text: {e}", e)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for multiple texts using Cohere API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors from Cohere API

        Raises:
            EmbeddingError: If API call fails or returns invalid data
        """
        if not texts:
            return []

        # Validate inputs
        clean_texts = []
        for text in texts:
            if not text or not text.strip():
                raise EmbeddingError("Cannot embed empty text")
            clean_texts.append(text.strip())

        try:
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "texts": clean_texts,
                "model": self._model,
                "input_type": self._input_type,
            }

            logger.debug(f"Making Cohere API request for {len(clean_texts)} texts")

            # Make API request
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/embed",
                    headers=headers,
                    json=payload,
                )

            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Cohere API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg)

            # Parse response
            data = response.json()

            if "embeddings" not in data:
                raise EmbeddingError("Invalid response format: missing 'embeddings'")

            embeddings = data["embeddings"]

            if len(embeddings) != len(clean_texts):
                raise EmbeddingError(
                    f"Expected {len(clean_texts)} embeddings, got {len(embeddings)}"
                )

            # Set embedding dimension from first response
            if embeddings and self._embedding_dim is None:
                self._embedding_dim = len(embeddings[0])
                logger.info(
                    f"Detected Cohere embedding dimension: {self._embedding_dim}"
                )

            logger.debug(f"Successfully embedded {len(clean_texts)} texts")
            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            error_msg = f"Failed to call Cohere API: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)


def create_embedding_client(api_key: str = None) -> EmbeddingClient:
    """Factory function to create the appropriate embedding client.

    This function creates either a real Cohere client (if API key is provided)
    or a fake client (for testing/development without API key).

    Args:
        api_key: Optional Cohere API key. If None, uses settings.

    Returns:
        An embedding client instance
    """
    # Determine API key to use
    # If api_key is explicitly provided (even if empty), use that
    # Otherwise, fall back to settings
    if api_key is not None:
        final_api_key = api_key
    else:
        final_api_key = getattr(settings, "cohere_api_key", None)

    # Create appropriate client based on API key availability
    if final_api_key and final_api_key.strip():
        logger.info("Creating CohereEmbeddingClient")
        return CohereEmbeddingClient(final_api_key)
    else:
        logger.info("No API key provided, creating FakeEmbeddingClient")
        return FakeEmbeddingClient()
