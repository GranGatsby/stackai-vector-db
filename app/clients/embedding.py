"""Embedding clients for computing text embeddings.

This module defines the embedding client protocol and concrete implementations
for different embedding providers (Cohere API, fake client for testing).
It follows the strategy pattern to allow switching between different embedding
providers based on configuration.
"""

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import settings
from app.utils import validate_non_empty_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding computation result with metadata.

    Attributes:
        embeddings: List of embedding vectors
        model_name: Name/identifier of the embedding model used
        embedding_dim: Dimension of each embedding vector
    """

    embeddings: list[list[float]]
    model_name: str
    embedding_dim: int

    @property
    def single_embedding(self) -> list[float]:
        """Get single embedding for single-text operations.

        Raises:
            ValueError: If result contains != 1 embedding
        """
        if len(self.embeddings) != 1:
            raise ValueError(f"Expected 1 embedding, got {len(self.embeddings)}")
        return self.embeddings[0]


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients with pluggable providers."""

    @property
    def embedding_dim(self) -> int:
        """Get embedding vector dimension."""
        ...

    def embed_text(self, text: str) -> EmbeddingResult:
        """Compute embedding for single text.

        Raises:
            EmbeddingError: If embedding computation fails
        """
        ...

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Compute embeddings for multiple texts.

        Raises:
            EmbeddingError: If embedding computation fails
        """
        ...


class EmbeddingError(Exception):
    """Exception raised when embedding computation fails."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class FakeEmbeddingClient:
    """Deterministic fake embedding client for testing."""

    def __init__(self, embedding_dim: int | None = None) -> None:
        """Initialize fake embedding client."""
        self._embedding_dim = embedding_dim or settings.default_embedding_dim
        logger.info(f"Initialized FakeEmbeddingClient with dim={self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Get embedding vector dimension."""
        return self._embedding_dim

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate deterministic fake embedding based on text characteristics."""
        try:
            text_clean = validate_non_empty_text(text, "Cannot embed empty text")
        except ValueError as e:
            raise EmbeddingError(str(e)) from e

        # Generate deterministic embedding based on text characteristics
        text_clean = text_clean.lower()

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

        return EmbeddingResult(
            embeddings=[embedding],
            model_name="fake-embedding-model",
            embedding_dim=self._embedding_dim,
        )

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Generate fake embeddings for multiple texts."""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name="fake-embedding-model",
                embedding_dim=self._embedding_dim,
            )

        embeddings = []
        for text in texts:
            # Reuse single embedding logic but extract just the embedding vector
            single_result = self.embed_text(text)
            embeddings.append(single_result.single_embedding)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name="fake-embedding-model",
            embedding_dim=self._embedding_dim,
        )


class CohereEmbeddingClient:
    """Cohere API embedding client with authentication and error handling."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        input_type: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Cohere embedding client."""
        if not api_key or not api_key.strip():
            raise ValueError("Cohere API key is required")

        self._api_key = api_key.strip()
        self._model = model or settings.cohere_model
        self._input_type = input_type or settings.cohere_input_type
        self._timeout = timeout
        self._base_url = settings.cohere_base_url

        # We'll determine embedding dimension on first call
        self._embedding_dim = None

        logger.info(
            f"Initialized CohereEmbeddingClient with model={self._model}, "
            f"input_type={self._input_type}"
        )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (determined after first API call)."""
        if self._embedding_dim is None:
            return settings.default_embedding_dim
        return self._embedding_dim

    def embed_text(self, text: str) -> EmbeddingResult:
        """Compute single text embedding using Cohere API.

        Raises:
            EmbeddingError: If API call fails or returns invalid data
        """
        try:
            validate_non_empty_text(text, "Cannot embed empty text")
        except ValueError as e:
            raise EmbeddingError(str(e)) from e

        try:
            return self.embed_texts([text])
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to embed single text: {e}", e) from e

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Compute batch embeddings using Cohere API.

        Raises:
            EmbeddingError: If API call fails or returns invalid data
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self._model,
                embedding_dim=self.embedding_dim,
            )

        # Validate inputs
        clean_texts = []
        for text in texts:
            try:
                clean_texts.append(
                    validate_non_empty_text(text, "Cannot embed empty text")
                )
            except ValueError as e:
                raise EmbeddingError(str(e)) from e

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
            if response.status_code != HTTPStatus.OK:
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

            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self._model,
                embedding_dim=(
                    self._embedding_dim or len(embeddings[0]) if embeddings else 0
                ),
            )

        except EmbeddingError:
            raise
        except Exception as e:
            error_msg = f"Failed to call Cohere API: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e) from e


def create_embedding_client(api_key: str | None = None) -> EmbeddingClient:
    """Create Cohere client (if API key available) or fake client."""
    # Determine API key: explicit param takes precedence over settings
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
