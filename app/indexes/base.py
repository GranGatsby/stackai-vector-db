"""Base protocol and interfaces for vector indexing algorithms."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector index implementations."""

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        ...

    @property
    def is_built(self) -> bool:
        """True if index is built and ready for queries."""
        ...

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build the index from vectors."""
        ...

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Find k nearest neighbors, returns (index, distance) tuples sorted by distance."""
        ...

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector, returns index position."""
        ...

    def remove_vector(self, index: int) -> bool:
        """Remove vector, returns True if successful."""
        ...


class VectorIndexError(Exception):
    """Base exception for vector index-related errors."""

    pass


class VectorIndexNotBuiltError(VectorIndexError):
    """Exception raised when trying to query an index that hasn't been built."""

    pass


class VectorIndexDimensionMismatchError(VectorIndexError):
    """Exception raised when vector dimensions don't match index expectations."""

    pass


class BaseVectorIndex(ABC):
    """Base class with common functionality for vector index implementations."""

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize base index."""
        self._dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._is_built = False
        self._size = 0

        logger.debug(f"Initialized {self.__class__.__name__} with dim={dimension}")

    @property
    def dim(self) -> int:
        """Get the dimension of vectors in this index."""
        if self._dimension is None:
            raise VectorIndexNotBuiltError("Index dimension not determined yet")
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of vectors currently in the index."""
        return self._size

    @property
    def is_built(self) -> bool:
        """Check if the index has been built and is ready for queries."""
        return self._is_built

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build the index from a collection of vectors."""
        if not vectors:
            raise ValueError("Cannot build index from empty vector collection")

        # Convert to numpy arrays and validate dimensions
        np_vectors = []
        first_dim = None

        for i, vector in enumerate(vectors):
            np_vector = np.array(vector, dtype=np.float32)

            if first_dim is None:
                first_dim = len(np_vector)
                if self._dimension is None:
                    self._dimension = first_dim
                elif self._dimension != first_dim:
                    raise VectorIndexDimensionMismatchError(
                        f"Expected dimension {self._dimension}, got {first_dim}"
                    )
            elif len(np_vector) != first_dim:
                raise VectorIndexDimensionMismatchError(
                    f"Inconsistent vector dimensions: {first_dim} vs {len(np_vector)} at index {i}"
                )

            np_vectors.append(np_vector)

        self._vectors = np_vectors
        self._size = len(np_vectors)

        logger.info(
            f"Building {self.__class__.__name__} with {self._size} vectors of dimension {self._dimension}"
        )

        # Call the concrete implementation
        self._build_index()
        self._is_built = True

        logger.info(f"Successfully built {self.__class__.__name__}")

    @abstractmethod
    def _build_index(self) -> None:
        """Build concrete index structure."""
        pass

    def query(
        self, query_vector: Sequence[float], k: int = 10
    ) -> list[tuple[int, float]]:
        """Find the k nearest neighbors to the query vector."""
        if not self._is_built:
            raise VectorIndexNotBuiltError("Index must be built before querying")

        if k <= 0:
            raise ValueError("k must be positive")

        # Convert and validate query vector
        query_np = np.array(query_vector, dtype=np.float32)
        if len(query_np) != self._dimension:
            raise VectorIndexDimensionMismatchError(
                f"Query vector dimension {len(query_np)} doesn't match index dimension {self._dimension}"
            )

        # Limit k to available vectors
        k = min(k, self._size)
        results = self._query_index(query_np, k)

        # Validate and sort results
        if len(results) > k:
            results = results[:k]

        results.sort(key=lambda x: x[1])
        return results

    @abstractmethod
    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Concrete query implementation."""
        pass

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector to index."""
        np_vector = np.array(vector, dtype=np.float32)

        if self._dimension is not None and len(np_vector) != self._dimension:
            raise VectorIndexDimensionMismatchError(
                f"Vector dimension {len(np_vector)} doesn't match index dimension {self._dimension}"
            )

        if self._dimension is None:
            self._dimension = len(np_vector)

        self._vectors.append(np_vector)
        index = self._size
        self._size += 1
        self._is_built = False

        logger.debug(f"Added vector at index {index}, index needs rebuild")
        return index

    def remove_vector(self, index: int) -> bool:
        """Remove vector from index."""
        if index < 0 or index >= self._size:
            return False

        # Mark vector as None (tombstone) rather than removing to preserve indices
        self._vectors[index] = None
        self._is_built = False

        logger.debug(f"Removed vector at index {index}, index needs rebuild")
        return True

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Euclidean distance between vectors."""
        return float(np.linalg.norm(v1 - v2))

    @staticmethod
    def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine distance between vectors."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            return 1.0  # Maximum distance for zero vectors
        cosine_similarity = dot_product / norms
        return 1.0 - cosine_similarity
