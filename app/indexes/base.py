"""Base protocol and interfaces for vector indexing algorithms.

This module defines the common interface that all vector index implementations
must follow, along with base classes and utilities for index management.
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector index implementations.
    
    This protocol defines the interface that all vector index algorithms must implement.
    It allows for easy substitution of different indexing strategies.
    """

    @property
    def dim(self) -> int:
        """Get the dimension of vectors in this index.
        
        Returns:
            The vector dimension
        """
        ...

    @property
    def size(self) -> int:
        """Get the number of vectors currently in the index.
        
        Returns:
            Number of indexed vectors
        """
        ...

    @property
    def is_built(self) -> bool:
        """Check if the index has been built and is ready for queries.
        
        Returns:
            True if index is built and ready for queries
        """
        ...

    def build(self, vectors: Sequence[Sequence[float]]) -> None:
        """Build the index from a collection of vectors.
        
        Args:
            vectors: Collection of vectors to index
            
        Raises:
            IndexError: If vectors have inconsistent dimensions
            ValueError: If vectors collection is empty
        """
        ...

    def query(self, query_vector: Sequence[float], k: int = 10) -> list[tuple[int, float]]:
        """Find the k nearest neighbors to the query vector.
        
        Args:
            query_vector: The vector to search for
            k: Number of nearest neighbors to return
            
        Returns:
            List of (index, distance) tuples, sorted by distance (ascending)
            
        Raises:
            IndexNotBuiltError: If index hasn't been built yet
            ValueError: If query vector dimension doesn't match index dimension
        """
        ...

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a single vector to the index (incremental update).
        
        Args:
            vector: Vector to add
            
        Returns:
            Index position of the added vector
            
        Raises:
            ValueError: If vector dimension doesn't match index dimension
            
        Note:
            Some index types may require rebuilding after additions.
        """
        ...

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the index.
        
        Args:
            index: Index position of vector to remove
            
        Returns:
            True if vector was removed, False if index was invalid
            
        Note:
            Some index types may require rebuilding after removals.
        """
        ...


class IndexError(Exception):
    """Base exception for index-related errors."""
    pass


class IndexNotBuiltError(IndexError):
    """Exception raised when trying to query an index that hasn't been built."""
    pass


class DimensionMismatchError(IndexError):
    """Exception raised when vector dimensions don't match index expectations."""
    pass


class BaseVectorIndex(ABC):
    """Abstract base class for vector index implementations.
    
    This class provides common functionality and enforces the interface
    for all concrete index implementations.
    """

    def __init__(self, dimension: int = None) -> None:
        """Initialize the base index.
        
        Args:
            dimension: Expected vector dimension (determined automatically if None)
        """
        self._dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._is_built = False
        self._size = 0
        
        logger.debug(f"Initialized {self.__class__.__name__} with dim={dimension}")

    @property
    def dim(self) -> int:
        """Get the dimension of vectors in this index."""
        if self._dimension is None:
            raise IndexNotBuiltError("Index dimension not determined yet")
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
                    raise DimensionMismatchError(
                        f"Expected dimension {self._dimension}, got {first_dim}"
                    )
            elif len(np_vector) != first_dim:
                raise DimensionMismatchError(
                    f"Inconsistent vector dimensions: {first_dim} vs {len(np_vector)} at index {i}"
                )
            
            np_vectors.append(np_vector)
        
        self._vectors = np_vectors
        self._size = len(np_vectors)
        
        logger.info(f"Building {self.__class__.__name__} with {self._size} vectors of dimension {self._dimension}")
        
        # Call the concrete implementation
        self._build_index()
        self._is_built = True
        
        logger.info(f"Successfully built {self.__class__.__name__}")

    @abstractmethod
    def _build_index(self) -> None:
        """Build the concrete index structure.
        
        This method is called by build() after vector validation and should
        implement the specific indexing algorithm.
        """
        pass

    def query(self, query_vector: Sequence[float], k: int = 10) -> list[tuple[int, float]]:
        """Find the k nearest neighbors to the query vector."""
        if not self._is_built:
            raise IndexNotBuiltError("Index must be built before querying")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Convert query vector to numpy array and validate dimension
        query_np = np.array(query_vector, dtype=np.float32)
        if len(query_np) != self._dimension:
            raise DimensionMismatchError(
                f"Query vector dimension {len(query_np)} doesn't match index dimension {self._dimension}"
            )
        
        # Limit k to available vectors
        k = min(k, self._size)
        
        # Call the concrete implementation
        results = self._query_index(query_np, k)
        
        # Validate and sort results
        if len(results) > k:
            results = results[:k]
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x[1])
        
        return results

    @abstractmethod
    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Perform the concrete query implementation.
        
        Args:
            query_vector: Normalized query vector as numpy array
            k: Number of neighbors to find
            
        Returns:
            List of (index, distance) tuples
        """
        pass

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a single vector to the index (incremental update).
        
        Default implementation appends to vector list and marks index as needing rebuild.
        Concrete implementations may override for more efficient incremental updates.
        """
        np_vector = np.array(vector, dtype=np.float32)
        
        if self._dimension is not None and len(np_vector) != self._dimension:
            raise DimensionMismatchError(
                f"Vector dimension {len(np_vector)} doesn't match index dimension {self._dimension}"
            )
        
        if self._dimension is None:
            self._dimension = len(np_vector)
        
        self._vectors.append(np_vector)
        index = self._size
        self._size += 1
        
        # Mark as needing rebuild (concrete implementations may override this behavior)
        self._is_built = False
        
        logger.debug(f"Added vector at index {index}, index needs rebuild")
        return index

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the index.
        
        Default implementation marks vector as removed and requires rebuild.
        """
        if index < 0 or index >= self._size:
            return False
        
        # Mark vector as None (tombstone) rather than removing to preserve indices
        self._vectors[index] = None
        self._is_built = False
        
        logger.debug(f"Removed vector at index {index}, index needs rebuild")
        return True

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(v1 - v2))

    @staticmethod
    def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine distance between two vectors.
        
        Args:
            v1: First vector  
            v2: Second vector
            
        Returns:
            Cosine distance (1 - cosine similarity)
        """
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            return 1.0  # Maximum distance for zero vectors
        cosine_similarity = dot_product / norms
        return 1.0 - cosine_similarity
