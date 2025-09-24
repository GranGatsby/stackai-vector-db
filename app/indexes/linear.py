"""Linear scan vector index implementation.

This module implements a simple linear scan index that serves as a baseline
for comparison with more sophisticated indexing algorithms. It performs
exhaustive search through all vectors.
"""

import logging
from typing import Sequence

import numpy as np

from .base import BaseVectorIndex

logger = logging.getLogger(__name__)


class LinearScanIndex(BaseVectorIndex):
    """Linear scan vector index implementation.

    This is the simplest possible vector index that performs an exhaustive
    linear scan through all vectors to find nearest neighbors. While not
    efficient for large datasets, it serves as a reliable baseline and
    is guaranteed to return exact results.

    Time Complexity:
    - Build: O(N) - Simply stores vectors in memory
    - Query: O(N * D) - Must compute distance to every vector
    - Add: O(1) - Append to vector list
    - Remove: O(1) - Mark as removed

    Space Complexity:
    - O(N * D) - Stores all vectors in memory

    Characteristics:
    - Exact results (no approximation)
    - No preprocessing required
    - Memory efficient (just stores raw vectors)
    - Poor query performance for large datasets
    - Excellent for small datasets or as a reference implementation
    """

    def __init__(
        self, dimension: int = None, distance_metric: str = "euclidean"
    ) -> None:
        """Initialize the linear scan index.

        Args:
            dimension: Expected vector dimension (auto-detected if None)
            distance_metric: Distance metric to use ("euclidean" or "cosine")
        """
        super().__init__(dimension)

        if distance_metric not in ("euclidean", "cosine"):
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")

        self._distance_metric = distance_metric
        self._distance_func = (
            self.euclidean_distance
            if distance_metric == "euclidean"
            else self.cosine_distance
        )

        logger.debug(
            f"Initialized LinearScanIndex with distance_metric={distance_metric}"
        )

    def _build_index(self) -> None:
        """Build the linear scan index.

        For linear scan, building is trivial as we just need to store
        the vectors in memory. No additional data structures are needed.
        """
        # Nothing special to do - vectors are already stored in self._vectors
        # This method exists to satisfy the abstract base class interface
        logger.debug(f"Built LinearScanIndex with {self.size} vectors")

    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Perform linear scan query to find k nearest neighbors.

        Args:
            query_vector: Query vector as numpy array
            k: Number of neighbors to find

        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        if self.size == 0:
            return []

        # Compute distances to all vectors
        distances = []

        for i, vector in enumerate(self._vectors):
            # Skip removed vectors (marked as None)
            if vector is None:
                continue

            distance = self._distance_func(query_vector, vector)
            distances.append((i, distance))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])

        return distances[:k]

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a vector to the linear scan index.

        For linear scan, adding a vector is very efficient as we just
        append it to our vector list. No index rebuilding is required.

        Args:
            vector: Vector to add

        Returns:
            Index position of the added vector
        """
        # Call parent implementation but override the rebuild requirement
        index = super().add_vector(vector)

        # Linear scan doesn't need rebuilding after additions
        self._is_built = True

        return index

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the linear scan index.

        For linear scan, removal is efficient as we just mark the vector
        as removed. No index rebuilding is required.

        Args:
            index: Index position of vector to remove

        Returns:
            True if vector was removed, False if index was invalid
        """
        # Call parent implementation but override the rebuild requirement
        result = super().remove_vector(index)

        if result:
            # Linear scan doesn't need rebuilding after removals
            self._is_built = True

        return result

    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage statistics for the index.

        Returns:
            Dictionary with memory usage information in bytes
        """
        vector_memory = sum(
            vector.nbytes for vector in self._vectors if vector is not None
        )

        return {
            "vectors": vector_memory,
            "total": vector_memory,
            "overhead": 0,  # Linear scan has no additional overhead
        }

    def get_stats(self) -> dict[str, any]:
        """Get comprehensive statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        active_vectors = sum(1 for vector in self._vectors if vector is not None)
        removed_vectors = self.size - active_vectors

        memory_stats = self.get_memory_usage()

        return {
            "algorithm": "LinearScan",
            "dimension": self.dim if self._dimension else None,
            "total_vectors": self.size,
            "active_vectors": active_vectors,
            "removed_vectors": removed_vectors,
            "is_built": self.is_built,
            "distance_metric": self._distance_metric,
            "memory_usage_bytes": memory_stats,
            "complexity": {
                "build_time": "O(N)",
                "query_time": "O(N * D)",
                "space": "O(N * D)",
            },
        }
