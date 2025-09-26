"""KD-Tree vector index implementation."""

import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .base import BaseVectorIndex

logger = logging.getLogger(__name__)


class KDNode:
    """KD-Tree node representing a vector space partition."""

    def __init__(
        self,
        vector: np.ndarray,
        index: int,
        split_dim: int,
        left: Optional["KDNode"] = None,
        right: Optional["KDNode"] = None,
    ) -> None:
        """Initialize KD-Tree node."""
        self.vector = vector
        self.index = index
        self.split_dim = split_dim
        self.left = left
        self.right = right


class KDTreeIndex(BaseVectorIndex):
    """KD-Tree index for low-medium dimensional data.

    Time: Build O(N log N), Query O(log N) avg/O(N) worst, Space O(N)
    Best for: Low dimensions (D <= 20), exact results required
    """

    def __init__(self, dimension: int | None = None, leaf_size: int = 10) -> None:
        """Initialize KD-Tree index."""
        super().__init__(dimension)
        self._leaf_size = max(1, leaf_size)
        self._root: KDNode | None = None

        logger.debug(f"Initialized KDTreeIndex with leaf_size={self._leaf_size}")

    def _build_index(self) -> None:
        """Build KD-Tree from stored vectors."""
        if not self._vectors:
            self._root = None
            return

        # Create (vector, index) pairs, filter removed
        points = [
            (vector, i) for i, vector in enumerate(self._vectors) if vector is not None
        ]

        if not points:
            self._root = None
            return

        self._root = self._build_tree(points, depth=0)

        logger.debug(f"Built KDTree with {len(points)} active vectors")

    def _build_tree(
        self, points: list[tuple[np.ndarray, int]], depth: int
    ) -> KDNode | None:
        """Recursively build KD-Tree."""
        if not points:
            return None

        if len(points) <= self._leaf_size:
            # Leaf node - pick first as representative
            vector, index = points[0]
            split_dim = depth % self.dim
            node = KDNode(vector, index, split_dim)

            # Store remaining points as children if there are any
            if len(points) > 1:
                remaining = points[1:]
                node.left = self._build_tree(remaining, depth + 1)

            return node

        # Choose splitting dimension (cycle through dimensions)
        split_dim = depth % self.dim

        # Sort points by the splitting dimension
        points.sort(key=lambda x: x[0][split_dim])

        # Find median
        median_idx = len(points) // 2
        median_vector, median_index = points[median_idx]

        # Create node and recursively build subtrees
        node = KDNode(median_vector, median_index, split_dim)
        node.left = self._build_tree(points[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1 :], depth + 1)

        return node

    def _query_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """KD-Tree query for k nearest neighbors."""
        if self._root is None:
            return []

        # Bounded priority queue for k best candidates
        best_candidates: list[tuple[float, int]] = []

        def add_candidate(distance: float, index: int) -> None:
            """Add candidate to k-best list."""
            best_candidates.append((distance, index))
            best_candidates.sort()
            if len(best_candidates) > k:
                best_candidates.pop()

        def search_tree(node: KDNode | None) -> None:
            """Recursively search KD-Tree."""
            if node is None:
                return

            # Compute distance to current node
            distance = self.euclidean_distance(query_vector, node.vector)
            add_candidate(distance, node.index)

            # Determine which side to search first
            split_val = query_vector[node.split_dim]
            node_val = node.vector[node.split_dim]

            if split_val < node_val:
                # Query point is on the left side
                first_child = node.left
                second_child = node.right
            else:
                # Query point is on the right side
                first_child = node.right
                second_child = node.left

            # Search the side that contains the query point
            search_tree(first_child)

            # Search other side if needed
            should_search_other_side = (
                len(best_candidates) < k
                or abs(split_val - node_val) < best_candidates[-1][0]
            )

            if should_search_other_side:
                search_tree(second_child)

        search_tree(self._root)

        return [(index, distance) for distance, index in best_candidates]

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add vector to KD-Tree (requires rebuild for optimal performance)."""
        return super().add_vector(vector)

    def remove_vector(self, index: int) -> bool:
        """Remove vector from KD-Tree (requires rebuild for optimal performance)."""
        return super().remove_vector(index)

    def get_tree_depth(self) -> int:
        """Maximum depth of KD-Tree."""

        def get_depth(node: KDNode | None) -> int:
            if node is None:
                return 0
            left_depth = get_depth(node.left)
            right_depth = get_depth(node.right)
            return 1 + max(left_depth, right_depth)

        return get_depth(self._root)

    def get_memory_usage(self) -> dict[str, int]:
        """Memory usage statistics in bytes."""
        vector_memory = sum(
            vector.nbytes for vector in self._vectors if vector is not None
        )

        # Tree structure overhead estimate
        node_count = sum(1 for vector in self._vectors if vector is not None)
        _BYTES_PER_NODE = 64  # Rough estimate
        tree_overhead = node_count * _BYTES_PER_NODE

        return {
            "vectors": vector_memory,
            "tree_structure": tree_overhead,
            "total": vector_memory + tree_overhead,
        }

    def get_stats(self) -> dict[str, any]:
        """Comprehensive KD-Tree statistics."""
        active_vectors = sum(1 for vector in self._vectors if vector is not None)
        removed_vectors = self.size - active_vectors

        memory_stats = self.get_memory_usage()
        tree_depth = self.get_tree_depth() if self.is_built else 0

        return {
            "algorithm": "KDTree",
            "dimension": self.dim if self._dimension else None,
            "total_vectors": self.size,
            "active_vectors": active_vectors,
            "removed_vectors": removed_vectors,
            "is_built": self.is_built,
            "leaf_size": self._leaf_size,
            "tree_depth": tree_depth,
            "memory_usage_bytes": memory_stats,
            "complexity": {
                "build_time": "O(N log N)",
                "query_time": "O(log N) average, O(N) worst case",
                "space": "O(N)",
            },
            "characteristics": [
                "Excellent for low dimensions (D <= 20)",
                "Performance degrades in high dimensions",
                "Exact results",
                "Supports incremental updates (may become unbalanced)",
            ],
        }
