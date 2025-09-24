"""KD-Tree vector index implementation.

This module implements a KD-Tree (k-dimensional tree) index for efficient
nearest neighbor search in low to medium dimensional spaces. KD-Trees work
well for dimensions up to about 10-20, after which they suffer from the
curse of dimensionality.
"""

import logging
from typing import Optional, Sequence

import numpy as np

from .base import BaseVectorIndex

logger = logging.getLogger(__name__)


class KDNode:
    """Node in a KD-Tree structure.

    Each node represents a partition of the vector space along one dimension.
    """

    def __init__(
        self,
        vector: np.ndarray,
        index: int,
        split_dim: int,
        left: Optional["KDNode"] = None,
        right: Optional["KDNode"] = None,
    ) -> None:
        """Initialize a KD-Tree node.

        Args:
            vector: The vector stored at this node
            index: Original index of the vector in the dataset
            split_dim: Dimension used to split at this node
            left: Left child node (vectors with smaller values in split_dim)
            right: Right child node (vectors with larger values in split_dim)
        """
        self.vector = vector
        self.index = index
        self.split_dim = split_dim
        self.left = left
        self.right = right


class KDTreeIndex(BaseVectorIndex):
    """KD-Tree vector index implementation.

    KD-Trees are binary trees that partition the vector space by recursively
    splitting along different dimensions. They provide efficient nearest neighbor
    search for low to medium dimensional data.

    Time Complexity:
    - Build: O(N log N) - Recursive partitioning with median finding
    - Query: O(log N) average, O(N) worst case - Tree traversal with backtracking
    - Add: O(log N) - Insert into existing tree (may cause imbalance)
    - Remove: O(log N) - Mark as removed, may require rebalancing

    Space Complexity:
    - O(N) - One node per vector plus tree structure overhead

    Characteristics:
    - Excellent for low dimensions (D <= 20)
    - Performance degrades in high dimensions due to curse of dimensionality
    - Exact results (no approximation)
    - Memory efficient tree structure
    - Supports incremental updates (though may become unbalanced)

    Best Use Cases:
    - Low to medium dimensional embeddings
    - When exact results are required
    - Datasets with frequent queries and infrequent updates
    """

    def __init__(self, dimension: int = None, leaf_size: int = 10) -> None:
        """Initialize the KD-Tree index.

        Args:
            dimension: Expected vector dimension (auto-detected if None)
            leaf_size: Minimum number of points in a leaf node before splitting
        """
        super().__init__(dimension)
        self._leaf_size = max(1, leaf_size)
        self._root: Optional[KDNode] = None

        logger.debug(f"Initialized KDTreeIndex with leaf_size={self._leaf_size}")

    def _build_index(self) -> None:
        """Build the KD-Tree from the stored vectors."""
        if not self._vectors:
            self._root = None
            return

        # Create list of (vector, original_index) pairs, filtering out removed vectors
        points = [
            (vector, i) for i, vector in enumerate(self._vectors) if vector is not None
        ]

        if not points:
            self._root = None
            return

        # Build the tree recursively
        self._root = self._build_tree(points, depth=0)

        logger.debug(f"Built KDTree with {len(points)} active vectors")

    def _build_tree(
        self, points: list[tuple[np.ndarray, int]], depth: int
    ) -> Optional[KDNode]:
        """Recursively build the KD-Tree.

        Args:
            points: List of (vector, index) tuples to build tree from
            depth: Current depth in the tree (determines split dimension)

        Returns:
            Root node of the subtree, or None if no points
        """
        if not points:
            return None

        # Stop recursion if we have few enough points
        if len(points) <= self._leaf_size:
            # For leaf nodes, we still create a tree structure but don't split further
            # Just pick the first point as the representative
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
        """Perform KD-Tree query to find k nearest neighbors.

        Args:
            query_vector: Query vector as numpy array
            k: Number of neighbors to find

        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        if self._root is None:
            return []

        # Use a max-heap to keep track of k best candidates
        # We'll use a list and manually maintain it as a bounded priority queue
        best_candidates: list[tuple[float, int]] = []  # (distance, index)

        def add_candidate(distance: float, index: int) -> None:
            """Add a candidate to our k-best list."""
            best_candidates.append((distance, index))
            best_candidates.sort()  # Keep sorted by distance
            if len(best_candidates) > k:
                best_candidates.pop()  # Remove worst candidate

        def search_tree(node: Optional[KDNode]) -> None:
            """Recursively search the KD-Tree."""
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

            # Check if we need to search the other side
            # We need to search if either:
            # 1. We don't have k candidates yet, or
            # 2. The distance to the splitting hyperplane is less than our worst candidate
            should_search_other_side = (
                len(best_candidates) < k
                or abs(split_val - node_val) < best_candidates[-1][0]
            )

            if should_search_other_side:
                search_tree(second_child)

        # Start the search from the root
        search_tree(self._root)

        # Convert to the expected format
        return [(index, distance) for distance, index in best_candidates]

    def add_vector(self, vector: Sequence[float]) -> int:
        """Add a vector to the KD-Tree index.

        Note: Adding vectors to a KD-Tree requires rebuilding for optimal performance.
        This implementation marks the index as needing rebuild.

        Args:
            vector: Vector to add

        Returns:
            Index position of the added vector
        """
        # KD-Trees need rebuilding after additions for optimal performance
        index = super().add_vector(vector)
        # The parent class already marks _is_built as False
        return index

    def remove_vector(self, index: int) -> bool:
        """Remove a vector from the KD-Tree index.

        Note: Removing vectors from a KD-Tree requires rebuilding for optimal performance.
        This implementation marks the vector as removed and the index as needing rebuild.

        Args:
            index: Index position of vector to remove

        Returns:
            True if vector was removed, False if index was invalid
        """
        # KD-Trees need rebuilding after removals for optimal performance
        result = super().remove_vector(index)
        # The parent class already marks _is_built as False
        return result

    def get_tree_depth(self) -> int:
        """Get the maximum depth of the KD-Tree.

        Returns:
            Maximum depth of the tree, or 0 if tree is empty
        """

        def get_depth(node: Optional[KDNode]) -> int:
            if node is None:
                return 0
            left_depth = get_depth(node.left)
            right_depth = get_depth(node.right)
            return 1 + max(left_depth, right_depth)

        return get_depth(self._root)

    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage statistics for the KD-Tree index.

        Returns:
            Dictionary with memory usage information in bytes
        """
        vector_memory = sum(
            vector.nbytes for vector in self._vectors if vector is not None
        )

        # Estimate tree structure overhead
        # Each node has: vector reference, index, split_dim, left/right pointers
        node_count = sum(1 for vector in self._vectors if vector is not None)
        # Rough estimate: 64 bytes per node for overhead
        tree_overhead = node_count * 64

        return {
            "vectors": vector_memory,
            "tree_structure": tree_overhead,
            "total": vector_memory + tree_overhead,
        }

    def get_stats(self) -> dict[str, any]:
        """Get comprehensive statistics about the KD-Tree index.

        Returns:
            Dictionary with index statistics
        """
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
