"""Vector indexing algorithms for efficient nearest neighbor search.

This package provides multiple indexing algorithms for vector similarity search,
each optimized for different use cases and dataset characteristics.

Available Algorithms:
- LinearScanIndex: Simple baseline with exact results
- KDTreeIndex: Efficient for low-dimensional data
- IVFIndex: Scalable approximate search for large datasets
"""

from .base import (
    BaseVectorIndex,
    DimensionMismatchError,
    IndexError,
    IndexNotBuiltError,
    VectorIndex,
)
from .ivf import IVFIndex
from .kdtree import KDTreeIndex
from .linear import LinearScanIndex
from .manager import IndexManager, create_index

__all__ = [
    # Base classes and protocols
    "VectorIndex",
    "BaseVectorIndex",
    # Exceptions
    "IndexError",
    "IndexNotBuiltError", 
    "DimensionMismatchError",
    # Concrete implementations
    "LinearScanIndex",
    "KDTreeIndex",
    "IVFIndex",
    # Factory and management
    "create_index",
    "IndexManager",
]
