"""Search service for performing k-NN queries on vector indexes.

This service orchestrates vector similarity searches across library indexes,
integrating with IndexService for lazy building and providing both text-based
and embedding-based query interfaces.

Key Features:
- Text and embedding-based search interfaces
- Integration with IndexService for lazy index building
- Thread-safe query operations using per-library read locks
- Parameter validation (k, dimensions, library existence)
- Consistent distance metrics across all index types
- Automatic embedding generation for text queries

Architecture:
- Uses IndexService snapshots for concurrent read access
- Triggers lazy index building when needed (dirty or not built)
- Validates all parameters before executing queries
- Maintains metric consistency between search and index algorithms
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from app.clients import EmbeddingClient, EmbeddingError, create_embedding_client
from app.domain import (
    Chunk,
    EmbeddingDimensionMismatchError,
    EmptyLibraryError,
    InvalidSearchParameterError,
    LibraryNotFoundError,
    VectorIndexNotBuiltError,
)
from app.repositories.ports import ChunkRepository, LibraryRepository
from app.services.index_service import IndexAlgo, IndexService, IndexStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Result of a vector similarity search.

    This immutable class represents the results of a k-NN search operation,
    containing the matching chunks ordered by similarity distance.

    Attributes:
        library_id: UUID of the library that was searched
        query_embedding: The embedding vector used for the search
        matches: List of (chunk, distance) tuples ordered by distance (ascending)
        total_results: Number of results returned (len(matches))
        index_size: Total number of vectors in the searched index
        algorithm: Index algorithm used for the search
        embedding_dim: Dimension of the embeddings used
    """

    library_id: UUID
    query_embedding: list[float]
    matches: list[tuple[Chunk, float]]
    total_results: int
    index_size: int
    algorithm: IndexAlgo
    embedding_dim: int

    @property
    def chunks(self) -> list[Chunk]:
        """Get just the chunks from the matches, ordered by relevance."""
        return [chunk for chunk, _ in self.matches]

    @property
    def distances(self) -> list[float]:
        """Get just the distances from the matches."""
        return [distance for _, distance in self.matches]

    @property
    def is_empty(self) -> bool:
        """Check if the search returned no results."""
        return self.total_results == 0


class SearchService:
    """Service for performing k-NN vector similarity searches.

    This service provides high-level search interfaces for both text-based and
    embedding-based queries. It integrates with IndexService to ensure indexes
    are built when needed and handles all parameter validation and error cases.

    Key Responsibilities:
    1. Provide text and embedding-based search interfaces
    2. Integrate with IndexService for lazy index building
    3. Validate all search parameters (k, dimensions, library existence)
    4. Execute thread-safe queries using index snapshots
    5. Generate embeddings for text queries
    6. Maintain consistent distance metrics with index algorithms

    Query Complexity:
    - query_text(): O(E + Q) where E=embedding generation time, Q=query time
    - query_embedding(): O(Q) where Q depends on index algorithm:
      - Linear: O(N*D) - Must check all N vectors of dimension D
      - KDTree: O(log(N)*D) average, O(N*D) worst case - Tree traversal
      - IVF: O(P*M*D + k) - P probes, M avg vectors per cluster, k results

    Distance Metrics:
    - Uses same distance metric as the underlying index algorithm
    - Linear and KDTree: Euclidean distance by default
    - IVF: Euclidean distance for clustering and search
    - All algorithms return results sorted by distance (ascending = most similar first)

    Thread Safety:
    - Uses read locks from IndexService for concurrent query access
    - Operates on immutable index snapshots
    - No shared mutable state between queries
    """

    def __init__(
        self,
        index_service: IndexService,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize the SearchService.

        Args:
            index_service: Service for managing vector indexes
            library_repository: Repository for library operations
            chunk_repository: Repository for chunk operations
            embedding_client: Client for generating embeddings (auto-created if None)
        """
        self._index_service = index_service
        self._library_repo = library_repository
        self._chunk_repo = chunk_repository
        self._embedding_client = embedding_client or create_embedding_client()

        logger.info("Initialized SearchService")

    def query_text(self, library_id: UUID, text: str, k: int = 10) -> SearchResult:
        """Search for similar chunks using text query.

        This method generates an embedding for the input text and then performs
        a k-NN search to find the most similar chunks in the specified library.

        Process:
        1. Validate parameters (library exists, k > 0, text not empty)
        2. Generate embedding for the input text
        3. Delegate to query_embedding() for the actual search

        Args:
            library_id: UUID of the library to search in
            text: Text query to search for
            k: Number of nearest neighbors to return (must be >= 1)

        Returns:
            SearchResult with matching chunks ordered by similarity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
            InvalidSearchParameterError: If k <= 0 or text is empty
            EmptyLibraryError: If the library contains no chunks
            VectorIndexNotBuiltError: If index building fails
            EmbeddingDimensionMismatchError: If embedding dimensions don't match
        """
        # Validate basic parameters
        if not text or not text.strip():
            raise InvalidSearchParameterError("text", text, "cannot be empty")

        if k <= 0:
            raise InvalidSearchParameterError("k", k, "must be greater than 0")

        # Verify library exists
        library = self._library_repo.get_by_id(library_id)
        if library is None:
            raise LibraryNotFoundError(str(library_id))

        logger.debug(f"Generating embedding for text query in library {library_id}")

        try:
            # Generate embedding for the text query
            embedding_result = self._embedding_client.embed_text(text.strip())

            # Delegate to embedding-based search
            return self.query_embedding(
                library_id, embedding_result.single_embedding, k
            )

        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for text query: {e}")
            raise InvalidSearchParameterError(
                "text", text, f"embedding generation failed: {e}"
            ) from e

    def _validate_query_params(
        self, library_id: UUID, embedding: list[float], k: int
    ) -> None:
        """Validate basic query parameters.

        Args:
            library_id: UUID of the library to search in
            embedding: Query embedding vector
            k: Number of nearest neighbors to return

        Raises:
            InvalidSearchParameterError: If parameters are invalid
            LibraryNotFoundError: If the library doesn't exist
        """
        if k <= 0:
            raise InvalidSearchParameterError("k", k, "must be greater than 0")

        if not embedding:
            raise InvalidSearchParameterError("embedding", embedding, "cannot be empty")

        # Verify library exists
        library = self._library_repo.get_by_id(library_id)
        if library is None:
            raise LibraryNotFoundError(str(library_id))

    def _ensure_index_ready(self, library_id: UUID) -> IndexStatus:
        """Ensure index is built and ready for querying.

        Args:
            library_id: UUID of the library

        Returns:
            IndexStatus after ensuring index is ready

        Raises:
            EmptyLibraryError: If the library contains no chunks
        """
        # Get current index status
        index_status = self._index_service.get_status(library_id)

        # Trigger lazy build if needed
        if not index_status.is_built or index_status.is_dirty:
            logger.info(
                f"Triggering lazy index build for library {library_id}: "
                f"is_built={index_status.is_built}, is_dirty={index_status.is_dirty}, "
                f"algorithm={index_status.algorithm.value}, size={index_status.size}"
            )
            index_status = self._index_service.build(library_id)

        # Check if library is empty after potential build
        if index_status.size == 0:
            raise EmptyLibraryError(str(library_id))

        return index_status

    def _validate_embedding_dimension(
        self, embedding: list[float], index_status: IndexStatus
    ) -> None:
        """Validate embedding dimension matches index dimension.

        Args:
            embedding: Query embedding vector
            index_status: Current index status

        Raises:
            EmbeddingDimensionMismatchError: If dimensions don't match
        """
        if (
            index_status.embedding_dim is not None
            and len(embedding) != index_status.embedding_dim
        ):
            raise EmbeddingDimensionMismatchError(
                index_status.embedding_dim, len(embedding)
            )

    def _execute_index_query(
        self, library_id: UUID, embedding: list[float], k: int, index_size: int
    ) -> list[tuple[UUID, float]]:
        """Execute k-NN query on index.

        Args:
            library_id: UUID of the library
            embedding: Query embedding vector
            k: Number of nearest neighbors requested
            index_size: Current index size

        Returns:
            List of (chunk_id, distance) tuples

        Raises:
            VectorIndexNotBuiltError: If index query fails
        """
        # Limit k to actual index size
        effective_k = min(k, index_size)
        if effective_k != k:
            logger.debug(f"Limited k from {k} to {effective_k} based on index size")

        # Execute k-NN query through IndexService
        try:
            query_results = self._index_service.query(
                library_id, embedding, effective_k
            )
            logger.debug(f"Index query returned {len(query_results)} results")
            return query_results
        except VectorIndexNotBuiltError as e:
            # This shouldn't happen since we built the index above, but handle it
            raise VectorIndexNotBuiltError(str(library_id)) from e

    def _process_query_results(
        self, query_results: list[tuple[UUID, float]]
    ) -> list[tuple[Chunk, float]]:
        """Convert index results to chunk entities with similarity filtering.

        Args:
            query_results: List of (chunk_id, distance) tuples from index

        Returns:
            List of (chunk, distance) tuples after filtering
        """
        matches = []
        filtered_count = 0

        for chunk_id, distance in query_results:
            # Retrieve chunk entity
            chunk = self._chunk_repo.get_by_id(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk {chunk_id} not found in repository")
                continue

            # Apply similarity threshold filtering if configured
            if self._should_filter_chunk(chunk, distance):
                logger.debug(
                    f"Chunk {chunk_id} filtered out by similarity_threshold: "
                    f"score={distance:.4f} > threshold={chunk.metadata.similarity_threshold}"
                )
                filtered_count += 1
                continue

            matches.append((chunk, distance))

        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} chunks by similarity_threshold "
                f"(kept {len(matches)} out of {len(query_results)} total results)"
            )

        return matches

    def _should_filter_chunk(self, chunk: Chunk, distance: float) -> bool:
        """Check if chunk should be filtered by similarity threshold.

        Args:
            chunk: Chunk entity
            distance: Similarity distance

        Returns:
            True if chunk should be filtered out
        """
        if not chunk.metadata or chunk.metadata.similarity_threshold is None:
            return False

        return distance > chunk.metadata.similarity_threshold

    def query_embedding(
        self, library_id: UUID, embedding: list[float], k: int = 10
    ) -> SearchResult:
        """Search for similar chunks using embedding vector.

        This method performs a k-NN search using the provided embedding vector
        to find the most similar chunks in the specified library.

        Process:
        1. Validate parameters (library exists, k > 0, embedding not empty)
        2. Ensure index is built (trigger lazy build if needed)
        3. Validate embedding dimension matches index dimension
        4. Execute k-NN query on index snapshot
        5. Retrieve chunk entities and construct result

        Args:
            library_id: UUID of the library to search in
            embedding: Query embedding vector
            k: Number of nearest neighbors to return (must be >= 1)

        Returns:
            SearchResult with matching chunks ordered by similarity

        Raises:
            LibraryNotFoundError: If the library doesn't exist
            InvalidSearchParameterError: If k <= 0 or embedding is invalid
            EmptyLibraryError: If the library contains no chunks
            VectorIndexNotBuiltError: If index building fails
            EmbeddingDimensionMismatchError: If embedding dimensions don't match
        """
        logger.debug(f"Executing embedding query in library {library_id} with k={k}")

        # Step 1: Validate parameters
        self._validate_query_params(library_id, embedding, k)

        # Step 2: Ensure index is ready
        index_status = self._ensure_index_ready(library_id)

        # Step 3: Validate embedding dimension
        self._validate_embedding_dimension(embedding, index_status)

        # Step 4: Execute query
        query_results = self._execute_index_query(
            library_id, embedding, k, index_status.size
        )

        # Step 5: Process results
        matches = self._process_query_results(query_results)

        # Create search result
        result = SearchResult(
            library_id=library_id,
            query_embedding=embedding,
            matches=matches,
            total_results=len(matches),
            index_size=index_status.size,
            algorithm=index_status.algorithm,
            embedding_dim=index_status.embedding_dim or len(embedding),
        )

        logger.info(
            f"Search completed for library {library_id}: "
            f"{result.total_results} results from {result.index_size} vectors "
            f"using {result.algorithm.value} algorithm"
        )

        return result
