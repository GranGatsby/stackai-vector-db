"""Search service for k-NN vector similarity queries."""

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
    """Result of a vector similarity search."""

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
    """Service for k-NN vector similarity searches."""

    def __init__(
        self,
        index_service: IndexService,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._index_service = index_service
        self._library_repo = library_repository
        self._chunk_repo = chunk_repository
        self._embedding_client = embedding_client or create_embedding_client()

        logger.info("Initialized SearchService")

    def query_text(self, library_id: UUID, text: str, k: int = 10) -> SearchResult:
        """Search for similar chunks using text query"""
        # Validate basic parameters
        if not text or not text.strip():
            raise InvalidSearchParameterError("text", text, "cannot be empty")
        if k <= 0:
            raise InvalidSearchParameterError("k", k, "must be greater than 0")

        # Verify library exists
        library = self._library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))

        logger.debug(f"Generating embedding for text query in library {library_id}")

        try:
            # Generate embedding for the text query
            embedding_result = self._embedding_client.embed_text(text.strip())
            return self.query_embedding(library_id, embedding_result.single_embedding, k)
        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for text query: {e}")
            raise InvalidSearchParameterError(
                "text", text, f"embedding generation failed: {e}"
            ) from e

    def _validate_query_params(
        self, library_id: UUID, embedding: list[float], k: int
    ) -> None:
        if k <= 0:
            raise InvalidSearchParameterError("k", k, "must be greater than 0")
        if not embedding:
            raise InvalidSearchParameterError("embedding", embedding, "cannot be empty")
        
        library = self._library_repo.get_by_id(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))

    def _ensure_index_ready(self, library_id: UUID) -> IndexStatus:
        """Ensure index is built and ready for querying"""
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
        """Execute k-NN query on index"""
        # Limit k to actual index size
        effective_k = min(k, index_size)
        if effective_k != k:
            logger.debug(f"Limited k from {k} to {effective_k} based on index size")

        # Execute k-NN query through IndexService
        try:
            query_results = self._index_service.query(library_id, embedding, effective_k)
            logger.debug(f"Index query returned {len(query_results)} results")
            return query_results
        except VectorIndexNotBuiltError as e:
            raise VectorIndexNotBuiltError(str(library_id)) from e

    def _process_query_results(
        self, query_results: list[tuple[UUID, float]]
    ) -> list[tuple[Chunk, float]]:
        matches = []
        filtered_count = 0

        for chunk_id, distance in query_results:
            chunk = self._chunk_repo.get_by_id(chunk_id)
            if not chunk:
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
        """Check if chunk should be filtered by similarity threshold"""
        if not chunk.metadata or chunk.metadata.similarity_threshold is None:
            return False
        return distance > chunk.metadata.similarity_threshold

    def query_embedding(
        self, library_id: UUID, embedding: list[float], k: int = 10
    ) -> SearchResult:
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
