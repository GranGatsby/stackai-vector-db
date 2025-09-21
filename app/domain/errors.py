"""Domain-specific exceptions for the vector database.

This module contains all domain-level exceptions that represent business
rule violations and error conditions within the vector database domain.
These exceptions are framework-agnostic and should be mapped to appropriate
HTTP responses at the API layer.
"""


class DomainError(Exception):
    """Base class for all domain-specific errors.

    This serves as the root exception for all business logic errors
    and provides a consistent interface for error handling.
    """

    def __init__(self, message: str, code: str = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__


class LibraryError(DomainError):
    """Base class for library-related errors."""

    pass


class LibraryNotFoundError(LibraryError):
    """Raised when a requested library cannot be found."""

    def __init__(self, library_id: str) -> None:
        message = f"Library with ID '{library_id}' not found"
        super().__init__(message, "LIBRARY_NOT_FOUND")
        self.library_id = library_id


class LibraryAlreadyExistsError(LibraryError):
    """Raised when attempting to create a library with a duplicate name."""

    def __init__(self, name: str) -> None:
        message = f"Library with name '{name}' already exists"
        super().__init__(message, "LIBRARY_ALREADY_EXISTS")
        self.name = name


class DocumentError(DomainError):
    """Base class for document-related errors."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a requested document cannot be found."""

    def __init__(self, document_id: str) -> None:
        message = f"Document with ID '{document_id}' not found"
        super().__init__(message, "DOCUMENT_NOT_FOUND")
        self.document_id = document_id


class DocumentAlreadyExistsError(DocumentError):
    """Raised when attempting to create a document with a duplicate title in a library."""

    def __init__(self, title: str, library_id: str) -> None:
        message = (
            f"Document with title '{title}' already exists in library '{library_id}'"
        )
        super().__init__(message, "DOCUMENT_ALREADY_EXISTS")
        self.title = title
        self.library_id = library_id


class ChunkError(DomainError):
    """Base class for chunk-related errors."""

    pass


class ChunkNotFoundError(ChunkError):
    """Raised when a requested chunk cannot be found."""

    def __init__(self, chunk_id: str) -> None:
        message = f"Chunk with ID '{chunk_id}' not found"
        super().__init__(message, "CHUNK_NOT_FOUND")
        self.chunk_id = chunk_id


class IndexError(DomainError):
    """Base class for index-related errors."""

    pass


class IndexNotBuiltError(IndexError):
    """Raised when attempting to query an index that hasn't been built."""

    def __init__(self, library_id: str) -> None:
        message = f"Index for library '{library_id}' has not been built"
        super().__init__(message, "INDEX_NOT_BUILT")
        self.library_id = library_id


class IndexBuildError(IndexError):
    """Raised when index building fails."""

    def __init__(self, library_id: str, reason: str) -> None:
        message = f"Failed to build index for library '{library_id}': {reason}"
        super().__init__(message, "INDEX_BUILD_FAILED")
        self.library_id = library_id
        self.reason = reason


class EmbeddingError(DomainError):
    """Base class for embedding-related errors."""

    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match expected dimensions."""

    def __init__(self, expected: int, actual: int) -> None:
        message = f"Embedding dimension mismatch: expected {expected}, got {actual}"
        super().__init__(message, "EMBEDDING_DIMENSION_MISMATCH")
        self.expected = expected
        self.actual = actual


class ValidationError(DomainError):
    """Raised when domain entity validation fails."""

    pass
