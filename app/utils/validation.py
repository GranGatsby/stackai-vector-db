"""Common validation utilities."""

from app.core.config import settings


def validate_non_empty_text(
    text: str, error_message: str = "Cannot process empty text"
) -> str:
    """Validate that text is not empty or whitespace-only."""
    if not text or not text.strip():
        raise ValueError(error_message)
    return text.strip()


def validate_name_length(name: str, field_name: str = "Name") -> str:
    """Validate name length against configured limits."""
    stripped = validate_non_empty_text(name, f"{field_name} cannot be empty")
    if len(stripped) > settings.max_name_length:
        raise ValueError(
            f"{field_name} cannot exceed {settings.max_name_length} characters"
        )
    return stripped


def validate_title_length(title: str, field_name: str = "Title") -> str:
    """Validate title length against configured limits."""
    stripped = validate_non_empty_text(title, f"{field_name} cannot be empty")
    if len(stripped) > settings.max_title_length:
        raise ValueError(
            f"{field_name} cannot exceed {settings.max_title_length} characters"
        )
    return stripped


def validate_non_negative(value: int, field_name: str) -> int:
    """Validate that integer value is non-negative."""
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def validate_index_range(
    start: int, end: int, field_prefix: str = ""
) -> tuple[int, int]:
    """Validate that end index is >= start index."""
    if end < start:
        if field_prefix:
            raise ValueError(f"{field_prefix}end_index must be >= start_index")
        else:
            raise ValueError("End index must be >= start index")
    return start, end
