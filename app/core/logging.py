"""Logging configuration for the application."""

import logging
from logging.config import dictConfig
from typing import Any

from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging."""
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "generic": {
                    "format": settings.log_format_general,
                },
                "request": {
                    "format": settings.log_format_request,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "generic",
                },
                "api_console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "request",
                },
            },
            "root": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
            },
            "loggers": {
                "api.request": {
                    "level": settings.log_level.upper(),
                    "handlers": ["api_console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "WARNING",
                },
                "uvicorn.error": {
                    "level": "INFO",
                },
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance by name."""
    return logging.getLogger(name)


def log_request_info(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log structured request information."""
    logger = get_logger("api.request")
    logger.info(
        "Request completed",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )
