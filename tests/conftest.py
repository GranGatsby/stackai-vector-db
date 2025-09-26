"""Shared test fixtures and configuration."""

import contextlib
import logging
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_cohere_env():
    """Mock COHERE_API_KEY environment variable for tests."""
    with patch.dict(os.environ, {"COHERE_API_KEY": "test-key-123"}):
        yield


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests to avoid interference."""
    import logging

    # Store original state
    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level

    yield

    # Restore original state
    logging.root.handlers = original_handlers
    logging.root.level = original_level


@contextlib.contextmanager
def capture_logger(
    caplog: pytest.LogCaptureFixture, logger_name: str, level: int = logging.INFO
):
    logger = logging.getLogger(logger_name)
    with caplog.at_level(level, logger=logger_name):
        logger.addHandler(caplog.handler)
        try:
            yield
        finally:
            logger.removeHandler(caplog.handler)
