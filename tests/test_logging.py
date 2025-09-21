"""Tests for logging configuration."""

import logging

import pytest

from app.core.logging import get_logger, log_request_info, setup_logging
from tests.conftest import capture_logger


class TestLogging:
    """Test cases for logging configuration."""

    def test_setup_logging_configures_correctly(self):
        """Test that setup_logging configures logging without errors."""
        # This should not raise any exceptions
        setup_logging()

        # Verify root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_api_request_logger_exists(self):
        """Test that api.request logger is configured."""
        setup_logging()

        api_logger = get_logger("api.request")
        assert isinstance(api_logger, logging.Logger)
        assert api_logger.name == "api.request"

    def test_log_request_info_logs_correctly(self, caplog):
        """Test that log_request_info logs with correct format."""
        setup_logging()
        with capture_logger(caplog, "api.request"):
            log_request_info(
                method="GET",
                path="/api/v1/health",
                status_code=200,
                duration_ms=1.23,
                request_id="test-uuid-123",
            )

        # Verify log was captured
        assert len(caplog.records) == 1
        record = caplog.records[0]

        # Verify log metadata
        assert record.name == "api.request"
        assert record.levelname == "INFO"
        assert record.getMessage() == "Request completed"

        # Verify extra fields are present
        assert record.method == "GET"
        assert record.path == "/api/v1/health"
        assert record.status_code == 200
        assert record.duration_ms == 1.23
        assert record.request_id == "test-uuid-123"

    def test_log_request_info_with_extra_kwargs(self, caplog):
        """Test that log_request_info handles extra kwargs correctly."""
        setup_logging()
        with capture_logger(caplog, "api.request"):
            log_request_info(
                method="POST",
                path="/api/v1/test",
                status_code=201,
                duration_ms=5.67,
                request_id="test-uuid-456",
                user_id="user123",
                extra_field="extra_value",
            )

        # Verify log was captured
        assert len(caplog.records) == 1
        record = caplog.records[0]

        # Verify base fields are present
        assert record.method == "POST"
        assert record.path == "/api/v1/test"
        assert record.status_code == 201
        assert record.duration_ms == 5.67
        assert record.request_id == "test-uuid-456"

        # Verify extra kwargs are captured
        assert record.user_id == "user123"
        assert record.extra_field == "extra_value"

    def test_general_logging_works(self, caplog):
        """Test that general logging works correctly."""
        setup_logging()
        logger = get_logger("test.general")
        with capture_logger(caplog, "test.general"):
            logger.info("Test general message")

        # Verify log was captured
        assert len(caplog.records) == 1
        record = caplog.records[0]

        assert record.name == "test.general"
        assert record.levelname == "INFO"
        assert record.getMessage() == "Test general message"

    def test_different_log_levels(self, caplog):
        """Test that different log levels work correctly."""
        setup_logging()

        logger = get_logger("test.levels")

        with capture_logger(caplog, "test.levels"):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        # Verify only INFO and above were captured
        messages = [record.getMessage() for record in caplog.records]

        # Debug should not appear (level is INFO)
        assert "Debug message" not in messages

        # Others should appear
        assert "Info message" in messages
        assert "Warning message" in messages
        assert "Error message" in messages

        # Verify log levels
        levels = [record.levelname for record in caplog.records]
        assert "DEBUG" not in levels
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels
