"""Tests for logging configuration."""

import logging
from io import StringIO
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.logging import get_logger, log_request_info, setup_logging


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

    def test_log_request_info_logs_correctly(self):
        """Test that log_request_info logs with correct format."""
        setup_logging()
        
        # Create a StringIO handler to capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter(settings.log_format_request))
        
        # Get the api.request logger and add our handler
        api_logger = get_logger("api.request")
        api_logger.handlers.clear()  # Clear existing handlers
        api_logger.addHandler(handler)
        api_logger.propagate = False
        
        log_request_info(
            method="GET",
            path="/api/v1/health",
            status_code=200,
            duration_ms=1.23,
            request_id="test-uuid-123"
        )
        
        log_output = log_capture.getvalue()
        
        # Verify log contains expected information
        assert "Request completed" in log_output
        assert "method=GET" in log_output
        assert "path=/api/v1/health" in log_output
        assert "status=200" in log_output
        assert "duration_ms=1.23" in log_output
        assert "request_id=test-uuid-123" in log_output

    def test_log_request_info_with_extra_kwargs(self):
        """Test that log_request_info handles extra kwargs correctly."""
        setup_logging()
        
        # Create a StringIO handler to capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter(settings.log_format_request))

        # Get the api.request logger and add our handler
        api_logger = get_logger("api.request")
        api_logger.handlers.clear()
        api_logger.addHandler(handler)
        api_logger.propagate = False
        
        log_request_info(
            method="POST",
            path="/api/v1/test",
            status_code=201,
            duration_ms=5.67,
            request_id="test-uuid-456",
            user_id="user123",
            extra_field="extra_value"
        )
        
        log_output = log_capture.getvalue()
        
        # Verify base fields are present
        assert "method=POST" in log_output
        assert "status=201" in log_output
        assert "request_id=test-uuid-456" in log_output

    def test_general_logging_works(self):
        """Test that general logging works correctly."""
        setup_logging()
        
        # Create a StringIO handler to capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter(settings.log_format_general))

        logger = get_logger("test.general")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        
        logger.info("Test general message")
        
        log_output = log_capture.getvalue()
        
        assert "test.general" in log_output
        assert "INFO" in log_output
        assert "Test general message" in log_output

    def test_different_log_levels(self):
        """Test that different log levels work correctly."""
        setup_logging()
        
        # Create a StringIO handler to capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter(settings.log_format_general))
        
        logger = get_logger("test.levels")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        log_output = log_capture.getvalue()
        
        # Debug should not appear (default level is INFO)
        assert "Debug message" not in log_output
        
        # Others should appear
        assert "Info message" in log_output
        assert "Warning message" in log_output
        assert "Error message" in log_output
