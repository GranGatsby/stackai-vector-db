# Makefile for StackAI Vector Database

.PHONY: help install install-dev setup clean test lint format type-check pre-commit run build docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup        Setup development environment"
	@echo "  clean        Clean cache and temporary files"
	@echo "  test         Run tests"
	@echo "  lint         Run linting with ruff"
	@echo "  format       Format code with black and ruff"
	@echo "  type-check   Run type checking with mypy"
	@echo "  pre-commit   Run pre-commit hooks"
	@echo "  run          Run the application locally"
	@echo "  build        Build the package"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

setup: install-dev
	pre-commit install

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Testing
test:
	pytest tests/ -v --cov=app --cov-report=term-missing

# Code quality
lint:
	ruff check app/ tests/

format:
	black app/ tests/
	ruff format app/ tests/
	ruff check app/ tests/ --fix

type-check:
	mypy app/

pre-commit:
	pre-commit run --all-files

# Development
run:
	python -m app.main

# Build
build:
	python -m build

# Docker
docker-build:
	docker build -t stackai-vector-db:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env stackai-vector-db:latest

# Quality checks (all)
check: lint type-check test

# CI pipeline
ci: format check
