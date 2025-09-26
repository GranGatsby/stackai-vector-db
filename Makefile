# StackAI Vector Database
# Makefile for development and deployment

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help install dev clean test lint format check run build docker ci

# Configuration
APP_NAME := stackai-vector-db
DOCKER_TAG := latest
PORT := 8000

## Display available commands
help:
	@echo "$(APP_NAME) - Development Commands"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development
install: ## Install production dependencies
	@pip install -e .

dev: ## Setup development environment
	@pip install -e ".[dev]" && pre-commit install

clean: ## Clean cache and build artifacts
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

##@ Quality Assurance
test: ## Run test suite with coverage
	@pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

lint: ## Check code quality
	@ruff check app/ tests/

format: ## Format code and fix issues
	@black app/ tests/ && ruff format app/ tests/ && ruff check app/ tests/ --fix

check: lint ## Run all quality checks
	@mypy app/
	@echo "✅ All quality checks passed"

##@ Application
run: ## Start development server
	@python -m app.main

build: ## Build distribution package
	@python -m build

##@ Docker
docker: ## Build and run Docker container
	@docker build -t $(APP_NAME):$(DOCKER_TAG) .
	@docker run -p $(PORT):$(PORT) --env-file .env $(APP_NAME):$(DOCKER_TAG)

##@ CI/CD
ci: format check test ## Run complete CI pipeline
	@echo "🚀 CI pipeline completed successfully"
