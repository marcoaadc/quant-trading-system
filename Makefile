.PHONY: install lint format test typecheck ci ingest pipeline clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies (with dev extras)
	pip install -e ".[dev]"

lint: ## Run ruff linter and format checker
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code with ruff
	ruff check --fix src/ tests/
	ruff format src/ tests/

test: ## Run tests with pytest and coverage
	pytest --cov=src --cov-report=term-missing --cov-report=html tests/

typecheck: ## Run mypy type checking
	mypy src/

ci: ## Run full CI pipeline (lint + typecheck + test)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

ingest: ## Run OHLCV data ingestion pipeline
	python -m src.data.ingestion

pipeline: ## Run full regime detection pipeline
	python -c "from src.pipeline import RegimePipeline; RegimePipeline.from_config('configs/default.yaml').run_full()"

clean: ## Remove temporary and generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
