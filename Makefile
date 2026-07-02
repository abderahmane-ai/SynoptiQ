.PHONY: help install check fmt lint typecheck test test-fast clean data

PYTHON := python3.12
PIP := pip
PKG_DIR := synoptiq
TEST_DIR := tests
SCRIPT_DIR := scripts

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────────────────────────

install:  ## Install package in editable mode with dev dependencies
	$(PIP) install -e ".[dev]"

install-all:  ## Install with all optional dependencies
	$(PIP) install -e ".[all]"

# ── Quality gates (run ALL before every commit) ───────────────────────────────

check: lint typecheck test  ## Run full quality gate: lint + typecheck + test

fmt:  ## Auto-format code with ruff
	ruff format $(PKG_DIR)/ $(TEST_DIR)/ $(SCRIPT_DIR)/
	ruff check --fix $(PKG_DIR)/ $(TEST_DIR)/ $(SCRIPT_DIR)/

lint:  ## Check code style and linting (ruff)
	ruff check $(PKG_DIR)/ $(TEST_DIR)/ $(SCRIPT_DIR)/
	ruff format --check $(PKG_DIR)/ $(TEST_DIR)/ $(SCRIPT_DIR)/

typecheck:  ## Run static type checking (mypy strict)
	mypy $(PKG_DIR)/

test:  ## Run all tests (excluding slow/gpu/downloads)
	python -m pytest $(TEST_DIR)/ -m "not slow and not gpu and not downloads" \
		-q --tb=short

test-fast:  ## Run only fast unit tests (no integration)
	python -m pytest $(TEST_DIR)/ -m "not slow and not gpu and not downloads and not integration" \
		-q --tb=short

test-all:  ## Run all tests including slow/integration (requires full corpus)
	python -m pytest $(TEST_DIR)/ -q --tb=short

test-cov:  ## Run tests with coverage report
	python -m pytest $(TEST_DIR)/ -m "not slow and not gpu and not downloads" \
		--cov=$(PKG_DIR) --cov-report=term-missing -q

# ── Data pipeline ─────────────────────────────────────────────────────────────

data:  ## Download and prepare all corpora (Phase 1)
	$(PYTHON) $(SCRIPT_DIR)/prepare_data.py

data-download:  ## Download raw corpora only (no processing)
	$(PYTHON) $(SCRIPT_DIR)/prepare_data.py --download-only

data-check:  ## Validate the processed corpus
	$(PYTHON) -c "from synoptiq import Corpus; c = Corpus.from_parquet('data/processed/tokens.parquet', 'data/processed/pericopes.parquet'); print(f'Tokens: {c.n_tokens:,}  Pericopes: {c.n_pericopes}')"

# ── Modal (cloud training) ────────────────────────────────────────────────────

modal-check:  ## Verify Modal setup and GPU access
	modal run modal/_common.py::check_gpu

train-dapt:  ## Run DAPT on Modal A10G
	modal run modal/app_dapt.py::train_dapt

train-direction:  ## Run direction scorer training on Modal T4
	modal run modal/app_train.py::train_direction

eval-all:  ## Run full evaluation suite
	$(PYTHON) $(SCRIPT_DIR)/eval_all.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ dist/ *.egg-info/ 2>/dev/null; true

clean-data:  ## Remove processed data (keeps raw downloads)
	rm -rf data/processed/

clean-models:  ## Remove trained model checkpoints
	rm -rf models/
