# Review-Crew Makefile
# Provides convenient commands for development, testing, and running the project

.PHONY: help install setup test clean lint format check run-example personas validate

# Helper to source .env file and activate virtual environment
define run_with_env
	@if [ -f .env ]; then \
		echo "📄 Loading .env file..."; \
	fi; \
	if [ -f .venv/bin/activate ]; then \
		set -a; [ -f .env ] && source .env; set +a; source .venv/bin/activate && $(1); \
	else \
		set -a; [ -f .env ] && source .env; set +a; $(2); \
	fi
endef

# Default target
help:
	@echo "Review-Crew Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install dependencies using uv"
	@echo "  install-pip      Install dependencies using pip"
	@echo "  install-lm       Install with LM Studio support (includes OpenAI)"
	@echo "  setup            Set up custom persona configurations"
	@echo ""
	@echo "Testing & Validation:"
	@echo "  test             Run all tests"
	@echo "  personas         List all available personas"
	@echo "  validate         Validate all persona configurations"
	@echo ""
	@echo "Development:"
	@echo "  lint             Run linting (flake8)"
	@echo "  format           Format code (black)"
	@echo "  check            Run type checking (mypy)"
	@echo "  dev-install      Install development dependencies"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean up temporary files"
	@echo "  clean-config     Remove custom persona configurations"
	@echo ""
	@echo "Examples & Usage:"
	@echo "  run-example      Run example integration test"
	@echo "  review           Run content review (use ARGS='content')"
	@echo "  review-lm        Run review with LM Studio (use ARGS='content')"
	@echo "  review-async     Run async review (faster, parallel agents)"
	@echo "  review-lm-async  Run async review with LM Studio (fastest)"
	@echo "  agents           List available review agents"
	@echo "  cli-status       Show CLI status and configuration"

# Installation targets
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --group dev

# Setup targets
setup:
	@echo "🎭 Setting up persona configurations..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python setup_personas.py; \
	else \
		python3 setup_personas.py; \
	fi

# Testing targets
test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v
	@echo "✅ All tests passed!"

personas:
	@echo "🎭 Available personas:"
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); [print(f'  - {p}') for p in loader.list_available_personas()]" 2>/dev/null || echo "  No personas found. Run 'make setup' first."; \
	else \
		python3 -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); [print(f'  - {p}') for p in loader.list_available_personas()]" 2>/dev/null || echo "  No personas found. Run 'make setup' first."; \
	fi

validate:
	@echo "✅ Validating persona configurations..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_personas(); print(f'✅ Successfully loaded {len(personas)} personas'); [print(f'  ✓ {p.name} ({p.role})') for p in personas]" 2>/dev/null || echo "❌ Validation failed. Check your configurations."; \
	else \
		python3 -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_personas(); print(f'✅ Successfully loaded {len(personas)} personas'); [print(f'  ✓ {p.name} ({p.role})') for p in personas]" 2>/dev/null || echo "❌ Validation failed. Check your configurations."; \
	fi

# Development targets
lint:
	@echo "🔍 Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ --max-line-length=88 --extend-ignore=E203,W503; \
	else \
		echo "flake8 not installed. Run 'make dev-install' first."; \
	fi

format:
	@echo "🎨 Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ setup_personas.py; \
	else \
		echo "black not installed. Run 'make dev-install' first."; \
	fi

check:
	@echo "🔍 Running type checking..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy src/ --ignore-missing-imports; \
	else \
		echo "mypy not installed. Run 'make dev-install' first."; \
	fi

# Example and demo targets
run-example:
	@echo "🧪 Running example integration test..."
	uv run pytest tests/test_integration.py::TestIntegration::test_mock_review_workflow -v

# Conversation targets
review:
	@echo "🎭 Starting Review-Crew CLI..."
	$(call run_with_env,python -m src.cli.main review $(ARGS),python3 -m src.cli.main review $(ARGS))

review-lm:
	@echo "🎭 Starting Review-Crew with LM Studio..."
	$(call run_with_env,python -m src.cli.main review $(ARGS) --provider lm_studio,python3 -m src.cli.main review $(ARGS) --provider lm_studio)

review-async:
	@echo "🎭 Starting Review-Crew CLI (async mode)..."
	$(call run_with_env,python -m src.cli.main review $(ARGS) --async-mode,python3 -m src.cli.main review $(ARGS) --async-mode)

review-lm-async:
	@echo "🎭 Starting Review-Crew with LM Studio (async mode)..."
	$(call run_with_env,python -m src.cli.main review $(ARGS) --provider lm_studio --async-mode,python3 -m src.cli.main review $(ARGS) --provider lm_studio --async-mode)



agents:
	@echo "🎭 Available agents:"
	$(call run_with_env,python -m src.cli.main agents,python3 -m src.cli.main agents)

cli-status:
	@echo "📊 Review-Crew CLI status:"
	$(call run_with_env,python -m src.cli.main status,python3 -m src.cli.main status)

# Cleanup targets
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean-config:
	@echo "🗑️  Removing custom persona configurations..."
	@if [ -d "config/personas" ]; then \
		rm -rf config/personas/; \
		echo "Custom personas removed. Run 'make setup' to recreate."; \
	else \
		echo "No custom personas to remove."; \
	fi

# Quick development workflow
dev: dev-install setup test
	@echo "🎉 Development environment ready!"

# Show project status
status:
	@echo "📊 Project Status"
	@echo "=================="
	@echo "Python version: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Working directory: $$(pwd)"
	@echo ""
	@echo "Dependencies:"
	@if command -v uv >/dev/null 2>&1; then echo "  ✅ uv available"; else echo "  ❌ uv not found"; fi
	@if python3 -c "import yaml" 2>/dev/null; then echo "  ✅ PyYAML installed"; else echo "  ❌ PyYAML not found"; fi
	@if python3 -c "import strands" 2>/dev/null; then echo "  ✅ Strands Agents installed"; else echo "  ❌ Strands Agents not found"; fi
	@echo ""
	@echo "Configuration:"
	@if [ -d "examples/personas" ]; then echo "  ✅ Example personas available"; else echo "  ❌ Example personas missing"; fi
	@if [ -d "config/personas" ]; then echo "  ✅ Custom personas configured"; else echo "  ⚠️  Custom personas not set up (run 'make setup')"; fi
	@echo ""
	@make personas 2>/dev/null || echo "  ❌ Cannot load personas"

# Full project initialization for new users
init: install setup test
	@echo ""
	@echo "🎉 Review-Crew initialization complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Customize personas in config/personas/"
	@echo "2. Run 'make test' to run all tests"
	@echo "3. Start building your review workflow!"
