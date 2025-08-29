# Review-Crew Makefile
# Provides convenient commands for development, testing, and running the project

.PHONY: help install setup test clean lint format check run-example personas persona-types validate

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
	@echo ""
	@echo "Testing & Validation:"
	@echo "  test             Run all tests"
	@echo "  personas         List all available personas by type"
	@echo "  persona-types    Show persona type breakdown summary"
	@echo "  validate         Validate all persona configurations"
	@echo ""
	@echo "Development:"
	@echo "  lint             Run linting (flake8)"
	@echo "  format           Format code (black)"
	@echo "  check            Run type checking (mypy)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean up temporary files"
	@echo "  clean-config     Remove custom persona configurations"
	@echo ""
	@echo "Examples & Usage:"
	@echo "  run-example      Run example integration test"
	@echo "  review           Run content review (use ARGS='content')"
	@echo "  review-lm        Run review with LM Studio (use ARGS='content')"

	@echo "  agents           List available review agents"
	@echo "  cli-status       Show CLI status and configuration"

# Installation targets
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync --group dev
# Testing targets
test:
	@echo "🧪 Running all tests..."
	uv run pytest tests/ -v
	@echo "✅ All tests passed!"

personas:
	@echo "🎭 Available personas by type:"
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); total = sum(len(p) for p in personas.values()); print(f'Total: {total} personas loaded'); [print(f'\n📋 {ptype.upper()} ({len(plist)}):') or [print(f'  - {p.name}') for p in plist] for ptype, plist in personas.items() if plist]" 2>/dev/null || echo "  No personas found. Check your configuration."; \
	else \
		python3 -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); total = sum(len(p) for p in personas.values()); print(f'Total: {total} personas loaded'); [print(f'\n📋 {ptype.upper()} ({len(plist)}):') or [print(f'  - {p.name}') for p in plist] for ptype, plist in personas.items() if plist]" 2>/dev/null || echo "  No personas found. Check your configuration."; \
	fi

validate:
	@echo "✅ Validating persona configurations..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); total = sum(len(p) for p in personas.values()); print(f'✅ Successfully loaded {total} personas across {len([k for k, v in personas.items() if v])} types'); [print(f'\n📋 {ptype.upper()} ({len(plist)}):') or [print(f'  ✓ {p.name} ({p.role})') for p in plist] for ptype, plist in personas.items() if plist]" 2>/dev/null || echo "❌ Validation failed. Check your configurations."; \
	else \
		python3 -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); total = sum(len(p) for p in personas.values()); print(f'✅ Successfully loaded {total} personas across {len([k for k, v in personas.items() if v])} types'); [print(f'\n📋 {ptype.upper()} ({len(plist)}):') or [print(f'  ✓ {p.name} ({p.role})') for p in plist] for ptype, plist in personas.items() if plist]" 2>/dev/null || echo "❌ Validation failed. Check your configurations."; \
	fi

persona-types:
	@echo "🎭 Persona types breakdown:"
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); print('\\n'.join([f'{ptype.capitalize()}: {len(plist)} personas' for ptype, plist in personas.items()])); print(f'\\nTotal: {sum(len(p) for p in personas.values())} personas')" 2>/dev/null || echo "  No personas found. Check your configuration."; \
	else \
		python3 -c "from src.config.persona_loader import PersonaLoader; loader = PersonaLoader(); personas = loader.load_all_persona_types(); print('\\n'.join([f'{ptype.capitalize()}: {len(plist)} personas' for ptype, plist in personas.items()])); print(f'\\nTotal: {sum(len(p) for p in personas.values())} personas')" 2>/dev/null || echo "  No personas found. Check your configuration."; \
	fi

# Development targets
lint:
	@echo "🔍 Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/; \
	else \
		echo "flake8 not installed. Run 'make dev-install' first."; \
	fi

format:
	@echo "🎨 Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/; \
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
	@make persona-types 2>/dev/null || echo "  ❌ Cannot load personas"

# Full project initialization for new users
init: install setup test
	@echo ""
	@echo "🎉 Review-Crew initialization complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Customize personas in config/personas/"
	@echo "2. Run 'make test' to run all tests"
	@echo "3. Start building your review workflow!"
