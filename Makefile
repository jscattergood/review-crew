# Review-Crew Makefile
# Provides convenient commands for development, testing, and running the project

.PHONY: help install setup test clean lint format check run-test personas list-personas

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
	@echo "  test-config      Test persona configuration loading"
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
	@echo "  run-example      Run a simple configuration test"
	@echo "  review           Run content review (use ARGS='content')"
	@echo "  review-lm        Run review with LM Studio (use ARGS='content')"
	@echo "  agents           List available review agents"
	@echo "  cli-status       Show CLI status and configuration"
	@echo "  test-lm-studio   Test LM Studio integration"
	@echo "  test-review      Run realistic review test (interactive)"
	@echo "  test-python      Test with Python code sample"
	@echo "  test-html        Test with HTML page sample"
	@echo "  test-docs        Test with API documentation sample"

# Installation targets
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync

install-pip:
	@echo "📦 Installing dependencies with pip..."
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	@source .venv/bin/activate && pip install -r requirements.txt

install-lm:
	@echo "📦 Installing dependencies with LM Studio support..."
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	@source .venv/bin/activate && pip install -e .[lm-studio]

dev-install:
	@echo "📦 Installing development dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra dev; \
	else \
		pip install -r requirements.txt pytest black flake8 mypy; \
	fi

# Setup targets
setup:
	@echo "🎭 Setting up persona configurations..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python setup_personas.py; \
	else \
		python3 setup_personas.py; \
	fi

# Testing targets
test: test-config validate
	@echo "✅ All tests passed!"

test-config:
	@echo "🧪 Testing persona configuration loading..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -m src.config.persona_loader; \
	else \
		python3 -m src.config.persona_loader; \
	fi

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
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_example.py; \
	else \
		python3 test_example.py; \
	fi

# Conversation targets
review:
	@echo "🎭 Starting Review-Crew CLI..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -m src.cli.main review $(ARGS); \
	else \
		python3 -m src.cli.main review $(ARGS); \
	fi

review-lm:
	@echo "🎭 Starting Review-Crew with LM Studio..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -m src.cli.main review $(ARGS) --provider lm_studio; \
	else \
		python3 -m src.cli.main review $(ARGS) --provider lm_studio; \
	fi

test-lm-studio:
	@echo "🧪 Testing LM Studio integration..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_lm_studio.py; \
	else \
		python3 test_lm_studio.py; \
	fi

test-review:
	@echo "🎭 Running realistic review test (interactive)..."
	@echo "📁 Available test files:"
	@echo "  1. Python Code (user_registration.py) - Security vulnerabilities, code quality issues"
	@echo "  2. HTML Page (product_page.html) - UX problems, accessibility issues"  
	@echo "  3. API Documentation (api_documentation.md) - Content clarity, security flaws"
	@echo ""
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_review.py; \
	else \
		python3 test_review.py; \
	fi

test-python:
	@echo "🐍 Testing with Python code (non-interactive)..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_review.py --provider 3 --file 1; \
	else \
		python3 test_review.py --provider 3 --file 1; \
	fi

test-html:
	@echo "🌐 Testing with HTML page (non-interactive)..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_review.py --provider 3 --file 2; \
	else \
		python3 test_review.py --provider 3 --file 2; \
	fi

test-docs:
	@echo "📝 Testing with API documentation (non-interactive)..."
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python test_review.py --provider 3 --file 3; \
	else \
		python3 test_review.py --provider 3 --file 3; \
	fi

agents:
	@echo "🎭 Available agents:"
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -m src.cli.main agents; \
	else \
		python3 -m src.cli.main agents; \
	fi

cli-status:
	@echo "📊 Review-Crew CLI status:"
	@if [ -f .venv/bin/activate ]; then \
		source .venv/bin/activate && python -m src.cli.main status; \
	else \
		python3 -m src.cli.main status; \
	fi

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
	@echo "2. Run 'make test' to validate changes"
	@echo "3. Start building your review workflow!"
