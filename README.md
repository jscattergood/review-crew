# Review-Crew

## Purpose
This repo implements a multi-agent workflow where agents take on different personas to evaluate content from their given perspective. The goal is to gather each agent's respective feedback. The user can then use that feedback to improve the content and resubmit it for evaluation.

## Technology
* **Language**: Python
* **Package Management**: uv
* **Agent Framework**: [Strands Agents](https://strandsagents.com/latest/documentation/docs/) - AWS-backed, production-ready multi-agent framework
* **Interface**: CLI
* **LLM**: LM Studio + Available Models (configurable model providers)

## Architecture

### Configuration-Driven Personas
- **Example Personas**: Tracked in `examples/personas/` for reference and getting started
- **Custom Personas**: Untracked in `config/personas/` for private/custom reviewer configurations
- **Flexible Setup**: Configure any number of reviewer agents with custom prompts and roles

### Project Structure
```
review-crew/
├── src/
│   ├── agents/          # Agent implementations
│   ├── config/          # Configuration loaders
│   └── cli/             # Command-line interface
├── examples/
│   └── personas/        # Example persona configurations (tracked)
├── config/
│   └── personas/        # Custom persona configurations (untracked)
└── requirements.txt     # Dependencies
```

## Getting Started

### Quick Start
```bash
# Install dependencies and set up project
make install-pip    # or 'make install' if you have uv
make setup         # Copy example personas to config/
make test          # Validate configuration

# View available personas
make personas

# Run example
make run-example

# Test with realistic samples (no LLM calls)
make test-python   # Test with Python code sample
make test-html     # Test with HTML page sample
make test-docs     # Test with API documentation sample
make test-review   # Interactive test with all options

# Test LM Studio integration (if you have LM Studio running)
make test-lm-studio
```

### Manual Setup
1. **Install dependencies**: 
   - With uv: `uv sync`
   - With pip: `pip install -r requirements.txt`
   - For LM Studio: `pip install -e .[lm-studio]` (includes OpenAI package)
2. **Setup personas**: `python setup_personas.py` (copies examples to config/)
3. **Customize personas**: Edit files in `config/personas/` to match your review needs
4. **Test configuration**: `python -m src.config.persona_loader`
5. **Run review**: `python -m src.cli.main review "your content here"`

### Development Commands
```bash
make help          # Show all available commands
make status        # Check project status
make format        # Format code with black
make lint          # Run linting with flake8
make check         # Run type checking with mypy
make clean         # Clean up temporary files
make dev-install   # Install development dependencies
```

### Running Conversations

#### Quick Commands (Makefile)
```bash
# List available review agents
make agents

# Review text content
make review ARGS='"Hello world, this is my content to review"'

# Review with LM Studio (local LLM) - RECOMMENDED
make review-lm ARGS='"Your content here"'

# Review a file
make review ARGS='"$(cat path/to/your/file.txt)"'
```

#### Advanced CLI Usage
```bash
# Review a file directly
python -m src.cli.main review path/to/your/file.txt

# Use specific agents
python -m src.cli.main review "content" -a "Technical Reviewer" -a "UX Reviewer"

# Use different providers
python -m src.cli.main review "content" --provider lm_studio
python -m src.cli.main review "content" --provider ollama
python -m src.cli.main review "content" --provider bedrock  # default

# Run async reviews (faster)
python -m src.cli.main review "content" --async-mode

# Save results to file
python -m src.cli.main review "content" -o results.txt

# Custom model configuration
python -m src.cli.main review "content" --provider lm_studio --model-url http://localhost:1234/v1
python -m src.cli.main review "content" --provider ollama --model-id llama2
```

### LLM Configuration
Before running reviews, configure your LLM provider:

**Option 1: AWS Bedrock (Default)**
- Configure AWS credentials: `aws configure`
- Enable Claude models in AWS Bedrock console
- Set region: `export AWS_REGION=us-west-2`

**Option 2: LM Studio (Local) ⭐ RECOMMENDED**
- Start LM Studio and load a model
- Enable the local server (default: http://localhost:1234)
- Use `--provider lm_studio` flag when running reviews
- Your existing personas work automatically with LM Studio!
- Example: `python -m src.cli.main review "content" --provider lm_studio`

**Option 3: Other Providers**
- See [Strands documentation](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/) for other providers

## Testing & Examples

### Realistic Test Inputs
The project includes realistic test content designed to trigger different types of feedback:

- **`test_inputs/user_registration.py`** - Python code with security vulnerabilities, code quality issues
- **`test_inputs/product_page.html`** - HTML page with UX problems, accessibility issues  
- **`test_inputs/api_documentation.md`** - API documentation with content clarity and security flaws

### Test Commands
```bash
# Non-interactive tests (great for CI/CD)
make test-python   # Review Python code sample
make test-html     # Review HTML page sample
make test-docs     # Review API documentation sample

# Interactive test with provider selection
make test-review   # Choose provider and test file interactively

# Integration tests
make test-lm-studio  # Test LM Studio connection
make test-config     # Validate persona configurations
```

### Expected Review Types
Each test input is designed to demonstrate different agent capabilities:

- **Technical Reviewer**: Code quality, security vulnerabilities, architecture issues
- **Security Reviewer**: Authentication flaws, data exposure, compliance violations  
- **UX Reviewer**: Accessibility problems, user experience issues, design inconsistencies
- **Content Reviewer**: Clarity issues, missing information, tone problems

## Configuration

Persona configuration files use YAML format:
```yaml
name: "Technical Reviewer"
role: "Senior Software Engineer"
goal: "Evaluate code quality, architecture, and best practices"
backstory: "10+ years experience in software development..."
prompt_template: "Review the following content from a technical perspective..."
model_config:
  temperature: 0.3
  max_tokens: 1500
```

## Troubleshooting

### Common Issues

**LM Studio Connection Issues:**
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Test with explicit URL
python -m src.cli.main review "test" --provider lm_studio --model-url http://localhost:1234/v1
```

**Missing Dependencies:**
```bash
# For LM Studio support
pip install -e .[lm-studio]

# For Ollama support  
pip install -e .[ollama]

# Reinstall everything
make clean && make install-pip && make setup
```

**Persona Configuration Errors:**
```bash
# Validate configurations
make test-config

# Reset to defaults
make clean-config && make setup

# Check specific persona
python -c "from src.config.persona_loader import PersonaLoader; PersonaLoader().load_personas()"
```

**Virtual Environment Issues:**
```bash
# Create fresh environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
make install-pip
```
