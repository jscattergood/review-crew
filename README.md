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
make clean         # Clean up temporary files
```

### Running Conversations
```bash
# List available review agents
make agents

# Review text content
make review ARGS='"Hello world, this is my content to review"'

# Review a file
python -m src.cli.main review path/to/your/file.txt

# Use specific agents
python -m src.cli.main review "content" --agents "Technical Reviewer" "UX Reviewer"

# Use LM Studio (local LLM)
python -m src.cli.main review "content" --provider lm_studio

# Run async reviews (faster)
python -m src.cli.main review "content" --async-mode

# Save results to file
python -m src.cli.main review "content" --output results.txt

# Custom LM Studio configuration
python -m src.cli.main review "content" --provider lm_studio --model-url http://localhost:1234/v1
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

## Configuration

Persona configuration files use YAML format:
```yaml
name: "Technical Reviewer"
role: "Senior Software Engineer"
goal: "Evaluate code quality, architecture, and best practices"
backstory: "10+ years experience in software development..."
prompt_template: "Review the following content from a technical perspective..."
```
