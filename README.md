# Review-Crew

## Purpose
Review-Crew is a powerful, generic multi-agent review platform that uses AI agents with different personas to evaluate any type of content. The system provides comprehensive feedback from multiple perspectives, then synthesizes all reviews through intelligent analysis agents. Perfect for content review, code analysis, document evaluation, and more.

### Key Features
- **Multi-Agent Reviews**: Configure any number of reviewer agents with custom personas
- **Intelligent Analysis**: Synthesis agents integrate feedback, resolve conflicts, and prioritize recommendations  
- **Generic & Extensible**: Works with any content type - not limited to specific domains
- **Smart Context Management**: Automatic chunking for large reviews with smaller models
- **Multiple LLM Providers**: AWS Bedrock, LM Studio, Ollama support
- **Async Processing**: Fast parallel reviews for better performance

## Technology
* **Language**: Python
* **Package Management**: uv
* **Agent Framework**: [Strands Agents](https://strandsagents.com/latest/documentation/docs/) - AWS-backed, production-ready multi-agent framework
* **Interface**: CLI
* **LLM**: LM Studio + Available Models (configurable model providers)

## Architecture

### Three-Stage Review Process
1. **Context Stage**: Contextualizer agents process and format context information (optional)
2. **Review Stage**: Multiple reviewer agents provide individual feedback from their specialized perspectives
3. **Analysis Stage**: Analysis agents synthesize all feedback, resolve conflicts, and provide actionable insights

### Configuration-Driven Personas
- **Contextualizer Personas**: Process and format context information before reviews
- **Reviewer Personas**: Domain experts, consultants, specialists (e.g., technical, security, UX reviewers)
- **Analyzer Personas**: Meta-analysis, sentiment analysis, quality metrics, custom synthesis types
- **Multiple Agent Support**: Configure any number of agents of each type with custom prompts and roles
- **Generic Design**: Works with any content type, not domain-specific

### Project Structure
```
review-crew/
├── src/
│   ├── agents/          # Agent implementations (ReviewAgent, AnalysisAgent)
│   ├── config/          # Configuration loaders (PersonaLoader)
│   └── cli/             # Command-line interface
├── examples/
│   └── personas/        # Example persona configurations (tracked)
│       ├── contextualizers/  # Context processing agent personas
│       ├── reviewers/   # Review agent personas
│       └── analyzers/   # Analysis agent personas
├── config/
│   └── personas/        # Custom persona configurations (untracked)
│       ├── contextualizers/  # Your custom contextualizer personas
│       ├── reviewers/   # Your custom review personas
│       └── analyzers/   # Your custom analysis personas
└── pyproject.toml      # Dependencies (uv)
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
   - For LM Studio: `uv sync --extra lm-studio` (includes OpenAI package)
   - For development: `uv sync --group dev` (includes testing tools)
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
make dev-install   # Install development dependencies (uses uv sync --group dev)
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

# Fast async reviews (parallel agents)
make review-async ARGS='"Your content here"'
make review-lm-async ARGS='"Your content here"'  # Fastest option!

# Review a file
make review ARGS='"$(cat path/to/your/file.txt)"'

# Review with output file
make review-lm ARGS='path/to/file.txt --output results.txt'

# Pipe content with options
cat essay.txt | make review-lm-async ARGS='--output results.txt --async-mode'
```

**Important:** Makefile commands use `ARGS='...'` syntax, not direct command-line arguments.

#### Advanced CLI Usage
```bash
# Review a file directly
python -m src.cli.main review path/to/your/file.txt

# Pipe content from stdin
cat essay.txt | python -m src.cli.main review
echo "Review this text" | make review-lm

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

# Disable analysis (reviewers only)
python -m src.cli.main review "content" --no-analysis

# Configure context length for chunking (default: 4096)
python -m src.cli.main review "content" --max-context-length 8192

# Custom model configuration
python -m src.cli.main review "content" --provider lm_studio --model-url http://localhost:1234/v1
python -m src.cli.main review "content" --provider ollama --model-id llama2

# Context processing with contextualizers
python -m src.cli.main review "content" --context context_file.txt
python -m src.cli.main review "content" --context context_file.txt --include-context
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

**Contextualizer Agents:**
- **Business Context Formatter**: Formats business context and requirements for reviews
- **Context Processor**: Processes and structures context information for better review focus

**Reviewer Agents:**
- **Technical Reviewer**: Code quality, security vulnerabilities, architecture issues
- **Security Reviewer**: Authentication flaws, data exposure, compliance violations  
- **UX Reviewer**: Accessibility problems, user experience issues, design inconsistencies
- **Content Reviewer**: Clarity issues, missing information, tone problems

**Analysis Agents:**
- **Meta-Analysis**: Synthesizes all feedback, resolves conflicts, prioritizes recommendations
- **Sentiment Analysis**: Analyzes tone, emotional impact, and communication effectiveness
- **Quality Metrics Analyzer**: Provides quantitative analysis and quality scoring

## Configuration

### Environment Configuration

The system uses a single environment variable to configure the personas directory:

**Default Behavior (Testing):**
- Uses `examples/personas/` (4 standard review personas)
- Perfect for testing and getting started

**Production Setup:**
- Create a `.env` file in the project root
- Set `REVIEW_CREW_PERSONAS_DIR=config/personas` to use your custom personas
- The Makefile automatically loads `.env` files when running CLI commands

```bash
# .env file (already created for you)
REVIEW_CREW_PERSONAS_DIR=config/personas

# To use standard test personas, comment out or change to:
# REVIEW_CREW_PERSONAS_DIR=examples/personas
```

**Manual Override:**
```bash
# Temporarily use different personas
export REVIEW_CREW_PERSONAS_DIR="/path/to/your/personas"

# Check current configuration
python -m src.cli.main status
```

**Priority Order:**
1. **Environment Variable** (`REVIEW_CREW_PERSONAS_DIR`)
2. **Default** (`examples/personas/` - for testing)

### Persona Files

Persona configuration files use YAML format and are organized by type:

**Reviewer Persona Example** (`config/personas/reviewers/technical.yaml`):
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

**Contextualizer Persona Example** (`config/personas/contextualizers/business_contextualizer.yaml`):
```yaml
name: "Business Context Formatter"
role: "Business Requirements Analyst"
goal: "Format and structure business context for comprehensive reviews"
backstory: "Expert in translating business requirements into actionable context..."
prompt_template: "Process and format the following context information..."
model_config:
  temperature: 0.2
  max_tokens: 1000
```

**Analyzer Persona Example** (`config/personas/analyzers/meta_analysis.yaml`):
```yaml
name: "Meta-Analysis & Synthesis Specialist"
role: "Editorial Consultant & Application Strategy Advisor"
goal: "Synthesize feedback from all reviewers and provide strategic recommendations"
backstory: "Expert in content analysis and strategic communication..."
prompt_template: "Analyze the following reviews and provide synthesis..."
model_config:
  temperature: 0.4
  max_tokens: 2000
  max_context_length: 8192  # Optional: persona-specific context limit
```

## Advanced Features

### Analysis & Synthesis
The system includes intelligent analysis agents that process all reviewer feedback:

- **Conflict Resolution**: Identifies and resolves contradictory recommendations
- **Priority Ranking**: Ranks feedback by importance and impact
- **Theme Identification**: Groups related feedback into coherent themes
- **Actionable Insights**: Provides clear, prioritized next steps
- **Context Generation**: Creates summaries and context for follow-up work

### Smart Context Management
For models with smaller context windows, the system automatically handles large review sets:

- **Automatic Chunking**: Splits large review sets into manageable chunks
- **Intelligent Synthesis**: Combines chunked analyses into coherent final output
- **Configurable Limits**: Set context length via `--max-context-length` (default: 4096)
- **Seamless Experience**: Chunking happens transparently when needed

### CLI Options Reference
```bash
# Core options
--provider PROVIDER          # bedrock, lm_studio, ollama (default: bedrock)
--async-mode                 # Run reviews in parallel (faster)
--output FILE                # Save results to file
--no-content                 # Hide original content in output

# Agent selection
--agents AGENT               # Use specific agents (can repeat)

# Context and analysis control
--context PATH               # Path to context file processed by contextualizers
--include-context            # Include contextualizer results in output
--no-analysis                # Disable analysis stage (reviewers only)
--max-context-length INT     # Context limit for chunking (default: 4096)

# Model configuration
--model-url URL              # Custom model URL
--model-id ID                # Custom model ID
```

### Output Format
The system generates structured output with clear sections:

1. **Original Content** (optional, use `--no-content` to hide)
2. **Context Information** (optional, use `--include-context` to show contextualizer results)
3. **Individual Reviews** - Detailed feedback from each reviewer
4. **Analysis & Synthesis** - Integrated insights and recommendations

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
# Install core dependencies
uv sync

# For LM Studio support
uv sync --extra lm-studio

# For Ollama support  
uv sync --extra ollama

# For development (includes pytest, black, mypy)
uv sync --group dev

# Reinstall everything
make clean && make install && make setup
```

**Persona Configuration Errors:**
```bash
# Validate configurations
make test-config

# Reset to defaults
make clean-config && make setup

# Check specific persona types
python -c "
from src.config.persona_loader import PersonaLoader
loader = PersonaLoader()
reviewers = loader.load_reviewer_personas()
analyzers = loader.load_analyzer_personas()
print(f'Reviewers: {len(reviewers)}, Analyzers: {len(analyzers)}')
"

# Use only standard test personas (temporarily)
export REVIEW_CREW_PERSONAS_DIR="examples/personas"
python -m src.cli.main status  # Should show reviewers and analyzers

# Or edit .env file to change default
echo "REVIEW_CREW_PERSONAS_DIR=examples/personas" > .env
```

**Virtual Environment Issues:**
```bash
# Create fresh environment with uv
rm -rf .venv
uv venv
source .venv/bin/activate  # or `. .venv/bin/activate`
make install
```
