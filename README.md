# Review-Crew

## Purpose
Review-Crew is a powerful, generic multi-agent review platform that uses AI agents with different personas to evaluate any type of content. The system provides comprehensive feedback from multiple perspectives, then synthesizes all reviews through intelligent analysis agents. Perfect for content review, code analysis, document evaluation, and more.

### Key Features
- **Multi-Agent Reviews**: Configure any number of reviewer agents with custom personas
- **Parallel Graph Execution**: Strands Graph architecture enables true parallel agent processing for maximum performance
- **Multi-Document Support**: Review entire document collections with manifest-driven configuration
- **Intelligent Analysis**: Synthesis agents integrate feedback, resolve conflicts, and prioritize recommendations  
- **Dynamic Writing Analysis Tools**: Context-aware word counting, readability analysis, and constraint validation
- **Clean Markdown Output**: Generates readable, shareable markdown reports with structured sections
- **Generic & Extensible**: Works with any content type - not limited to specific domains
- **Smart Context Management**: Automatic chunking for large reviews with smaller models
- **Multiple LLM Providers**: AWS Bedrock, LM Studio, Ollama support
- **Hybrid Architecture**: Graph-based execution with legacy fallback for maximum reliability
- **Type Safety**: Comprehensive type checking with MyPy for robust, maintainable code

## Technology
* **Language**: Python 3.10+
* **Package Management**: uv
* **Agent Framework**: [Strands Agents](https://strandsagents.com/latest/documentation/docs/) - AWS-backed, production-ready multi-agent framework
* **Code Quality**: [Ruff](https://docs.astral.sh/ruff/) - Ultra-fast Python linter and formatter (replaces Black + Flake8)
* **Type Checking**: MyPy
* **Interface**: CLI
* **LLM**: LM Studio + Available Models (configurable model providers)

## Architecture

### Strands Graph Multi-Agent System
Review-Crew uses a modern **Strands Graph architecture** for optimal performance and reliability:

#### **Graph-Based Parallel Execution**
- **Parallel Processing**: All reviewer agents run simultaneously using Strands Graph DAG execution
- **Intelligent Orchestration**: DocumentProcessorNode handles document loading and validation
- **Result Conversion**: Seamless integration between Strands results and Review-Crew format
- **Error Handling**: Robust error detection prevents agent hallucination on invalid inputs

#### **Three-Stage Review Process**
1. **Document Processing**: DocumentProcessorNode loads, validates, and prepares content
2. **Parallel Review Stage**: Multiple reviewer agents provide simultaneous feedback from specialized perspectives  
3. **Analysis Stage**: Analysis agents synthesize all feedback, resolve conflicts, and provide actionable insights

#### **Hybrid Implementation**
- **Primary**: Strands Graph execution for maximum performance and parallel processing
- **Fallback**: Legacy orchestration methods for backward compatibility
- **Seamless**: Automatic selection based on content type and system state

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
make install       # Install with uv (recommended)
make setup         # Copy example personas to config/
make test          # Validate configuration

# Code quality checks (optional but recommended)
make quality       # Run linting, formatting, and type checking

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

# Code Quality (Ruff-powered)
make format        # Format code with Ruff (replaces Black)
make lint          # Run linting with Ruff (replaces Flake8)
make lint fix      # Auto-fix linting issues with Ruff
make check         # Run type checking with MyPy
make check fix     # Run type checking (MyPy doesn't auto-fix, same as check)
make quality       # Run all code quality checks (lint + format + type check)
make quality fix   # Run all quality checks with auto-fixes where possible

# Project Management
make clean         # Clean up temporary files
make install       # Install dependencies with uv
```

#### Code Quality with Ruff
Review-Crew uses [Ruff](https://docs.astral.sh/ruff/) for blazing-fast code quality management:

**Benefits:**
- ⚡ **~1000x faster** than traditional Black + Flake8 combination
- 🔧 **Auto-fixes** 800+ types of issues automatically
- 📦 **Single tool** replaces multiple dependencies (Black, Flake8, isort)
- 🎯 **Modern standards** - enforces latest Python typing conventions
- ⚙️ **Zero config** - works out of the box with sensible defaults

**Quick Usage:**
```bash
make quality fix   # Run all checks with auto-fixes (recommended for development)
make quality       # Run all checks without fixes (recommended for CI/CD)
make lint fix      # Auto-fix linting issues only
make format        # Format code to consistent style
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

# All reviews now run asynchronously with parallel agents for maximum performance

# Review a file
make review ARGS='"$(cat path/to/your/file.txt)"'

# Review with output file
make review-lm ARGS='path/to/file.txt --output results.txt'

# Multi-document reviews
make review-lm ARGS='project-docs/'
make review-lm ARGS='document-collection/ --output comprehensive_review.md'

# Pipe content with options
cat essay.txt | make review-lm ARGS='--output results.txt'
```

**Important:** Makefile commands use `ARGS='...'` syntax, not direct command-line arguments.

#### Advanced CLI Usage
```bash
# Review a file directly
python -m src.cli.main review path/to/your/file.txt

# Multi-document review
python -m src.cli.main review project-docs/
python -m src.cli.main review document-collection/ --provider lm_studio

# Pipe content from stdin
cat essay.txt | python -m src.cli.main review
echo "Review this text" | make review-lm

# Use specific agents
python -m src.cli.main review "content" -a "Technical Reviewer" -a "UX Reviewer"

# Use different providers
python -m src.cli.main review "content" --provider lm_studio
python -m src.cli.main review "content" --provider ollama
python -m src.cli.main review "content" --provider bedrock  # default

# All reviews now run asynchronously with parallel processing for optimal performance

# Save results to file
python -m src.cli.main review "content" -o results.txt

# Disable analysis (reviewers only)
python -m src.cli.main review "content" --no-analysis

# Context length is now configured per-model in persona files

# Custom model configuration
python -m src.cli.main review "content" --provider lm_studio --model-url http://localhost:1234/v1
python -m src.cli.main review "content" --provider ollama --model-id llama2

# Context processing with contextualizers
python -m src.cli.main review "content" --context context_file.txt
python -m src.cli.main review "content" --context context_file.txt --include-context
```

## Multi-Document Reviews

Review-Crew supports reviewing entire document collections with advanced manifest-driven configuration. This is perfect for reviewing applications, project documentation, or any related set of documents that should be evaluated together.

### Basic Multi-Document Usage

```bash
# Review all documents in a directory
python -m src.cli.main review project-docs/

# With LM Studio (recommended)
make review-lm ARGS='project-docs/'

# Multi-document review with async parallel processing
make review-lm ARGS='project-docs/'
```

### Manifest Configuration

Place a `manifest.yaml` file in your document directory to control the review process:

**Basic Manifest Example:**
```yaml
review_configuration:
  name: "Document Review"
  description: "Comprehensive document review"
  
  # Agent Selection
  reviewers:
    - "Content Reviewer"
    - "Technical Reviewer"
  
  analyzers:
    - "Quality Metrics Analyzer"
```

**Advanced Manifest Example:**
```yaml
review_configuration:
  name: "Multi-Document Project Review"
  description: "Comprehensive review with document relationships"
  version: "2.0"
  
  # Agent Configuration
  contextualizers:
    - "Business Context Formatter"
  
  reviewers:
    - "Content Reviewer"
    - "Technical Reviewer"
  
  analyzers:
    - "Quality Metrics Analyzer"
  
  # Document Structure
  documents:
    primary: "main_document.md"
    supporting:
      - "technical_spec.md"
      - "user_guide.md"
    
    # Context files provide background information
    context_files:
      - path: "project_requirements.md"
        type: "requirements"
        weight: "high"
      - path: "context/business_context.md"
        type: "business_context"
        weight: "medium"
    
    # Document relationships guide review focus
    relationships:
      - source: "technical_spec.md"
        target: "main_document.md"
        type: "complements"
        note: "Technical details should align with main document"
      
      - source: "project_requirements.md"
        target: "main_document.md"
        type: "evidence_support"
        note: "Requirements should validate document approach"
  
  # Review Focus Configuration
  review_focus:
    primary_concerns:
      - concern: "Cross-document consistency"
        weight: "high"
        description: "Ensure technical and content consistency across documents"
      
      - concern: "Requirements alignment"
        weight: "medium"
        description: "Each document should align with project requirements"
  
  # Processing Configuration
  processing:
    max_content_length: 8000
```

### Manifest Features

**Agent Selection:**
- `reviewers`: Specific reviewer personas to use
- `reviewer_categories`: Load reviewers by category (academic, technical, content, business)
- `contextualizers`: Agents to process context before reviews
- `analyzers`: Analysis agents for synthesis

**Document Configuration:**
- `primary`: Main document for review
- `supporting`: Additional documents that support the primary
- `context_files`: Background information with type and weight
- `relationships`: Define how documents relate to each other

**Review Focus:**
- `primary_concerns`: High-priority review areas
- `secondary_concerns`: Additional focus points
- Each concern has weight (critical/high/medium/low) and description

**Processing Rules:**
- `max_content_length`: Token limit for analysis
- Custom processing instructions

### Document Relationships

Define how documents relate to guide reviewer focus:

**Relationship Types:**
- `complements`: Documents that should build on each other
- `evidence_support`: Background that validates claims
- `relates_to`: General relationship
- `contradicts`: Documents that might conflict (flagged for review)

**Context File Types:**
- `requirements`: Project requirements and specifications
- `business_context`: Business background and stakeholder information
- `analysis_supplement`: Additional analysis context
- `general`: General context information

### Multi-Document Examples

```bash
# Review project documentation with manifest
make review-lm ARGS='project-docs/'

# Review technical documentation
python -m src.cli.main review technical-docs/ --provider lm_studio

# Review with specific output
python -m src.cli.main review document-collection/ -o comprehensive_review.md

# Review with structured markdown output
python -m src.cli.main review docs/ -o comprehensive_review.md
```

### Directory Structure Example

```
project-docs/
├── manifest.yaml              # Review configuration
├── main_document.md          # Primary document
├── technical_spec.md         # Supporting document
├── user_guide.md             # Supporting document
├── project_requirements.md   # Context file
└── context/
    └── business_context.md   # Additional context
```

**Without Manifest:**
- Reviews all readable files in directory
- Uses all available reviewers
- Basic document compilation

**With Manifest:**
- Uses specified reviewers and configuration
- Processes documents according to relationships
- Applies custom review focus and processing rules

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
  model_id: "qwen/qwen3-4b-2507"  # Use standard model for technical reviews
  temperature: 0.3
  max_tokens: 1500
  max_context_length: 8192  # 8K context for standard model

# Optional: Enable writing analysis tools
tools_config:
  enabled: true
  analysis_types:
    - "metrics"        # Word/character counts, readability scores
    - "structure"      # Document structure and paragraph flow analysis
    - "readability"    # Reading level assessment
```

**Contextualizer Persona Example** (`config/personas/contextualizers/business_contextualizer.yaml`):
```yaml
name: "Business Context Formatter"
role: "Business Requirements Analyst"
goal: "Format and structure business context for comprehensive reviews"
backstory: "Expert in translating business requirements into actionable context..."
prompt_template: "Process and format the following context information..."
model_config:
  model_id: "qwen/qwen3-4b-2507"  # Use standard model for context processing
  temperature: 0.2
  max_tokens: 1000
  max_context_length: 8192  # 8K context for standard model
```

**Analyzer Persona Example** (`config/personas/analyzers/meta_analysis.yaml`):
```yaml
name: "Meta-Analysis & Synthesis Specialist"
role: "Editorial Consultant & Application Strategy Advisor"
goal: "Synthesize feedback from all reviewers and provide strategic recommendations"
backstory: "Expert in content analysis and strategic communication..."
prompt_template: "Analyze the following reviews and provide synthesis..."
model_config:
  model_id: "qwen/qwen3-4b-thinking-2507"  # Use reasoning model for complex analysis
  temperature: 0.4
  max_tokens: 2000
  max_context_length: 32768  # 32K context for reasoning model
```

## Model Configuration

### Reasoning vs Standard Models

Review-Crew supports both reasoning and standard models for optimal performance:

**Reasoning Models** (for complex analysis):
- **Model**: `qwen/qwen3-4b-thinking-2507`
- **Context**: 32K tokens
- **Best for**: Analysis agents, complex synthesis, strategic evaluation
- **Features**: Chain-of-thought reasoning, multi-step analysis

**Standard Models** (for straightforward tasks):
- **Model**: `qwen/qwen3-4b-2507`  
- **Context**: 8K tokens
- **Best for**: Review agents, contextualizers, basic evaluation
- **Features**: Fast processing, efficient resource usage

### Model Selection in Personas

Configure models in your persona files:

```yaml
# For complex analysis (reasoning model)
model_config:
  model_id: "qwen/qwen3-4b-thinking-2507"
  temperature: 0.3
  max_tokens: 2000
  max_context_length: 32768

# For basic reviews (standard model)
model_config:
  model_id: "qwen/qwen3-4b-2507"
  temperature: 0.5
  max_tokens: 1500
  max_context_length: 8192

# Use LM Studio default model (fallback)
model_config:
  temperature: 0.4
  max_tokens: 1000
  # No model_id - uses LM Studio's currently loaded model
```

### Context Length Management

- **Automatic Detection**: System detects context limits based on model_id
- **Smart Truncation**: Content automatically truncated when exceeding limits
- **Per-Model Optimization**: Each model uses its optimal context window
- **Chunking Support**: Large content automatically chunked for analysis agents

## Writing Analysis Tools

### Dynamic Context-Aware Analysis

Review-Crew includes sophisticated writing analysis tools that provide precise, deterministic measurements that LLMs cannot reliably perform:

**Key Capabilities:**
- **Exact Word/Character Counts**: Precise measurements vs. LLM approximations (±5-15% error)
- **Dynamic Constraint Extraction**: Automatically detects limits from content (e.g., "Word Limit: 650 words")
- **Context-Aware Analysis**: Adapts analysis types based on content type (Common App vs. Technical docs)
- **Readability Assessment**: Quantified Flesch-Kincaid scores vs. subjective assessments
- **Structural Analysis**: Document coherence and paragraph flow metrics
- **Academic Writing Tools**: Cliché detection, personal voice analysis, essay strength scoring

### Tool Integration

**Simple Configuration:**
```yaml
# Add to any persona YAML file
tools_config:
  enabled: true  # Enables automatic constraint extraction and analysis
  analysis_types:  # Optional: specify which tools to use
    - "metrics"        # Word/character counts, readability scores
    - "constraints"    # Validation against word limits and requirements
    - "structure"      # Document structure and paragraph flow analysis
    - "strength"       # Essay strength and admissions impact assessment
    - "cliches"        # Detection of overused phrases and generic language
    - "voice"          # Personal voice and authenticity analysis
```

**Automatic Context Detection:**
The system automatically extracts constraints from your existing content format:

```markdown
**ASSIGNMENT CONTEXT:**
- Essay Type: Common Application Personal Statement
- Word Limit: 650 words
- Constraint: Must work for ALL schools

**ESSAY TO REVIEW:**
[essay content]
```

**Automatically extracts:** 650 word limit, Common App type, generic requirement

### Enhanced Review Quality

**Before (LLM only):**
> "This content appears to be well-structured with good readability."

**After (Tools + LLM):**
> "This content scores 0.82/1.0 for structural coherence with a Grade 9.1 reading level. At 647 words (99.5% of 650 limit), it's optimally sized. The vocabulary diversity of 0.65 indicates strong word choice variety."

### Available Analysis Types

| Type | Purpose | Best For |
|------|---------|----------|
| `metrics` | Basic text measurements | All content types |
| `readability` | Reading level assessment | Audience-targeted content |
| `vocabulary` | Word complexity analysis | Technical/academic writing |
| `structure` | Document organization | Long-form content |
| `constraints` | Limit validation | Strict word/character limits |
| `cliches` | Overused phrase detection | Creative/marketing writing |
| `strength` | Impact assessment | Persuasive/sales content |
| `voice` | Authenticity measurement | Personal/narrative writing |

## Advanced Features

### Strands Graph Architecture Benefits

**Performance Improvements:**
- **True Parallel Execution**: All reviewer agents run simultaneously instead of sequentially
- **Intelligent Resource Management**: Strands Graph optimizes agent scheduling and resource usage
- **Faster Processing**: Significant speed improvements for multi-agent reviews
- **Scalable**: Easily add more agents without performance degradation

**Reliability Enhancements:**
- **Error Isolation**: Individual agent failures don't crash the entire review process
- **Robust Error Handling**: Automatic detection and graceful handling of invalid inputs
- **Hallucination Prevention**: Agents no longer generate fake responses for missing content
- **Status Tracking**: Comprehensive execution status and error reporting

**Technical Features:**
- **Hybrid Architecture**: Graph execution with legacy fallback for maximum compatibility
- **Clean JSON Parsing**: Automatic extraction of clean text from LLM JSON responses
- **Result Conversion**: Seamless integration between Strands and Review-Crew data formats
- **Comprehensive Testing**: 157 passing tests ensure reliability and correctness

### Analysis & Synthesis
The system includes intelligent analysis agents that process all reviewer feedback:

- **Conflict Resolution**: Identifies and resolves contradictory recommendations
- **Priority Ranking**: Ranks feedback by importance and impact
- **Theme Identification**: Groups related feedback into coherent themes
- **Actionable Insights**: Provides clear, prioritized next steps
- **Context Generation**: Creates summaries and context for follow-up work

### Smart Context Management
The system automatically handles context length management based on each model's capabilities:

- **Model-Specific Context Lengths**: Each model uses its optimal context window (32K for reasoning, 8K for standard models)
- **Automatic Chunking**: Splits large review sets into manageable chunks when needed
- **Intelligent Synthesis**: Combines chunked analyses into coherent final output
- **Per-Persona Configuration**: Set context length in persona `model_config` section
- **Seamless Experience**: Context management happens transparently based on model capabilities

### CLI Options Reference
```bash
# Core options
--provider PROVIDER          # bedrock, lm_studio, ollama (default: bedrock)
# All reviews now run asynchronously with parallel processing by default
--output FILE                # Save results to file
--no-content                 # Hide original content in output

# Agent selection
--agents AGENT               # Use specific agents (can repeat)
                            # Note: Overridden by manifest.yaml when present

# Context and analysis control
--context PATH               # Path to context file processed by contextualizers
--include-context            # Include contextualizer results in output
--no-analysis                # Disable analysis stage (reviewers only)

# Output control
--no-content                 # Hide original content in markdown output

# Model configuration
--model-url URL              # Custom model URL
--model-id ID                # Custom model ID
```

### Output Format
The system generates clean, readable markdown output with structured sections:

```markdown
## Content
[Original document content - optional, use --no-content to hide]

## Reviews

### Agent Name
*Agent Role*

[Detailed feedback from each reviewer]

## Context
[Context processing results - optional, use --include-context to show]

## Analysis

### Meta-Analysis Summary
[Synthesized analysis from all reviews]

### Key Themes
• [Identified themes across reviews]

### Priority Recommendations
• [Actionable next steps prioritized by importance]

## Analysis Errors
[Any analysis failures or issues - if applicable]
```

**Multi-Document Output:**
```markdown
## Primary Document

• **File:** personal_statement.md
• **Source:** input/usc/manifest.yaml

[Document content...]

## Supporting Document

• **File:** supplemental_essays.md
• **Source:** input/usc/manifest.yaml

[Document content...]

## Reviews

### USC Admissions Officer
*University of Southern California Admissions Officer*

[Review content...]
```

**Document Structure:**
- **Document Collection Summary** - Overview of all documents reviewed
- **Cross-Document Analysis** - Relationships and consistency analysis
- **Document-Specific Reviews** - Individual feedback per document
- **Synthesis & Recommendations** - Integrated multi-document insights

All output is formatted as clean, readable markdown that can be easily reviewed, shared, or processed further.

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

# For development (includes pytest, ruff, mypy)
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
