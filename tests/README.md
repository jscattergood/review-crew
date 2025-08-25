# Review-Crew Test Suite

This directory contains the pytest-based test suite for Review-Crew, designed to provide fast testing with mock LLM responses.

## Directory Structure

```
tests/
├── __init__.py                    # Test package
├── conftest.py                    # Pytest fixtures and configuration
├── README.md                      # This file
├── test_integration.py            # Integration tests
├── agents/                        # Tests for src/agents/
│   ├── test_context_agent.py      # ContextAgent tests
│   ├── test_conversation_manager.py # ConversationManager tests
│   └── test_review_agent.py       # ReviewAgent tests
├── cli/                          # Tests for src/cli/ (future)
└── config/                       # Tests for src/config/
    └── test_persona_loader.py     # PersonaLoader tests
```

## Key Features

### 🚀 **Mock LLM Support**
- All tests use mocked LLM responses
- No API calls or costs
- Fast execution (seconds instead of minutes)
- Predictable, repeatable results

### 🧪 **Comprehensive Coverage**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Configuration Tests**: Persona loading and validation
- **System Health Tests**: Import and setup verification

### ⚡ **Fast Execution**
- Mock responses eliminate network latency
- Parallel test execution where possible
- Focused test categories for quick feedback

## Running Tests

### Quick Start
```bash
# Install dev dependencies
make install

# Run all tests
make test

```

### Direct pytest Commands
```bash
# Run all tests (with uv)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/agents/test_context_agent.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing

# Run tests matching pattern
uv run pytest tests/ -k "context" -v
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual classes and functions
- Mock all external dependencies
- Fast execution (< 1 second per test)
- High isolation and predictability

### Integration Tests (`@pytest.mark.integration`)
- Test complete workflows
- Mock only LLM calls, use real components
- Test realistic scenarios
- Verify component interactions

### Slow Tests (`@pytest.mark.slow`)
- More comprehensive integration tests
- File-based testing
- Complex scenario testing
- Can be excluded with `-m "not slow"`

## Mock LLM Responses

The test suite uses sophisticated mocking to simulate LLM behavior:

```python
# Example mock setup
with patch('strands.Agent') as mock_agent_class:
    mock_agent = Mock()
    mock_agent.return_value = Mock(message="Mock review response")
    mock_agent_class.return_value = mock_agent
    
    # Now test your code - no real LLM calls!
    result = agent.review("test content")
```

### Mock Response Patterns
- **Security Reviews**: Focus on vulnerabilities and compliance
- **Technical Reviews**: Code quality and best practices
- **Content Reviews**: Documentation and clarity
- **Context Processing**: Formatted context output

## Fixtures Available

### Core Fixtures
- `mock_persona`: Test persona configuration
- `mock_contextualizer_persona`: Test contextualizer persona
- `sample_content`: Sample content for testing
- `sample_context`: Sample context data
- `mock_llm_response`: Mock LLM response object

### Mock Objects
- `mock_strands_agent`: Mocked Strands Agent
- `mock_persona_loader`: Mocked PersonaLoader
- `empty_persona_loader`: PersonaLoader with no personas

## Writing New Tests

### Unit Test Example
```python
def test_my_function(mock_persona):
    """Test my function with mocked dependencies."""
    with patch('src.module.external_dependency') as mock_dep:
        mock_dep.return_value = "expected_result"
        
        result = my_function(mock_persona)
        
        assert result == "expected_result"
        mock_dep.assert_called_once()
```

### Integration Test Example
```python
@pytest.mark.integration
def test_complete_workflow(sample_content):
    """Test complete workflow with mocked LLM."""
    with patch('strands.Agent') as mock_agent:
        mock_agent.return_value.return_value = Mock(message="Mock response")
        
        manager = ConversationManager()
        result = manager.run_review(sample_content)
        
        assert len(result.reviews) > 0
```

## Benefits Over Manual Testing

1. **Speed**: Tests run in seconds vs. minutes
2. **Cost**: No LLM API costs during development
3. **Reliability**: Consistent, predictable results
4. **Coverage**: Test edge cases and error conditions
5. **Automation**: Can run in CI/CD pipelines
6. **Debugging**: Easier to isolate and fix issues

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**No Personas Found**
```bash
# This is expected in test environment
# Tests will skip gracefully if personas aren't configured
```

### Debug Mode
```bash
# Run with maximum verbosity
pytest tests/ -vvv --tb=long

# Run single test with debugging
pytest tests/agents/test_context_agent.py::TestContextAgent::test_process_context_with_agent -vvv
```
