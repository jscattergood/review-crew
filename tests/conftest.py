"""Pytest configuration and fixtures for Review-Crew tests."""

import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.persona_loader import PersonaConfig


@pytest.fixture
def mock_persona():
    """Create a mock persona for testing."""
    return PersonaConfig(
        name="Test Reviewer",
        role="Test Specialist",
        goal="Test goal for unit testing",
        backstory="Test backstory for unit testing",
        prompt_template="Test prompt: {content}",
        model_config={
            'temperature': 0.3,
            'max_tokens': 1000
        }
    )


@pytest.fixture
def mock_contextualizer_persona():
    """Create a mock contextualizer persona for testing."""
    return PersonaConfig(
        name="Test Contextualizer",
        role="Test Context Processor",
        goal="Format test context information",
        backstory="Test contextualizer for unit testing",
        prompt_template="Format this context: {content}",
        model_config={
            'temperature': 0.2,
            'max_tokens': 1500
        }
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return Mock(message="This is a mock review response from the test LLM.")


@pytest.fixture
def mock_async_llm_response():
    """Mock async LLM response for testing."""
    mock = AsyncMock()
    mock.return_value = Mock(message="This is a mock async review response from the test LLM.")
    return mock


@pytest.fixture
def mock_strands_agent(mock_llm_response):
    """Mock Strands Agent for testing."""
    mock_agent = Mock()
    mock_agent.return_value = mock_llm_response
    mock_agent.invoke_async = AsyncMock(return_value=mock_llm_response)
    return mock_agent


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return """
    # Sample API Documentation
    
    This is a test API with some security issues:
    
    ## Authentication
    No authentication required!
    
    ## Endpoints
    
    ### POST /users
    Creates a new user with plaintext password storage.
    """


@pytest.fixture
def sample_context():
    """Sample context data for testing."""
    return """
    **Project Requirements:**
    - GDPR compliant system
    - Handle 1000+ users
    - Security audit required
    
    **Tech Stack:**
    - Node.js, PostgreSQL
    - Performance target: <200ms
    """


@pytest.fixture
def test_files_dir():
    """Path to test input files."""
    return Path(__file__).parent.parent / "test_inputs"


class MockPersonaLoader:
    """Mock PersonaLoader for testing."""
    
    def __init__(self, personas=None):
        self.personas = personas or []
        # Create mock methods that can be configured by tests
        self.load_reviewer_personas = Mock(return_value=self.personas)
        self.load_analyzer_personas = Mock(return_value=[])
        self.load_contextualizer_personas = Mock(return_value=[])
        self.load_persona = Mock(side_effect=self._load_persona_side_effect)
    
    def _load_persona_side_effect(self, filepath):
        if self.personas:
            return self.personas[0]
        raise FileNotFoundError(f"Mock persona not found: {filepath}")


@pytest.fixture
def mock_persona_loader(mock_persona):
    """Mock PersonaLoader with test personas."""
    return MockPersonaLoader([mock_persona])


@pytest.fixture
def empty_persona_loader():
    """Mock PersonaLoader with no personas."""
    return MockPersonaLoader([])
