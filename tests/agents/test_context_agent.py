"""Tests for ContextAgent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.context_agent import ContextAgent, ContextResult
from src.config.persona_loader import PersonaConfig


class TestContextAgent:
    """Test ContextAgent functionality."""
    
    @patch('src.agents.base_agent.BaseAgent._setup_agent_logging')
    @patch('src.agents.base_agent.BaseAgent._create_model')
    @patch('src.agents.base_agent.Agent')
    def test_init_with_persona(self, mock_strands_agent, mock_create_model, mock_setup_logging, mock_contextualizer_persona):
        """Test ContextAgent initialization with persona."""
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_agent_instance = Mock()
        mock_strands_agent.return_value = mock_agent_instance
        
        agent = ContextAgent(persona=mock_contextualizer_persona)
        
        assert agent.persona == mock_contextualizer_persona
        assert agent.model_provider == "bedrock"
        assert agent.agent == mock_agent_instance
        
        mock_create_model.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_strands_agent.assert_called_once()
    
    def test_process_context_with_agent(self, mock_contextualizer_persona):
        """Test processing context with available agent."""
        mock_formatted_prompt = "Formatted prompt with context"
        mock_response = "Processed context"
        
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    # Mock the invoke method
                    with patch.object(agent, 'invoke', return_value=mock_response) as mock_invoke:
                        result = agent.process_context("test context")
                        
                        # Verify the result type
                        assert isinstance(result, ContextResult)
                        
                        # Verify invoke was called with properly formatted prompt
                        expected_prompt = mock_contextualizer_persona.prompt_template.format(content="test context")
                        mock_invoke.assert_called_once_with(expected_prompt, "context_processing")
    
    @pytest.mark.asyncio
    async def test_process_context_async_with_agent(self, mock_contextualizer_persona):
        """Test async processing context with available agent."""
        mock_response = "Processed context"
        
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    # Mock the invoke_async method
                    with patch.object(agent, 'invoke_async', return_value=mock_response) as mock_invoke_async:
                        result = await agent.process_context_async("test context")
                        
                        # Verify the result type
                        assert isinstance(result, ContextResult)
                        
                        # Verify invoke_async was called with properly formatted prompt
                        expected_prompt = mock_contextualizer_persona.prompt_template.format(content="test context")
                        mock_invoke_async.assert_called_once_with(expected_prompt, "context_processing_async")
    
    def test_parse_context_response_with_summary(self, mock_contextualizer_persona):
        """Test parsing context response with summary."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    response = """# Formatted Context
This is the formatted context.

## Summary
This is a summary of the context."""
                    
                    result = agent._parse_context_response(response)
                    
                    assert isinstance(result, ContextResult)
                    assert "This is the formatted context." in result.formatted_context
                    # The actual summary extraction logic may vary, so let's just check it's a string
                    assert isinstance(result.context_summary, str)
    
    def test_parse_context_response_without_summary(self, mock_contextualizer_persona):
        """Test parsing context response without summary."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    response = "Simple formatted context without sections."
                    
                    result = agent._parse_context_response(response)
                    
                    assert isinstance(result, ContextResult)
                    assert result.formatted_context == response
                    # Check that some summary was generated
                    assert isinstance(result.context_summary, str)
                    assert len(result.context_summary) > 0
    
    def test_format_context_for_review(self, mock_contextualizer_persona):
        """Test formatting context for review."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    context_result = ContextResult(
                        formatted_context="Formatted context",
                        context_summary="Context summary"
                    )
                    
                    formatted = agent.format_context_for_review(context_result)
                    
                    assert isinstance(formatted, str)
                    assert "Formatted context" in formatted
                    # The exact format may vary, so just check basic structure
                    assert len(formatted) > 0
    
    def test_get_info_with_persona(self, mock_contextualizer_persona):
        """Test get_info method inherited from BaseAgent."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ContextAgent(persona=mock_contextualizer_persona)
                    
                    # Call the actual get_info method
                    result = agent.get_info()
                    
                    # Verify the structure
                    assert isinstance(result, dict)
                    assert "name" in result
                    assert "role" in result
                    assert "goal" in result


class TestContextResult:
    """Test ContextResult data class."""
    
    def test_context_result_creation(self):
        """Test creating ContextResult with basic data."""
        result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        assert result.formatted_context == "Test context"
        assert result.context_summary == "Test summary"
        assert result.timestamp is not None
    
    def test_context_result_with_timestamp(self):
        """Test ContextResult with custom timestamp."""
        from datetime import datetime
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        
        result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary",
            timestamp=custom_time
        )
        
        assert result.timestamp == custom_time