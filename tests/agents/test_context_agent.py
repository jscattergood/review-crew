"""Tests for ContextAgent class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.context_agent import ContextAgent, ContextResult
from src.config.persona_loader import PersonaConfig


class TestContextAgent:
    """Test cases for ContextAgent."""

    @pytest.fixture
    def mock_persona(self):
        """Create a mock persona config."""
        persona = PersonaConfig(
            name="Test Contextualizer",
            role="Context Processor", 
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test prompt: {content}",
            model_config={}
        )
        return persona

    @pytest.fixture
    def agent(self, mock_persona):
        """Create a ContextAgent with mocked dependencies."""
        agent = ContextAgent(persona=mock_persona)
        
        # Mock the agent property to return a mock agent without creating real models
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value="Test context result")
        mock_agent.invoke_async = AsyncMock(return_value="Test async context result")
        
        # Replace the lazy-loaded agent with our mock
        agent._agent = mock_agent
        
        return agent

    def test_init(self, agent, mock_persona):
        """Test ContextAgent initialization."""
        assert agent.persona == mock_persona
        assert agent.persona.name == "Test Contextualizer"

    @pytest.mark.asyncio
    async def test_process_context_async(self, agent):
        """Test asynchronous context processing."""
        content = "Test content to process"
        result = await agent.process_context(content)
        
        assert isinstance(result, ContextResult)
        assert result.formatted_context == "Test async context result"
        # Verify the async agent was called
        agent.agent.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_context_async_with_error_content(self, agent):
        """Test async context processing with error content."""
        error_content = "ERROR_NO_CONTENT"
        result = await agent.process_context(error_content)
        
        assert isinstance(result, ContextResult)
        assert "No content was provided" in result.formatted_context

    @pytest.mark.asyncio
    async def test_invoke_async_graph(self, agent):
        """Test graph-compatible async invoke method."""
        # Mock the process_context_async method
        with patch.object(agent, 'process_context', new_callable=AsyncMock) as mock_process:
            mock_context_result = ContextResult(
                formatted_context="Graph context result",
                context_summary="Summary",
                timestamp=datetime.now()
            )
            mock_process.return_value = mock_context_result
            
            result = await agent.invoke_async_graph("test content")
            
            # Should return MultiAgentResult
            assert hasattr(result, 'results')
            assert hasattr(result, 'execution_time')
            assert hasattr(result, 'execution_count')
            
            # Verify process_context was called
            mock_process.assert_called_once_with("test content")

    def test_parse_context_response(self, agent):
        """Test context response parsing."""
        response = "Formatted context response"
        result = agent._parse_context_response(response)
        
        assert isinstance(result, ContextResult)
        assert result.formatted_context == response
        # The actual behavior creates a default summary
        assert isinstance(result.context_summary, str)
