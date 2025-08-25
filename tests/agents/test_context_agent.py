"""Tests for ContextAgent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.context_agent import ContextAgent, ContextResult
from src.config.persona_loader import PersonaConfig


class TestContextAgent:
    """Test ContextAgent functionality."""
    
    @patch('src.agents.context_agent.ReviewAgent')
    def test_init_with_persona(self, mock_review_agent, mock_contextualizer_persona):
        """Test ContextAgent initialization with persona."""
        mock_agent_instance = Mock()
        mock_review_agent.return_value = mock_agent_instance
        
        agent = ContextAgent(persona=mock_contextualizer_persona)
        
        assert agent.persona == mock_contextualizer_persona
        assert agent.agent == mock_agent_instance
        # Check that ReviewAgent was called with the persona
        mock_review_agent.assert_called_once()
        call_args = mock_review_agent.call_args
        assert call_args[0][0] == mock_contextualizer_persona  # First positional arg is persona
        assert call_args[1]['model_provider'] == "bedrock"
    
    def test_process_context_with_agent(self, mock_contextualizer_persona):
        """Test processing context with available agent."""
        mock_agent = Mock()
        mock_agent.review.return_value = "Processed context"
        
        with patch('src.agents.context_agent.ReviewAgent', return_value=mock_agent):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            result = agent.process_context("test context")
            
            assert isinstance(result, ContextResult)
            assert result.formatted_context == "Processed context"
            mock_agent.review.assert_called_once_with("test context")
    
    @pytest.mark.asyncio
    async def test_process_context_async_with_agent(self, mock_contextualizer_persona):
        """Test async processing context with available agent."""
        mock_agent = Mock()
        mock_agent.review_async = AsyncMock(return_value="Processed context")
        
        with patch('src.agents.context_agent.ReviewAgent', return_value=mock_agent):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            result = await agent.process_context_async("test context")
            
            assert isinstance(result, ContextResult)
            assert result.formatted_context == "Processed context"
            mock_agent.review_async.assert_called_once_with("test context")
    
    def test_parse_context_response_with_summary(self, mock_contextualizer_persona):
        """Test parsing context response with summary section."""
        with patch('src.agents.context_agent.ReviewAgent'):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            response = """
            ## CONTEXT SUMMARY
            This is a test summary
            
            ## OTHER SECTION
            Other content
            """
            
            result = agent._parse_context_response(response)
            
            assert isinstance(result, ContextResult)
            assert "This is a test summary" in result.context_summary
    
    def test_parse_context_response_without_summary(self, mock_contextualizer_persona):
        """Test parsing context response without summary section."""
        with patch('src.agents.context_agent.ReviewAgent'):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            response = "Just some context without sections"
            
            result = agent._parse_context_response(response)
            
            assert isinstance(result, ContextResult)
            assert result.context_summary == "Context processed by contextualizer persona"
    
    def test_format_context_for_review(self, mock_contextualizer_persona):
        """Test formatting context for review."""
        with patch('src.agents.context_agent.ReviewAgent'):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            context_result = ContextResult(
                formatted_context="Test context",
                context_summary="Test summary"
            )
            
            formatted = agent.format_context_for_review(context_result)
            
            assert "## CONTEXT" in formatted
            assert "Test context" in formatted
    
    def test_get_info_with_persona(self, mock_contextualizer_persona):
        """Test getting info with available persona."""
        with patch('src.agents.context_agent.ReviewAgent'):
            agent = ContextAgent(persona=mock_contextualizer_persona)
            
            info = agent.get_info()
            
            assert info['name'] == mock_contextualizer_persona.name
            assert info['role'] == mock_contextualizer_persona.role


class TestContextResult:
    """Test ContextResult dataclass."""
    
    def test_context_result_creation(self):
        """Test creating a ContextResult."""
        result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        assert result.formatted_context == "Test context"
        assert result.context_summary == "Test summary"
        assert result.timestamp is not None
    
    def test_context_result_with_timestamp(self):
        """Test creating a ContextResult with explicit timestamp."""
        from datetime import datetime
        timestamp = datetime.now()
        
        result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary",
            timestamp=timestamp
        )
        
        assert result.timestamp == timestamp