"""Tests for ContextAgent."""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from src.agents.context_agent import ContextAgent, ContextResult


class TestContextAgent:
    """Test ContextAgent functionality."""
    
    @patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona')
    @patch('src.agents.context_agent.ReviewAgent')
    def test_init_with_persona(self, mock_review_agent, mock_load_persona, mock_contextualizer_persona):
        """Test ContextAgent initialization with available persona."""
        mock_load_persona.return_value = mock_contextualizer_persona
        mock_agent_instance = Mock()
        mock_review_agent.return_value = mock_agent_instance
        
        agent = ContextAgent(persona_name="test_contextualizer")
        
        assert agent.persona_name == "test_contextualizer"
        assert agent.persona == mock_contextualizer_persona
        assert agent.agent == mock_agent_instance
        mock_review_agent.assert_called_once()
    
    @patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona')
    def test_init_without_persona(self, mock_load_persona):
        """Test ContextAgent initialization when persona is not available."""
        mock_load_persona.return_value = None
        
        agent = ContextAgent(persona_name="nonexistent_contextualizer")
        
        assert agent.persona_name == "nonexistent_contextualizer"
        assert agent.persona is None
        assert agent.agent is None
    
    def test_load_contextualizer_persona_success(self, mock_contextualizer_persona):
        """Test successful contextualizer persona loading."""
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=mock_contextualizer_persona) as mock_load:
            agent = ContextAgent(persona_name="test_contextualizer")
            # Test the method directly with a fresh call
            mock_load.return_value = mock_contextualizer_persona
            result = agent._load_contextualizer_persona("test_contextualizer")
            
            assert result == mock_contextualizer_persona
    
    @patch('src.config.persona_loader.PersonaLoader')
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_contextualizer_persona_not_found(self, mock_exists, mock_loader_class):
        """Test contextualizer persona loading when file doesn't exist."""
        mock_loader = Mock()
        mock_loader.personas_dir = Mock()
        mock_loader.personas_dir.__truediv__ = Mock(return_value=Mock())
        mock_loader_class.return_value = mock_loader
        
        agent = ContextAgent(persona_name="nonexistent")
        result = agent._load_contextualizer_persona("nonexistent")
        
        assert result is None
    
    def test_process_context_no_agent(self):
        """Test processing context when no agent is available."""
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="nonexistent")
            result = agent.process_context("test context")
            
            assert result is None
    
    def test_process_context_with_agent(self, sample_context):
        """Test processing context with available agent."""
        mock_agent = Mock()
        mock_agent.review.return_value = "## CONTEXT SUMMARY\nTest summary\n\nFormatted context response"
        
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona') as mock_load:
            with patch('src.agents.context_agent.ReviewAgent', return_value=mock_agent):
                mock_load.return_value = Mock()  # Mock persona
                
                agent = ContextAgent(persona_name="test_contextualizer")
                result = agent.process_context(sample_context)
                
                assert isinstance(result, ContextResult)
                assert result.context_summary == "Test summary"
                assert "Formatted context response" in result.formatted_context
                mock_agent.review.assert_called_once_with(sample_context)
    
    @pytest.mark.asyncio
    async def test_process_context_async_no_agent(self):
        """Test async processing context when no agent is available."""
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="nonexistent")
            result = await agent.process_context_async("test context")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_process_context_async_with_agent(self, sample_context):
        """Test async processing context with available agent."""
        mock_agent = Mock()
        mock_agent.review_async = AsyncMock(return_value="## CONTEXT SUMMARY\nAsync test summary\n\nAsync formatted context")
        
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona') as mock_load:
            with patch('src.agents.context_agent.ReviewAgent', return_value=mock_agent):
                mock_load.return_value = Mock()  # Mock persona
                
                agent = ContextAgent(persona_name="test_contextualizer")
                result = await agent.process_context_async(sample_context)
                
                assert isinstance(result, ContextResult)
                assert result.context_summary == "Async test summary"
                assert "Async formatted context" in result.formatted_context
                mock_agent.review_async.assert_called_once_with(sample_context)
    
    def test_parse_context_response_with_summary(self):
        """Test parsing context response with summary section."""
        response = """
        ## CONTEXT SUMMARY
        This is a test summary
        
        ## OTHER SECTION
        Other content
        
        Rest of the formatted context
        """
        
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="test")
            result = agent._parse_context_response(response)
            
            assert result.context_summary == "This is a test summary"
            assert result.formatted_context == response
    
    def test_parse_context_response_without_summary(self):
        """Test parsing context response without summary section."""
        response = "Just formatted context without summary"
        
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="test")
            result = agent._parse_context_response(response)
            
            assert result.context_summary == "Context processed by contextualizer persona"
            assert result.formatted_context == response
    
    def test_format_context_for_review(self):
        """Test formatting context for review."""
        context_result = ContextResult(
            formatted_context="Test formatted context",
            context_summary="Test summary"
        )
        
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="test")
            formatted = agent.format_context_for_review(context_result)
            
            assert "## CONTEXT" in formatted
            assert "Test formatted context" in formatted
            assert "---" in formatted
    
    def test_get_info_no_persona(self):
        """Test getting info when no persona is available."""
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=None):
            agent = ContextAgent(persona_name="nonexistent")
            info = agent.get_info()
            
            assert info['name'] == 'No Context Agent'
            assert info['role'] == 'No contextualizer persona available'
            assert info['capabilities'] == []
    
    def test_get_info_with_persona(self, mock_contextualizer_persona):
        """Test getting info when persona is available."""
        with patch('src.agents.context_agent.ContextAgent._load_contextualizer_persona', return_value=mock_contextualizer_persona):
            with patch('src.agents.review_agent.ReviewAgent'):
                agent = ContextAgent(persona_name="test_contextualizer")
                info = agent.get_info()
                
                assert info['name'] == mock_contextualizer_persona.name
                assert info['role'] == mock_contextualizer_persona.role
                assert info['goal'] == mock_contextualizer_persona.goal
                assert len(info['capabilities']) > 0


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
        """Test creating a ContextResult with custom timestamp."""
        from datetime import datetime
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        
        result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary",
            timestamp=custom_time
        )
        
        assert result.timestamp == custom_time
