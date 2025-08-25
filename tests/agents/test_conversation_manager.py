"""Tests for ConversationManager."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents.conversation_manager import ConversationManager, ConversationResult, ReviewResult
from src.agents.context_agent import ContextResult


class TestConversationManager:
    """Test ConversationManager functionality."""
    
    @patch('src.agents.conversation_manager.ConversationManager._load_analyzers')
    @patch('src.agents.conversation_manager.ConversationManager._load_contextualizers')
    @patch('src.agents.conversation_manager.ConversationManager._load_agents')
    def test_init_with_all_features(self, mock_load_agents, mock_load_contextualizers, mock_load_analyzers, mock_persona_loader):
        """Test ConversationManager initialization with all features enabled."""
        manager = ConversationManager(
            persona_loader=mock_persona_loader,
            model_provider="test",
            enable_analysis=True
        )
        
        assert manager.persona_loader == mock_persona_loader
        assert manager.model_provider == "test"
        assert manager.enable_analysis is True
        assert manager.context_agents == []
        assert manager.analysis_agents == []
        
        mock_load_agents.assert_called_once()
        mock_load_contextualizers.assert_called_once()
        mock_load_analyzers.assert_called_once()
    
    @patch('src.agents.conversation_manager.ConversationManager._load_contextualizers')
    @patch('src.agents.conversation_manager.ConversationManager._load_agents')
    def test_init_minimal(self, mock_load_agents, mock_load_contextualizers):
        """Test ConversationManager initialization with minimal features."""
        manager = ConversationManager(enable_analysis=False)
        
        assert manager.enable_analysis is False
        assert manager.analysis_agents == []
        assert manager.context_agents == []
        
        mock_load_agents.assert_called_once()
        mock_load_contextualizers.assert_called_once()
    
    @patch('src.agents.conversation_manager.ReviewAgent')
    def test_load_agents_success(self, mock_review_agent, mock_persona_loader, mock_persona):
        """Test successful agent loading."""
        mock_persona_loader.load_reviewer_personas.return_value = [mock_persona]
        mock_agent_instance = Mock()
        mock_review_agent.return_value = mock_agent_instance
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            assert len(manager.agents) == 1
            assert manager.agents[0] == mock_agent_instance
            mock_review_agent.assert_called_once_with(
                mock_persona,
                model_provider=manager.model_provider,
                model_config_override=manager.model_config
            )
    
    def test_load_agents_failure(self, mock_persona_loader):
        """Test agent loading failure."""
        mock_persona_loader.load_reviewer_personas.side_effect = Exception("Load failed")
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            assert len(manager.agents) == 0
    
    def test_get_available_agents(self, mock_persona_loader):
        """Test getting available agents."""
        mock_agent = Mock()
        mock_agent.get_info.return_value = {"name": "Test Agent", "role": "Tester"}
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.agents = [mock_agent]
            
            agents = manager.get_available_agents()
            
            assert len(agents) == 1
            assert agents[0] == {"name": "Test Agent", "role": "Tester"}
            mock_agent.get_info.assert_called_once()
    
    def test_run_review_no_agents(self, mock_persona_loader):
        """Test running review with no agents available."""
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.agents = []
            
            with pytest.raises(ValueError, match="No review agents available"):
                manager.run_review("test content")
    
    @patch('src.agents.conversation_manager.ConversationManager._prepare_content_for_review')
    @patch('src.agents.conversation_manager.ConversationManager._filter_agents')
    def test_run_review_success(self, mock_filter_agents, mock_prepare_content, mock_persona_loader, sample_content):
        """Test successful review run."""
        # Setup mock agent
        mock_agent = Mock()
        mock_agent.persona.name = "Test Agent"
        mock_agent.persona.role = "Tester"
        mock_agent.review.return_value = "Test review response"
        
        mock_filter_agents.return_value = [mock_agent]
        mock_prepare_content.return_value = sample_content
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader, enable_analysis=False)
            manager.agents = [mock_agent]
            
            result = manager.run_review(sample_content)
            
            assert isinstance(result, ConversationResult)
            assert result.content == sample_content
            assert len(result.reviews) == 1
            assert result.reviews[0].agent_name == "Test Agent"
            assert result.reviews[0].feedback == "Test review response"
            assert result.reviews[0].error is None
    
    def test_run_review_with_context(self, mock_persona_loader, sample_content, sample_context):
        """Test review run with context processing."""
        # Setup mock agent
        mock_agent = Mock()
        mock_agent.persona.name = "Test Agent"
        mock_agent.persona.role = "Tester"
        mock_agent.review.return_value = "Test review response"
        
        # Setup mock context agent
        mock_context_agent = Mock()
        mock_context_result = ContextResult(
            formatted_context="Formatted context",
            context_summary="Context summary"
        )
        mock_context_agent.process_context.return_value = mock_context_result
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            with patch('src.agents.conversation_manager.ConversationManager._filter_agents', return_value=[mock_agent]):
                manager = ConversationManager(
                    persona_loader=mock_persona_loader,
                    enable_analysis=False
                )
                manager.agents = [mock_agent]
                manager.context_agents = [mock_context_agent]
                
                result = manager.run_review(sample_content, context_data=sample_context)
                
                assert result.context_results == [mock_context_result]
                mock_context_agent.process_context.assert_called_once_with(sample_context)
    
    def test_run_review_agent_failure(self, mock_persona_loader, sample_content):
        """Test review run with agent failure."""
        # Setup mock agent that fails
        mock_agent = Mock()
        mock_agent.persona.name = "Failing Agent"
        mock_agent.persona.role = "Failer"
        mock_agent.review.side_effect = Exception("Review failed")
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            with patch('src.agents.conversation_manager.ConversationManager._filter_agents', return_value=[mock_agent]):
                manager = ConversationManager(persona_loader=mock_persona_loader, enable_analysis=False)
                manager.agents = [mock_agent]
                
                result = manager.run_review(sample_content)
                
                assert len(result.reviews) == 1
                assert result.reviews[0].agent_name == "Failing Agent"
                assert result.reviews[0].error == "Review failed"
                assert result.reviews[0].feedback == ""
    
    @pytest.mark.asyncio
    async def test_run_review_async_success(self, mock_persona_loader, sample_content):
        """Test successful async review run."""
        # Setup mock agent
        mock_agent = Mock()
        mock_agent.persona.name = "Test Agent"
        mock_agent.persona.role = "Tester"
        
        # Mock the async review method
        async def mock_review_async(content):
            return "Test async review response"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            with patch('src.agents.conversation_manager.ConversationManager._filter_agents', return_value=[mock_agent]):
                with patch('src.agents.conversation_manager.ConversationManager._review_with_agent_async') as mock_review_async:
                    mock_review_result = ReviewResult(
                        agent_name="Test Agent",
                        agent_role="Tester",
                        feedback="Test async review response",
                        timestamp=datetime.now()
                    )
                    mock_review_async.return_value = mock_review_result
                    
                    manager = ConversationManager(persona_loader=mock_persona_loader, enable_analysis=False)
                    manager.agents = [mock_agent]
                    
                    result = await manager.run_review_async(sample_content)
                    
                    assert isinstance(result, ConversationResult)
                    assert len(result.reviews) == 1
                    assert result.reviews[0].agent_name == "Test Agent"
                    assert result.reviews[0].feedback == "Test async review response"
    
    def test_filter_agents_all(self, mock_persona_loader):
        """Test filtering agents with None (all agents)."""
        mock_agent1 = Mock()
        mock_agent1.persona.name = "Agent 1"
        mock_agent2 = Mock()
        mock_agent2.persona.name = "Agent 2"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.agents = [mock_agent1, mock_agent2]
            
            filtered = manager._filter_agents(None)
            
            assert len(filtered) == 2
            assert filtered == [mock_agent1, mock_agent2]
    
    def test_filter_agents_specific(self, mock_persona_loader):
        """Test filtering agents with specific names."""
        mock_agent1 = Mock()
        mock_agent1.persona.name = "Agent 1"
        mock_agent2 = Mock()
        mock_agent2.persona.name = "Agent 2"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.agents = [mock_agent1, mock_agent2]
            
            filtered = manager._filter_agents(["Agent 1"])
            
            assert len(filtered) == 1
            assert filtered[0] == mock_agent1
    
    def test_filter_agents_not_found(self, mock_persona_loader):
        """Test filtering agents when none match."""
        mock_agent1 = Mock()
        mock_agent1.persona.name = "Agent 1"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.agents = [mock_agent1]
            
            filtered = manager._filter_agents(["Nonexistent Agent"])
            
            # Should return all agents when none match
            assert len(filtered) == 1
            assert filtered[0] == mock_agent1
    
    def test_prepare_content_for_review_no_context(self, mock_persona_loader, sample_content):
        """Test preparing content for review without context."""
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            prepared = manager._prepare_content_for_review(sample_content, None)
            
            assert prepared == sample_content
    
    def test_prepare_content_for_review_with_context(self, mock_persona_loader, sample_content):
        """Test preparing content for review with context."""
        mock_context_result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        mock_context_agent = Mock()
        mock_context_agent.format_context_for_review.return_value = "## CONTEXT\nTest context\n---\n"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.context_agents = [mock_context_agent]
            
            prepared = manager._prepare_content_for_review(sample_content, [mock_context_result])
            
            assert "## CONTEXT" in prepared
            assert "Test context" in prepared
            assert "## CONTENT TO REVIEW" in prepared
            assert sample_content in prepared


class TestReviewResult:
    """Test ReviewResult dataclass."""
    
    def test_review_result_creation(self):
        """Test creating a ReviewResult."""
        timestamp = datetime.now()
        result = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        assert result.agent_name == "Test Agent"
        assert result.agent_role == "Tester"
        assert result.feedback == "Test feedback"
        assert result.timestamp == timestamp
        assert result.error is None
    
    def test_review_result_with_error(self):
        """Test creating a ReviewResult with error."""
        timestamp = datetime.now()
        result = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="",
            timestamp=timestamp,
            error="Test error"
        )
        
        assert result.error == "Test error"
        assert result.feedback == ""


class TestConversationResult:
    """Test ConversationResult dataclass."""
    
    def test_conversation_result_creation(self):
        """Test creating a ConversationResult."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        result = ConversationResult(
            content="Test content",
            reviews=[review],
            timestamp=timestamp
        )
        
        assert result.content == "Test content"
        assert len(result.reviews) == 1
        assert result.reviews[0] == review
        assert result.timestamp == timestamp
        assert result.summary is None
        assert result.analysis_results == []
        assert result.context_results == []
        assert result.original_content is None
