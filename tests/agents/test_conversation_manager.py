"""Tests for ConversationManager."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

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
    
    def test_format_results_basic(self, mock_persona_loader, sample_content):
        """Test basic format_results functionality."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            timestamp=timestamp
        )
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            formatted = manager.format_results(result)
            
            assert "# Review-Crew Analysis Results" in formatted
            assert "## Summary" in formatted
            assert "## Individual Reviews" in formatted
            assert "Test Agent" in formatted
            assert "Test feedback" in formatted
            assert "Sample API Documentation" in formatted  # Key content from sample
    
    def test_format_results_no_content(self, mock_persona_loader, sample_content):
        """Test format_results with include_content=False."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            timestamp=timestamp
        )
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            formatted = manager.format_results(result, include_content=False)
            
            assert "# Review-Crew Analysis Results" in formatted
            assert "## Content Reviewed" not in formatted
            assert "Sample API Documentation" not in formatted  # Content should not be included
            assert "Test Agent" in formatted
            assert "Test feedback" in formatted
    
    def test_format_results_with_context_false(self, mock_persona_loader, sample_content):
        """Test format_results with include_context=False (default)."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        context_result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            context_results=[context_result],
            timestamp=timestamp
        )
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            formatted = manager.format_results(result, include_context=False)
            
            assert "# Review-Crew Analysis Results" in formatted
            assert "## Context Information" not in formatted
            assert "Test context" not in formatted
            assert "Test summary" not in formatted
            assert "- **Context Results:** 1 contextualizers 🔍" in formatted
    
    def test_format_results_with_context_true(self, mock_persona_loader, sample_content):
        """Test format_results with include_context=True."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        context_result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            context_results=[context_result],
            timestamp=timestamp
        )
        
        # Mock context agent with persona
        mock_context_agent = Mock()
        mock_context_agent.persona.name = "Test Contextualizer"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.context_agents = [mock_context_agent]
            
            formatted = manager.format_results(result, include_context=True)
            
            assert "# Review-Crew Analysis Results" in formatted
            assert "## Context Information" in formatted
            assert "Test Contextualizer" in formatted
            assert "**Summary:** Test summary" in formatted
            assert "**Formatted Context:**" in formatted
            assert "Test context" in formatted
            assert "- **Context Results:** 1 contextualizers 🔍" in formatted
    
    def test_format_results_with_multiple_contexts(self, mock_persona_loader, sample_content):
        """Test format_results with multiple context results."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        context_result1 = ContextResult(
            formatted_context="Test context 1",
            context_summary="Test summary 1"
        )
        
        context_result2 = ContextResult(
            formatted_context="Test context 2",
            context_summary="Test summary 2"
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            context_results=[context_result1, context_result2],
            timestamp=timestamp
        )
        
        # Mock context agents with personas
        mock_context_agent1 = Mock()
        mock_context_agent1.persona.name = "Contextualizer 1"
        mock_context_agent2 = Mock()
        mock_context_agent2.persona.name = "Contextualizer 2"
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            manager.context_agents = [mock_context_agent1, mock_context_agent2]
            
            formatted = manager.format_results(result, include_context=True)
            
            assert "## Context Information" in formatted
            assert "1. Contextualizer 1" in formatted
            assert "2. Contextualizer 2" in formatted
            assert "Test context 1" in formatted
            assert "Test context 2" in formatted
            assert "Test summary 1" in formatted
            assert "Test summary 2" in formatted
            assert "---" in formatted  # Separator between contexts
    
    def test_format_results_with_context_no_agent_names(self, mock_persona_loader, sample_content):
        """Test format_results with context but no agent names available."""
        timestamp = datetime.now()
        review = ReviewResult(
            agent_name="Test Agent",
            agent_role="Tester",
            feedback="Test feedback",
            timestamp=timestamp
        )
        
        context_result = ContextResult(
            formatted_context="Test context",
            context_summary="Test summary"
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[review],
            context_results=[context_result],
            timestamp=timestamp
        )
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            # No context agents set, should use generic names
            
            formatted = manager.format_results(result, include_context=True)
            
            assert "## Context Information" in formatted
            assert "1. Contextualizer 1" in formatted  # Generic name
            assert "Test context" in formatted
            assert "Test summary" in formatted
    
    def test_format_results_with_failed_reviews(self, mock_persona_loader, sample_content):
        """Test format_results with failed reviews."""
        timestamp = datetime.now()
        successful_review = ReviewResult(
            agent_name="Good Agent",
            agent_role="Tester",
            feedback="Good feedback",
            timestamp=timestamp
        )
        
        failed_review = ReviewResult(
            agent_name="Bad Agent",
            agent_role="Tester",
            feedback="",
            timestamp=timestamp,
            error="Something went wrong"
        )
        
        result = ConversationResult(
            content=sample_content,
            reviews=[successful_review, failed_review],
            timestamp=timestamp
        )
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            formatted = manager.format_results(result)
            
            assert "## Summary" in formatted
            assert "- **Total Reviews:** 2" in formatted
            assert "- **Successful:** 1 ✅" in formatted
            assert "- **Failed:** 1 ❌" in formatted
            assert "## Individual Reviews" in formatted
            assert "Good Agent" in formatted
            assert "Good feedback" in formatted
            assert "## Failed Reviews" in formatted
            assert "Bad Agent" in formatted
            assert "Something went wrong" in formatted


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


class TestManifestDocumentLoading:
    """Test manifest-driven document loading functionality."""
    
    def test_collect_documents_from_manifest_primary_and_supporting(self, mock_persona_loader, tmp_path):
        """Test loading primary and supporting documents from manifest."""
        # Create test files
        primary_file = tmp_path / "primary.md"
        primary_file.write_text("# Primary Document\nThis is the main content.")
        
        supporting1_file = tmp_path / "supporting1.md"
        supporting1_file.write_text("# Supporting Document 1\nSupporting content 1.")
        
        supporting2_file = tmp_path / "supporting2.md" 
        supporting2_file.write_text("# Supporting Document 2\nSupporting content 2.")
        
        # Create manifest config
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "primary": "primary.md",
                    "supporting": ["supporting1.md", "supporting2.md"]
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            assert len(documents) == 3
            
            # Check primary document
            primary_docs = [d for d in documents if d["type"] == "primary"]
            assert len(primary_docs) == 1
            assert primary_docs[0]["name"] == "primary.md"
            assert primary_docs[0]["content"] == "# Primary Document\nThis is the main content."
            assert primary_docs[0]["manifest_path"] == "primary.md"
            
            # Check supporting documents
            supporting_docs = [d for d in documents if d["type"] == "supporting"]
            assert len(supporting_docs) == 2
            supporting_names = [d["name"] for d in supporting_docs]
            assert "supporting1.md" in supporting_names
            assert "supporting2.md" in supporting_names
    
    def test_collect_documents_from_manifest_relative_paths(self, mock_persona_loader, tmp_path):
        """Test loading documents with relative paths including ../paths."""
        # Create nested directory structure
        base_dir = tmp_path / "project"
        base_dir.mkdir()
        parent_dir = tmp_path
        
        # Create files in different locations
        primary_file = parent_dir / "common.md"
        primary_file.write_text("# Common Document\nShared content.")
        
        supporting_file = base_dir / "specific.md"
        supporting_file.write_text("# Specific Document\nProject-specific content.")
        
        # Create manifest config with relative paths
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "primary": "../common.md",  # Go up one level
                    "supporting": ["specific.md"]  # Relative to base_dir
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, base_dir)
            
            assert len(documents) == 2
            
            # Check primary document (from parent directory)
            primary_docs = [d for d in documents if d["type"] == "primary"]
            assert len(primary_docs) == 1
            assert primary_docs[0]["name"] == "common.md"
            assert primary_docs[0]["content"] == "# Common Document\nShared content."
            assert primary_docs[0]["manifest_path"] == "../common.md"
            
            # Check supporting document (from base directory)
            supporting_docs = [d for d in documents if d["type"] == "supporting"]
            assert len(supporting_docs) == 1
            assert supporting_docs[0]["name"] == "specific.md"
            assert supporting_docs[0]["content"] == "# Specific Document\nProject-specific content."
    
    def test_collect_documents_from_manifest_missing_files(self, mock_persona_loader, tmp_path):
        """Test handling of missing files in manifest."""
        # Create only one of the files
        existing_file = tmp_path / "existing.md"
        existing_file.write_text("# Existing Document\nThis file exists.")
        
        # Create manifest config with missing files
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "primary": "missing_primary.md",  # This file doesn't exist
                    "supporting": ["existing.md", "missing_supporting.md"]  # One exists, one doesn't
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            # Should only load the existing file
            assert len(documents) == 1
            assert documents[0]["name"] == "existing.md"
            assert documents[0]["type"] == "supporting"
            assert documents[0]["content"] == "# Existing Document\nThis file exists."
    
    def test_collect_documents_from_manifest_no_documents_section(self, mock_persona_loader, tmp_path):
        """Test handling of manifest with no documents section."""
        manifest_config = {
            "review_configuration": {
                "name": "Test Review",
                "reviewers": ["Test Reviewer"]
                # No documents section
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            assert len(documents) == 0
    
    def test_collect_documents_from_manifest_empty_documents_section(self, mock_persona_loader, tmp_path):
        """Test handling of manifest with empty documents section."""
        manifest_config = {
            "review_configuration": {
                "documents": {}  # Empty documents section
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            assert len(documents) == 0
    
    def test_collect_documents_from_manifest_only_primary(self, mock_persona_loader, tmp_path):
        """Test loading only primary document when no supporting documents specified."""
        # Create test file
        primary_file = tmp_path / "primary_only.md"
        primary_file.write_text("# Primary Only\nThis is the only document.")
        
        # Create manifest config with only primary
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "primary": "primary_only.md"
                    # No supporting documents
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            assert len(documents) == 1
            assert documents[0]["type"] == "primary"
            assert documents[0]["name"] == "primary_only.md"
            assert documents[0]["content"] == "# Primary Only\nThis is the only document."
    
    def test_collect_documents_from_manifest_only_supporting(self, mock_persona_loader, tmp_path):
        """Test loading only supporting documents when no primary specified."""
        # Create test files
        supporting1_file = tmp_path / "support1.md"
        supporting1_file.write_text("# Support 1\nSupporting content 1.")
        
        supporting2_file = tmp_path / "support2.md"
        supporting2_file.write_text("# Support 2\nSupporting content 2.")
        
        # Create manifest config with only supporting
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "supporting": ["support1.md", "support2.md"]
                    # No primary document
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
            
            assert len(documents) == 2
            assert all(d["type"] == "supporting" for d in documents)
            supporting_names = [d["name"] for d in documents]
            assert "support1.md" in supporting_names
            assert "support2.md" in supporting_names
    
    def test_collect_documents_from_manifest_file_read_error(self, mock_persona_loader, tmp_path):
        """Test handling of file read errors."""
        # Create a file that exists but will cause read error
        problematic_file = tmp_path / "problematic.md"
        problematic_file.write_text("# Test\nContent.")
        
        # Create manifest config
        manifest_config = {
            "review_configuration": {
                "documents": {
                    "primary": "problematic.md"
                }
            }
        }
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            # Mock open to raise an exception
            with patch('builtins.open', side_effect=IOError("Permission denied")):
                documents = manager._collect_documents_from_manifest(manifest_config, tmp_path)
                
                # Should handle error gracefully and return empty list
                assert len(documents) == 0
    
    def test_resolve_document_path_relative(self, mock_persona_loader):
        """Test path resolution for various relative path formats."""
        base_dir = Path("/base/project")
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            # Test relative path starting with ../
            resolved = manager._resolve_document_path("../common.md", base_dir)
            assert resolved == Path("/base/common.md")
            
            # Test simple relative path
            resolved = manager._resolve_document_path("local.md", base_dir)
            assert resolved == base_dir / "local.md"
            
            # Test absolute path
            resolved = manager._resolve_document_path("/absolute/path.md", base_dir)
            assert resolved == Path("/absolute/path.md")
    
    def test_resolve_document_path_error_handling(self, mock_persona_loader):
        """Test path resolution error handling."""
        base_dir = Path("/base/project")
        
        with patch('src.agents.conversation_manager.AnalysisAgent'):
            manager = ConversationManager(persona_loader=mock_persona_loader)
            
            # Test with invalid path that causes exception
            with patch('pathlib.Path.__truediv__', side_effect=OSError("Invalid path")):
                resolved = manager._resolve_document_path("invalid.md", base_dir)
                assert resolved is None


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
