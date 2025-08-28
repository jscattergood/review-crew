"""
Tests for ReviewGraphBuilder.

This module tests the ReviewGraphBuilder class that constructs Strands graphs
using our refactored agents that directly inherit from MultiAgentBase.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.agents.review_graph_builder import ReviewGraphBuilder
from src.agents.review_agent import ReviewAgent
from src.agents.context_agent import ContextAgent
from src.agents.analysis_agent import AnalysisAgent
from src.config.persona_loader import PersonaLoader, PersonaConfig
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentResult, Status


class TestReviewGraphBuilder:
    """Test cases for ReviewGraphBuilder."""
    
    @pytest.fixture
    def mock_persona_loader(self):
        """Create a mock PersonaLoader."""
        loader = Mock(spec=PersonaLoader)
        
        # Mock reviewer personas
        reviewer_persona = Mock(spec=PersonaConfig)
        reviewer_persona.name = "Test Reviewer"
        reviewer_persona.role = "Content Reviewer"
        reviewer_persona.goal = "Review content"
        reviewer_persona.backstory = "Expert reviewer"
        reviewer_persona.prompt_template = "Review: {content}"
        reviewer_persona.model_config = {}
        
        loader.load_reviewer_personas.return_value = [reviewer_persona]
        
        # Mock contextualizer personas
        context_persona = Mock(spec=PersonaConfig)
        context_persona.name = "Test Contextualizer"
        context_persona.role = "Context Processor"
        context_persona.goal = "Process context"
        context_persona.backstory = "Expert contextualizer"
        context_persona.prompt_template = "Process: {content}"
        context_persona.model_config = {}
        
        loader.load_contextualizer_personas.return_value = [context_persona]
        
        # Mock analyzer personas
        analyzer_persona = Mock(spec=PersonaConfig)
        analyzer_persona.name = "Test Analyzer"
        analyzer_persona.role = "Analysis Specialist"
        analyzer_persona.goal = "Analyze reviews"
        analyzer_persona.backstory = "Expert analyzer"
        analyzer_persona.prompt_template = "Analyze: {content}"
        analyzer_persona.model_config = {}
        
        loader.load_analyzer_personas.return_value = [analyzer_persona]
        
        # Mock manifest loading methods
        loader.load_contextualizers_from_manifest.return_value = [context_persona]
        loader.load_reviewers_from_manifest.return_value = [reviewer_persona]
        loader.load_analyzers_from_manifest.return_value = [analyzer_persona]
        
        return loader
    
    @pytest.fixture
    def builder(self, mock_persona_loader):
        """Create a ReviewGraphBuilder for testing."""
        with patch('src.agents.base_agent.Agent'):  # Mock the internal Strands Agent
            return ReviewGraphBuilder(
                persona_loader=mock_persona_loader,
                model_provider="bedrock",
                model_config={"test": "config"},
                enable_analysis=True
            )
    
    def test_init(self, builder, mock_persona_loader):
        """Test ReviewGraphBuilder initialization."""
        assert builder.persona_loader == mock_persona_loader
        assert builder.model_provider == "bedrock"
        assert builder.model_config == {"test": "config"}
        assert builder.enable_analysis is True
        assert len(builder.review_agents) == 1
        assert len(builder.context_agents) == 1
        assert len(builder.analysis_agents) == 1
    
    def test_init_without_analysis(self, mock_persona_loader):
        """Test ReviewGraphBuilder initialization with analysis disabled."""
        with patch('src.agents.base_agent.Agent'):
            builder = ReviewGraphBuilder(
                persona_loader=mock_persona_loader,
                enable_analysis=False
            )
        
        assert builder.enable_analysis is False
        assert len(builder.analysis_agents) == 0
    
    def test_init_default_persona_loader(self):
        """Test ReviewGraphBuilder initialization with default PersonaLoader."""
        with patch('src.agents.base_agent.Agent'), \
             patch('src.agents.review_graph_builder.PersonaLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_reviewer_personas.return_value = []
            mock_loader.load_contextualizer_personas.return_value = []
            mock_loader.load_analyzer_personas.return_value = []
            mock_loader_class.return_value = mock_loader
            
            builder = ReviewGraphBuilder()
            
            assert builder.persona_loader == mock_loader
            mock_loader_class.assert_called_once()
    
    @patch('src.agents.review_graph_builder.GraphBuilder')
    def test_build_standard_review_graph(self, mock_graph_builder_class, builder):
        """Test building a standard review graph."""
        mock_builder = Mock()
        mock_graph = Mock()
        mock_builder.build.return_value = mock_graph
        mock_graph_builder_class.return_value = mock_builder
        
        result = builder.build_standard_review_graph()
        
        assert result == mock_graph
        
        # Verify graph construction calls
        mock_builder.add_node.assert_called()
        mock_builder.add_edge.assert_called()
        mock_builder.set_entry_point.assert_called_with("document_processor")
        mock_builder.build.assert_called_once()
    
    @patch('src.agents.review_graph_builder.GraphBuilder')
    def test_build_standard_review_graph_with_selections(self, mock_graph_builder_class, builder):
        """Test building a standard review graph with agent selections."""
        mock_builder = Mock()
        mock_graph = Mock()
        mock_builder.build.return_value = mock_graph
        mock_graph_builder_class.return_value = mock_builder
        
        result = builder.build_standard_review_graph(
            selected_reviewers=["Test Reviewer"],
            selected_contextualizers=["Test Contextualizer"],
            selected_analyzers=["Test Analyzer"]
        )
        
        assert result == mock_graph
        mock_builder.build.assert_called_once()
    
    @patch('src.agents.review_graph_builder.GraphBuilder')
    def test_build_manifest_driven_graph(self, mock_graph_builder_class, builder):
        """Test building a manifest-driven review graph."""
        mock_builder = Mock()
        mock_graph = Mock()
        mock_builder.build.return_value = mock_graph
        mock_graph_builder_class.return_value = mock_builder
        
        manifest_config = {
            "review_configuration": {
                "reviewers": {"names": ["Test Reviewer"]},
                "contextualizers": {"names": ["Test Contextualizer"]},
                "analyzers": {"names": ["Test Analyzer"]},
                "processed_focus": {
                    "focus_instructions": ["Focus on accuracy"]
                }
            }
        }
        
        result = builder.build_manifest_driven_graph(manifest_config)
        
        assert result == mock_graph
        mock_builder.build.assert_called_once()
    
    @patch('src.agents.review_graph_builder.GraphBuilder')
    def test_build_simple_review_graph(self, mock_graph_builder_class, builder):
        """Test building a simple review graph."""
        mock_builder = Mock()
        mock_graph = Mock()
        mock_builder.build.return_value = mock_graph
        mock_graph_builder_class.return_value = mock_builder
        
        result = builder.build_simple_review_graph("test content")
        
        assert result == mock_graph
        mock_builder.build.assert_called_once()
    
    def test_filter_review_agents_all(self, builder):
        """Test filtering review agents with None (all agents)."""
        result = builder._filter_review_agents(None)
        assert result == builder.review_agents
    
    def test_filter_review_agents_specific(self, builder):
        """Test filtering review agents with specific names."""
        result = builder._filter_review_agents(["Test Reviewer"])
        assert len(result) == 1
        assert result[0].persona.name == "Test Reviewer"
    
    def test_filter_review_agents_not_found(self, builder):
        """Test filtering review agents with non-existent names."""
        result = builder._filter_review_agents(["Nonexistent Agent"])
        # Should fallback to all agents
        assert result == builder.review_agents
    
    def test_filter_context_agents_all(self, builder):
        """Test filtering context agents with None (all agents)."""
        result = builder._filter_context_agents(None)
        assert result == builder.context_agents
    
    def test_filter_context_agents_specific(self, builder):
        """Test filtering context agents with specific names."""
        result = builder._filter_context_agents(["Test Contextualizer"])
        assert len(result) == 1
        assert result[0].persona.name == "Test Contextualizer"
    
    def test_filter_analysis_agents_all(self, builder):
        """Test filtering analysis agents with None (all agents)."""
        result = builder._filter_analysis_agents(None)
        assert result == builder.analysis_agents
    
    def test_filter_analysis_agents_specific(self, builder):
        """Test filtering analysis agents with specific names."""
        result = builder._filter_analysis_agents(["Test Analyzer"])
        assert len(result) == 1
        assert result[0].persona.name == "Test Analyzer"
    
    def test_load_contextualizers_from_manifest_success(self, builder):
        """Test loading contextualizers from manifest successfully."""
        review_config = {"contextualizers": {"names": ["Test Contextualizer"]}}
        
        result = builder._load_contextualizers_from_manifest(review_config)
        
        assert len(result) == 1
        assert isinstance(result[0], ContextAgent)
        builder.persona_loader.load_contextualizers_from_manifest.assert_called_once_with(review_config)
    
    def test_load_contextualizers_from_manifest_error(self, builder):
        """Test loading contextualizers from manifest with error."""
        builder.persona_loader.load_contextualizers_from_manifest.side_effect = Exception("Load error")
        review_config = {"contextualizers": {"names": ["Test Contextualizer"]}}
        
        result = builder._load_contextualizers_from_manifest(review_config)
        
        # Should fallback to all available context agents
        assert result == builder.context_agents
    
    def test_load_reviewers_from_manifest_success(self, builder):
        """Test loading reviewers from manifest successfully."""
        review_config = {"reviewers": {"names": ["Test Reviewer"]}}
        
        result = builder._load_reviewers_from_manifest(review_config)
        
        assert len(result) == 1
        assert isinstance(result[0], ReviewAgent)
        builder.persona_loader.load_reviewers_from_manifest.assert_called_once_with(review_config)
    
    def test_load_reviewers_from_manifest_error(self, builder):
        """Test loading reviewers from manifest with error."""
        builder.persona_loader.load_reviewers_from_manifest.side_effect = Exception("Load error")
        review_config = {"reviewers": {"names": ["Test Reviewer"]}}
        
        result = builder._load_reviewers_from_manifest(review_config)
        
        # Should fallback to all available review agents
        assert result == builder.review_agents
    
    def test_load_analyzers_from_manifest_success(self, builder):
        """Test loading analyzers from manifest successfully."""
        review_config = {"analyzers": {"names": ["Test Analyzer"]}}
        
        result = builder._load_analyzers_from_manifest(review_config)
        
        assert len(result) == 1
        assert isinstance(result[0], AnalysisAgent)
        builder.persona_loader.load_analyzers_from_manifest.assert_called_once_with(review_config)
    
    def test_load_analyzers_from_manifest_disabled(self, mock_persona_loader):
        """Test loading analyzers from manifest when analysis is disabled."""
        with patch('src.agents.base_agent.Agent'):
            builder = ReviewGraphBuilder(
                persona_loader=mock_persona_loader,
                enable_analysis=False
            )
        
        result = builder._load_analyzers_from_manifest({})
        
        assert result == []
    
    def test_load_analyzers_from_manifest_error(self, builder):
        """Test loading analyzers from manifest with error."""
        builder.persona_loader.load_analyzers_from_manifest.side_effect = Exception("Load error")
        review_config = {"analyzers": {"names": ["Test Analyzer"]}}
        
        result = builder._load_analyzers_from_manifest(review_config)
        
        # Should fallback to all available analysis agents
        assert result == builder.analysis_agents
    
    def test_apply_focus_to_reviewer(self, builder):
        """Test applying focus instructions to a reviewer."""
        reviewer = builder.review_agents[0]
        focus_config = {
            "focus_instructions": [
                "🔴 CRITICAL: Pay attention to accuracy",
                "🟡 HIGH PRIORITY: Check grammar"
            ]
        }
        
        result = builder._apply_focus_to_reviewer(reviewer, focus_config)
        
        assert isinstance(result, ReviewAgent)
        assert result.persona.name == reviewer.persona.name
        assert "SPECIAL FOCUS AREAS" in result.persona.prompt_template
        assert "Pay attention to accuracy" in result.persona.prompt_template
        assert "Check grammar" in result.persona.prompt_template
    
    def test_apply_focus_to_reviewer_no_instructions(self, builder):
        """Test applying focus to reviewer with no instructions."""
        reviewer = builder.review_agents[0]
        focus_config = {"focus_instructions": []}
        
        result = builder._apply_focus_to_reviewer(reviewer, focus_config)
        
        assert result == reviewer  # Should return original reviewer
    
    def test_get_available_agents_info(self, builder):
        """Test getting available agents information."""
        result = builder.get_available_agents_info()
        
        assert "reviewers" in result
        assert "contextualizers" in result
        assert "analyzers" in result
        
        assert len(result["reviewers"]) == 1
        assert len(result["contextualizers"]) == 1
        assert len(result["analyzers"]) == 1
        
        # Check structure of agent info
        reviewer_info = result["reviewers"][0]
        assert "name" in reviewer_info
        assert "role" in reviewer_info
        assert "goal" in reviewer_info
    
    def test_create_conditional_edge_function_document_type(self, builder):
        """Test creating document type conditional edge function."""
        condition_func = builder.create_conditional_edge_function(
            "document_type",
            expected_type="multi"
        )
        
        # Mock MultiAgentResult with document processor result
        mock_result = Mock(spec=MultiAgentResult)
        mock_node_result = Mock()
        mock_agent_result = Mock()
        mock_doc_result = Mock()
        mock_doc_result.document_type = "multi"
        
        mock_agent_result.state = {"document_processor_result": mock_doc_result}
        mock_node_result.result = mock_agent_result
        mock_result.results = {"document_processor": mock_node_result}
        
        assert condition_func(mock_result) is True
        
        # Test with different document type
        mock_doc_result.document_type = "single"
        assert condition_func(mock_result) is False
    
    def test_create_conditional_edge_function_has_context(self, builder):
        """Test creating has context conditional edge function."""
        condition_func = builder.create_conditional_edge_function("has_context")
        
        # Mock MultiAgentResult with successful context result
        mock_result = Mock(spec=MultiAgentResult)
        mock_node_result = Mock()
        mock_agent_result = Mock()
        
        mock_agent_result.state = {"response": "context data"}
        mock_node_result.result = mock_agent_result
        mock_result.results = {builder.context_agents[0].name: mock_node_result}
        
        assert condition_func(mock_result) is True
        
        # Test with no context data
        mock_agent_result.state = {"response": None}
        assert condition_func(mock_result) is False
    
    def test_create_conditional_edge_function_reviews_successful(self, builder):
        """Test creating reviews successful conditional edge function."""
        condition_func = builder.create_conditional_edge_function(
            "reviews_successful",
            min_reviews=1
        )
        
        # Mock MultiAgentResult with successful reviews
        mock_result = Mock(spec=MultiAgentResult)
        mock_node_result = Mock()
        mock_agent_result = Mock()
        
        mock_agent_result.state = {}  # No error means success
        mock_node_result.result = mock_agent_result
        mock_result.results = {builder.review_agents[0].name: mock_node_result}
        
        assert condition_func(mock_result) is True
        
        # Test with failed review
        mock_agent_result.state = {"error": "Review failed"}
        assert condition_func(mock_result) is False
    
    def test_create_conditional_edge_function_unknown_type(self, builder):
        """Test creating conditional edge function with unknown type."""
        with pytest.raises(ValueError, match="Unknown condition type"):
            builder.create_conditional_edge_function("unknown_type")
    
    def test_load_review_agents_error_handling(self, mock_persona_loader):
        """Test error handling when loading review agents fails."""
        mock_persona_loader.load_reviewer_personas.side_effect = Exception("Load error")
        
        with patch('src.agents.base_agent.Agent'):
            builder = ReviewGraphBuilder(persona_loader=mock_persona_loader)
        
        assert len(builder.review_agents) == 0
    
    def test_load_context_agents_error_handling(self, mock_persona_loader):
        """Test error handling when loading context agents fails."""
        mock_persona_loader.load_contextualizer_personas.side_effect = Exception("Load error")
        
        with patch('src.agents.base_agent.Agent'):
            builder = ReviewGraphBuilder(persona_loader=mock_persona_loader)
        
        assert len(builder.context_agents) == 0
    
    def test_load_analysis_agents_error_handling(self, mock_persona_loader):
        """Test error handling when loading analysis agents fails."""
        mock_persona_loader.load_analyzer_personas.side_effect = Exception("Load error")
        
        with patch('src.agents.base_agent.Agent'):
            builder = ReviewGraphBuilder(
                persona_loader=mock_persona_loader, 
                enable_analysis=True
            )
        
        assert len(builder.analysis_agents) == 0
    
    @pytest.mark.asyncio
    async def test_execute_graph_async(self, builder):
        """Test asynchronous graph execution."""
        mock_graph = Mock()
        mock_result = Mock(spec=MultiAgentResult)
        
        # Mock the async method properly
        async def mock_invoke_async(input_data):
            return mock_result
        mock_graph.invoke_async = mock_invoke_async
        
        result = await builder.execute_graph(mock_graph, "test input")
        
        assert result == mock_result
    
    def test_execute_graph_sync(self, builder):
        """Test synchronous graph execution."""
        mock_graph = Mock()
        mock_result = Mock(spec=MultiAgentResult)
        mock_graph.return_value = mock_result
        
        result = builder.execute_graph_sync(mock_graph, "test input")
        
        assert result == mock_result
        mock_graph.assert_called_once_with("test input")
    
    def test_agents_are_multiagent_base_compatible(self, builder):
        """Test that our agents are compatible with MultiAgentBase interface."""
        # Test that agents have the required methods
        for agent in builder.review_agents:
            assert hasattr(agent, '__call__')
            assert hasattr(agent, 'invoke_async_graph')
            assert hasattr(agent, 'name')
        
        for agent in builder.context_agents:
            assert hasattr(agent, '__call__')
            assert hasattr(agent, 'invoke_async_graph')
            assert hasattr(agent, 'name')
        
        for agent in builder.analysis_agents:
            assert hasattr(agent, '__call__')
            assert hasattr(agent, 'invoke_async_graph')
            assert hasattr(agent, 'name')
