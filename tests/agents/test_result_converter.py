"""
Tests for ResultConverter.

This module tests the ResultConverter class that converts Strands MultiAgentResult
back to ConversationResult format for backward compatibility.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from strands.multiagent.base import MultiAgentResult, NodeResult, Status
from strands.agent.agent_result import AgentResult
from strands.types.content import ContentBlock, Message

from src.agents.result_converter import ResultConverter, convert_graph_result_to_conversation
from src.agents.conversation_manager import ConversationResult, ReviewResult
from src.agents.analysis_agent import AnalysisResult
from src.agents.context_agent import ContextResult
from src.agents.document_processor_node import DocumentProcessorResult


class TestResultConverter:
    """Test cases for ResultConverter."""
    
    @pytest.fixture
    def converter(self):
        """Create a ResultConverter for testing."""
        return ResultConverter()
    
    @pytest.fixture
    def mock_graph_result(self):
        """Create a mock MultiAgentResult for testing."""
        # Create mock document processor result
        doc_processor_result = Mock()
        doc_processor_result.document_type = "multi"
        doc_processor_result.compiled_content = "Test compiled content"
        doc_processor_result.documents = [{"name": "test.txt", "content": "test"}]
        doc_processor_result.manifest_config = None
        doc_processor_result.validation_results = None
        doc_processor_result.original_path = "/test/path"
        
        # Create mock agent results
        doc_agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Document processed")]),
            metrics={},
            state={"document_processor_result": doc_processor_result}
        )
        
        review_agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Great review feedback")]),
            metrics={},
            state={
                "agent_name": "Test Reviewer",
                "agent_role": "Content Reviewer",
                "response": "Great review feedback"
            }
        )
        
        context_agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Context information")]),
            metrics={},
            state={
                "agent_name": "Test Contextualizer",
                "agent_role": "Context Processor",
                "response": "Context information about the document"
            }
        )
        
        analysis_agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Analysis synthesis")]),
            metrics={},
            state={
                "agent_name": "Test Analyzer",
                "agent_role": "Analysis Specialist",
                "response": "Analysis synthesis of all reviews"
            }
        )
        
        # Create node results
        doc_node_result = NodeResult(result=doc_agent_result, status=Status.COMPLETED)
        review_node_result = NodeResult(result=review_agent_result, status=Status.COMPLETED)
        context_node_result = NodeResult(result=context_agent_result, status=Status.COMPLETED)
        analysis_node_result = NodeResult(result=analysis_agent_result, status=Status.COMPLETED)
        
        # Create MultiAgentResult
        graph_result = MultiAgentResult(
            status=Status.COMPLETED,
            results={
                "document_processor": doc_node_result,
                "test_reviewer": review_node_result,
                "test_contextualizer": context_node_result,
                "test_analyzer": analysis_node_result,
            }
        )
        
        return graph_result
    
    def test_init(self, converter):
        """Test ResultConverter initialization."""
        assert isinstance(converter, ResultConverter)
    
    def test_convert_to_conversation_result(self, converter, mock_graph_result):
        """Test converting MultiAgentResult to ConversationResult."""
        result = converter.convert_to_conversation_result(mock_graph_result)
        
        assert isinstance(result, ConversationResult)
        assert result.content == "Test compiled content"
        assert len(result.reviews) == 1
        assert len(result.context_results) == 1
        assert len(result.analysis_results) == 1
        assert result.analysis_errors == []
        assert isinstance(result.timestamp, datetime)
    
    def test_convert_with_original_content(self, converter, mock_graph_result):
        """Test converting with original content provided."""
        original_content = "Original test content"
        result = converter.convert_to_conversation_result(mock_graph_result, original_content)
        
        assert result.original_content == original_content
        # Should still use compiled content from document processor
        assert result.content == "Test compiled content"
    
    def test_extract_reviews(self, converter, mock_graph_result):
        """Test extracting review results."""
        reviews = converter._extract_reviews(mock_graph_result)
        
        assert len(reviews) == 1
        review = reviews[0]
        assert isinstance(review, ReviewResult)
        assert review.agent_name == "Test Reviewer"
        assert review.agent_role == "Content Reviewer"
        assert review.feedback == "Great review feedback"
        assert review.error is None
    
    def test_extract_reviews_with_error(self, converter):
        """Test extracting review results with errors."""
        # Create a failed review result
        failed_agent_result = AgentResult(
            stop_reason="error",
            message=Message(role="assistant", content=[ContentBlock(text="Review failed")]),
            metrics={},
            state={
                "agent_name": "Failed Reviewer",
                "agent_role": "Content Reviewer",
                "response": "",
                "error": "Processing failed"
            }
        )
        
        failed_node_result = NodeResult(result=failed_agent_result, status=Status.FAILED)
        
        graph_result = MultiAgentResult(
            status=Status.FAILED,
            results={"failed_reviewer": failed_node_result}
        )
        
        reviews = converter._extract_reviews(graph_result)
        
        assert len(reviews) == 1
        review = reviews[0]
        assert review.agent_name == "Failed Reviewer"
        assert review.error == "Processing failed"
    
    def test_extract_context_results(self, converter, mock_graph_result):
        """Test extracting context results."""
        context_results = converter._extract_context_results(mock_graph_result)
        
        assert len(context_results) == 1
        context_result = context_results[0]
        assert isinstance(context_result, ContextResult)
        assert "Context information" in context_result.context_summary
        assert context_result.formatted_context == "Context information about the document"
    
    def test_extract_analysis_results(self, converter, mock_graph_result):
        """Test extracting analysis results."""
        analysis_results = converter._extract_analysis_results(mock_graph_result)
        
        assert len(analysis_results) == 1
        analysis_result = analysis_results[0]
        assert isinstance(analysis_result, AnalysisResult)
        assert analysis_result.synthesis == "Analysis synthesis of all reviews"
    
    def test_extract_analysis_errors(self, converter):
        """Test extracting analysis errors."""
        # Create a failed analysis result
        failed_agent_result = AgentResult(
            stop_reason="error",
            message=Message(role="assistant", content=[ContentBlock(text="Analysis failed")]),
            metrics={},
            state={
                "agent_name": "Failed Analyzer",
                "agent_role": "Analysis Specialist",
                "error": "Analysis processing failed"
            }
        )
        
        failed_node_result = NodeResult(result=failed_agent_result, status=Status.FAILED)
        
        graph_result = MultiAgentResult(
            status=Status.FAILED,
            results={"failed_analyzer": failed_node_result}
        )
        
        errors = converter._extract_analysis_errors(graph_result)
        
        assert len(errors) == 1
        assert "failed_analyzer: Analysis processing failed" in errors[0]
    
    def test_determine_content_from_document_processor(self, converter, mock_graph_result):
        """Test determining content from document processor."""
        content = converter._determine_content(mock_graph_result, "original")
        assert content == "Test compiled content"
    
    def test_determine_content_fallback_to_original(self, converter):
        """Test determining content falls back to original when no document processor."""
        graph_result = MultiAgentResult(status=Status.COMPLETED, results={})
        content = converter._determine_content(graph_result, "original content")
        assert content == "original content"
    
    def test_determine_content_empty_fallback(self, converter):
        """Test determining content falls back to empty string."""
        graph_result = MultiAgentResult(status=Status.COMPLETED, results={})
        content = converter._determine_content(graph_result, None)
        assert content == ""
    
    def test_is_review_node(self, converter):
        """Test identifying review nodes."""
        assert converter._is_review_node("test_reviewer") is True
        assert converter._is_review_node("some_agent") is True
        assert converter._is_review_node("document_processor") is False
        assert converter._is_review_node("contextualizer_0") is False
        assert converter._is_review_node("analyzer_0") is False
    
    def test_is_context_node(self, converter):
        """Test identifying context nodes."""
        assert converter._is_context_node("contextualizer_0") is True
        assert converter._is_context_node("test_contextualizer") is True
        assert converter._is_context_node("context_agent") is True
        assert converter._is_context_node("test_reviewer") is False
        assert converter._is_context_node("analyzer_0") is False
    
    def test_is_analysis_node(self, converter):
        """Test identifying analysis nodes."""
        assert converter._is_analysis_node("analyzer_0") is True
        assert converter._is_analysis_node("test_analyzer") is True
        assert converter._is_analysis_node("analysis_agent") is True
        assert converter._is_analysis_node("test_reviewer") is False
        assert converter._is_analysis_node("contextualizer_0") is False
    
    def test_parse_context_response(self, converter):
        """Test parsing context response."""
        response = "This is a detailed context response with lots of information"
        context_result = converter._parse_context_response(response)
        
        assert isinstance(context_result, ContextResult)
        assert context_result.formatted_context == response
        assert len(context_result.context_summary) <= 203  # 200 + "..."
    
    def test_parse_context_response_short(self, converter):
        """Test parsing short context response."""
        response = "Short context"
        context_result = converter._parse_context_response(response)
        
        assert isinstance(context_result, ContextResult)
        assert context_result.context_summary == response
        assert context_result.formatted_context == response
    
    def test_parse_analysis_response(self, converter):
        """Test parsing analysis response."""
        response = "This is an analysis synthesis"
        analysis_result = converter._parse_analysis_response(response)
        
        assert isinstance(analysis_result, AnalysisResult)
        assert analysis_result.synthesis == response
        assert isinstance(analysis_result.timestamp, datetime)
    
    def test_get_execution_summary(self, converter, mock_graph_result):
        """Test getting execution summary."""
        summary = converter.get_execution_summary(mock_graph_result)
        
        assert summary["total_nodes"] == 4
        assert summary["successful_nodes"] == 4
        assert summary["failed_nodes"] == 0
        assert summary["review_agents"] == 1
        assert summary["context_agents"] == 1
        assert summary["analysis_agents"] == 1
        assert summary["overall_status"] == "completed"
    
    def test_get_execution_summary_with_failures(self, converter):
        """Test getting execution summary with failures."""
        failed_result = AgentResult(
            stop_reason="error",
            message=Message(role="assistant", content=[ContentBlock(text="Failed")]),
            metrics={},
            state={"error": "Test error"}
        )
        
        graph_result = MultiAgentResult(
            status=Status.FAILED,
            results={
                "test_reviewer": NodeResult(result=failed_result, status=Status.FAILED),
                "test_contextualizer": NodeResult(result=failed_result, status=Status.COMPLETED),
            }
        )
        
        summary = converter.get_execution_summary(graph_result)
        
        assert summary["total_nodes"] == 2
        assert summary["successful_nodes"] == 1
        assert summary["failed_nodes"] == 1
        assert summary["overall_status"] == "failed"
    
    def test_extract_document_info(self, converter, mock_graph_result):
        """Test extracting document information."""
        doc_info = converter.extract_document_info(mock_graph_result)
        
        assert doc_info is not None
        assert doc_info["document_type"] == "multi"
        assert doc_info["document_count"] == 1
        assert doc_info["has_manifest"] is False
        assert doc_info["original_path"] == "/test/path"
    
    def test_extract_document_info_no_processor(self, converter):
        """Test extracting document info when no document processor."""
        graph_result = MultiAgentResult(status=Status.COMPLETED, results={})
        doc_info = converter.extract_document_info(graph_result)
        
        assert doc_info is None
    
    def test_convenience_function(self, mock_graph_result):
        """Test the convenience function."""
        result = convert_graph_result_to_conversation(mock_graph_result, "original")
        
        assert isinstance(result, ConversationResult)
        assert result.content == "Test compiled content"
        assert result.original_content == "original"
        assert len(result.reviews) == 1
    
    def test_empty_graph_result(self, converter):
        """Test converting empty graph result."""
        empty_result = MultiAgentResult(status=Status.COMPLETED, results={})
        
        result = converter.convert_to_conversation_result(empty_result)
        
        assert isinstance(result, ConversationResult)
        assert result.content == ""
        assert len(result.reviews) == 0
        assert len(result.context_results) == 0
        assert len(result.analysis_results) == 0
        assert result.analysis_errors == []
    
    def test_mixed_node_types(self, converter):
        """Test with mixed node types and naming patterns."""
        # Create various node types with different naming patterns
        review_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Review")]),
            metrics={},
            state={"agent_name": "Custom Reviewer", "agent_role": "Reviewer", "response": "Review"}
        )
        
        context_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Context")]),
            metrics={},
            state={"agent_name": "Custom Context", "agent_role": "Context", "response": "Context"}
        )
        
        analysis_result = AgentResult(
            stop_reason="end_turn",
            message=Message(role="assistant", content=[ContentBlock(text="Analysis")]),
            metrics={},
            state={"agent_name": "Custom Analyzer", "agent_role": "Analyzer", "response": "Analysis"}
        )
        
        graph_result = MultiAgentResult(
            status=Status.COMPLETED,
            results={
                "custom_reviewer_agent": NodeResult(result=review_result, status=Status.COMPLETED),
                "my_contextualizer": NodeResult(result=context_result, status=Status.COMPLETED),
                "super_analysis_agent": NodeResult(result=analysis_result, status=Status.COMPLETED),
            }
        )
        
        result = converter.convert_to_conversation_result(graph_result)
        
        assert len(result.reviews) == 1
        assert len(result.context_results) == 1
        assert len(result.analysis_results) == 1
        
        assert result.reviews[0].agent_name == "Custom Reviewer"
        assert result.context_results[0].formatted_context == "Context"
        assert result.analysis_results[0].synthesis == "Analysis"
