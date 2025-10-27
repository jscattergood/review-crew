"""Tests for ResultConverter class."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from strands.agent.agent_result import AgentResult
from strands.multiagent.base import MultiAgentResult, NodeResult, Status
from strands.types.content import ContentBlock, Message

from src.agents.analysis_agent import AnalysisResult
from src.agents.context_agent import ContextResult
from src.agents.data_models import ConversationResult, ReviewResult
from src.agents.result_converter import (
    ResultConverter,
    convert_graph_result_to_conversation,
)


class TestResultConverter:
    """Test cases for ResultConverter."""

    @pytest.fixture
    def converter(self):
        """Create a ResultConverter instance."""
        return ResultConverter()

    @pytest.fixture
    def simple_graph_result(self):
        """Create a simple graph result for testing."""
        # Create a review agent result
        review_agent_result = AgentResult(
            stop_reason="end_turn",
            message=Message(
                role="assistant", content=[ContentBlock(text="Great review feedback")]
            ),
            metrics=Mock(),
            state={
                "agent_name": "Test Reviewer",
                "agent_role": "Content Reviewer",
                "response": "Great review feedback",
            },
        )

        # Create the nested structure: node_result.result.results[node_id].result
        review_multi_result = MultiAgentResult(
            results={
                "test_reviewer": NodeResult(
                    result=review_agent_result, status=Status.COMPLETED
                )
            },
            execution_time=0.2,
            execution_count=1,
        )

        review_node_result = NodeResult(
            result=review_multi_result, status=Status.COMPLETED
        )

        # Create the main graph result
        graph_result = MultiAgentResult(
            results={"test_reviewer": review_node_result},
            execution_time=0.5,
            execution_count=1,
        )

        return graph_result

    def test_init(self, converter):
        """Test ResultConverter initialization."""
        assert converter is not None

    def test_convert_simple_result(self, converter, simple_graph_result):
        """Test converting a simple graph result."""
        result = converter.convert_to_conversation_result(simple_graph_result)

        assert isinstance(result, ConversationResult)
        assert len(result.reviews) == 1
        assert result.reviews[0].agent_name == "Test Reviewer"
        assert result.reviews[0].feedback == "Great review feedback"

    def test_extract_reviews(self, converter, simple_graph_result):
        """Test extracting reviews from graph result."""
        reviews = converter._extract_reviews(simple_graph_result)

        assert len(reviews) == 1
        review = reviews[0]
        assert review.agent_name == "Test Reviewer"
        assert review.agent_role == "Content Reviewer"
        assert review.feedback == "Great review feedback"
        assert review.error is None

    def test_is_review_node(self, converter):
        """Test review node identification."""
        assert converter._is_review_node("test_reviewer") == True
        assert converter._is_review_node("some_agent") == True
        assert converter._is_review_node("document_processor") == False
        assert converter._is_context_node("contextualizer_test") == True
        assert converter._is_analysis_node("analyzer_test") == True

    def test_empty_graph_result(self, converter):
        """Test handling empty graph result."""
        empty_result = MultiAgentResult(
            results={}, execution_time=0.0, execution_count=0
        )

        result = converter.convert_to_conversation_result(empty_result)
        assert isinstance(result, ConversationResult)
        assert len(result.reviews) == 0
        assert len(result.context_results) == 0
        assert len(result.analysis_results) == 0

    def test_convenience_function(self, simple_graph_result):
        """Test the convenience function."""
        result = convert_graph_result_to_conversation(
            simple_graph_result, "original content"
        )

        assert isinstance(result, ConversationResult)
        assert result.original_content == "original content"
        assert len(result.reviews) == 1

    def test_json_parsing_helper(self, converter):
        """Test the JSON parsing helper method."""
        # Test with valid JSON string
        json_string = '{"role": "assistant", "content": [{"text": "Clean text here"}]}'
        result = converter._try_parse_json_response(json_string)
        assert result == "Clean text here"

        # Test with Python dict string
        dict_string = "{'role': 'assistant', 'content': [{'text': 'Clean text here'}]}"
        result = converter._try_parse_json_response(dict_string)
        assert result == "Clean text here"

        # Test with non-JSON string
        regular_string = "This is just regular text"
        result = converter._try_parse_json_response(regular_string)
        assert result is None

        # Test with empty string
        result = converter._try_parse_json_response("")
        assert result is None

    def test_error_handling(self, converter):
        """Test error handling with empty graph result."""
        # Create an empty result (more realistic than None)
        empty_result = MultiAgentResult(
            results={},  # Empty results
            execution_time=0.0,
            execution_count=0,
        )

        # Should not crash, should return empty results
        result = converter.convert_to_conversation_result(empty_result)
        assert isinstance(result, ConversationResult)
        assert len(result.reviews) == 0
        assert len(result.context_results) == 0
        assert len(result.analysis_results) == 0
