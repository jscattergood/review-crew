"""Tests for ConversationManager.format_results method."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.agents.analysis_agent import AnalysisResult
from src.agents.context_agent import ContextResult
from src.agents.data_models import ConversationResult, ReviewResult
from src.conversation.manager import ConversationManager


class TestFormatResults:
    """Test cases for format_results method."""

    @pytest.fixture
    def mock_persona_loader(self):
        """Create a mock persona loader."""
        mock_loader = Mock()
        mock_loader.load_reviewers.return_value = []
        mock_loader.load_contextualizers.return_value = []
        mock_loader.load_analyzers.return_value = []
        return mock_loader

    @pytest.fixture
    def manager(self, mock_persona_loader):
        """Create a ConversationManager with mocked dependencies."""
        with patch("src.conversation.manager.AnalysisAgent"):
            return ConversationManager(persona_loader=mock_persona_loader)

    def test_format_results_basic(self, manager):
        """Test basic result formatting."""
        result = ConversationResult(
            content="Test content",
            reviews=[
                ReviewResult(
                    agent_name="Test Agent",
                    agent_role="Tester",
                    feedback="Test feedback",
                    timestamp=datetime.now(),
                )
            ],
            timestamp=datetime.now(),
        )

        formatted = manager.format_results(result)

        assert "## Content" in formatted
        assert "Test content" in formatted
        assert "## Reviews" in formatted
        assert "### Test Agent" in formatted
        assert "*Tester*" in formatted
        assert "Test feedback" in formatted

    def test_format_results_no_content(self, manager):
        """Test result formatting without content."""
        result = ConversationResult(
            content="Test content",
            reviews=[
                ReviewResult(
                    agent_name="Test Agent",
                    agent_role="Tester",
                    feedback="Test feedback",
                    timestamp=datetime.now(),
                )
            ],
            timestamp=datetime.now(),
        )

        formatted = manager.format_results(result, include_content=False)

        assert "## Content" not in formatted
        assert "Test content" not in formatted
        assert "## Reviews" in formatted
        assert "### Test Agent" in formatted
        assert "*Tester*" in formatted
        assert "Test feedback" in formatted

    def test_format_results_with_context(self, manager):
        """Test result formatting with context."""
        result = ConversationResult(
            content="Test content",
            reviews=[
                ReviewResult(
                    agent_name="Test Agent",
                    agent_role="Tester",
                    feedback="Test feedback",
                    timestamp=datetime.now(),
                )
            ],
            context_results=[
                ContextResult(
                    formatted_context="Test context",
                    context_summary="Context summary",
                    timestamp=datetime.now(),
                )
            ],
            timestamp=datetime.now(),
        )

        formatted = manager.format_results(result, include_context=True)

        assert "## Content" in formatted
        assert "## Reviews" in formatted
        assert "## Context" in formatted
        assert "Test context" in formatted

    def test_format_results_with_analysis(self, manager):
        """Test result formatting with analysis."""
        result = ConversationResult(
            content="Test content",
            reviews=[
                ReviewResult(
                    agent_name="Test Agent",
                    agent_role="Tester",
                    feedback="Test feedback",
                    timestamp=datetime.now(),
                )
            ],
            analysis_results=[
                AnalysisResult(
                    synthesis="Test analysis synthesis",
                    key_themes=["Theme 1", "Theme 2"],
                    priority_recommendations=["Rec 1", "Rec 2"],
                    timestamp=datetime.now(),
                )
            ],
            timestamp=datetime.now(),
        )

        formatted = manager.format_results(result)

        assert "## Content" in formatted
        assert "## Reviews" in formatted
        assert "## Analysis" in formatted
        assert "### Meta-Analysis Summary" in formatted
        assert "Test analysis synthesis" in formatted
        assert "### Key Themes" in formatted
        assert "• Theme 1" in formatted
        assert "• Theme 2" in formatted
        assert "### Priority Recommendations" in formatted
        assert "• Rec 1" in formatted
        assert "• Rec 2" in formatted

    def test_format_results_with_errors(self, manager):
        """Test result formatting with failed reviews."""
        result = ConversationResult(
            content="Test content",
            reviews=[
                ReviewResult(
                    agent_name="Good Agent",
                    agent_role="Tester",
                    feedback="Good feedback",
                    timestamp=datetime.now(),
                ),
                ReviewResult(
                    agent_name="Bad Agent",
                    agent_role="Tester",
                    feedback="",
                    error="Something went wrong",
                    timestamp=datetime.now(),
                ),
            ],
            analysis_errors=["Analysis failed"],
            timestamp=datetime.now(),
        )

        formatted = manager.format_results(result)

        assert "## Content" in formatted
        assert "## Reviews" in formatted
        assert "### Good Agent" in formatted
        assert "*Tester*" in formatted
        assert "Good feedback" in formatted
        assert "### Bad Agent" in formatted
        assert "❌ **Error:** Something went wrong" in formatted
        assert "## Analysis Errors" in formatted
        assert "❌ Analysis failed" in formatted

    def test_format_results_empty(self, manager):
        """Test result formatting with empty results."""
        result = ConversationResult(content="", reviews=[], timestamp=datetime.now())

        formatted = manager.format_results(result)

        # Should handle empty results gracefully
        assert isinstance(formatted, str)
        # With no content and no reviews, should be minimal output
        assert "## Content" not in formatted or formatted.count("## Content") == 1
