"""
Result Converter for Strands Graph to ConversationResult.

This module provides utilities to convert Strands MultiAgentResult back to
ConversationResult format for backward compatibility with existing code.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from strands.multiagent.base import MultiAgentResult, Status

from .conversation_manager import ConversationResult, ReviewResult
from .analysis_agent import AnalysisResult
from .context_agent import ContextResult
from .document_processor_node import DocumentProcessorResult


class ResultConverter:
    """Converts Strands MultiAgentResult to ConversationResult format."""

    def __init__(self):
        """Initialize the result converter."""
        pass

    def convert_to_conversation_result(
        self,
        graph_result: MultiAgentResult,
        original_content: Optional[str] = None,
    ) -> ConversationResult:
        """Convert MultiAgentResult to ConversationResult.

        Args:
            graph_result: Result from Strands graph execution
            original_content: Original content that was processed (if available)

        Returns:
            ConversationResult compatible with existing code
        """
        # Extract components from graph result
        reviews = self._extract_reviews(graph_result)
        context_results = self._extract_context_results(graph_result)
        analysis_results = self._extract_analysis_results(graph_result)
        analysis_errors = self._extract_analysis_errors(graph_result)

        # Determine content to use
        content = self._determine_content(graph_result, original_content)

        return ConversationResult(
            content=content,
            reviews=reviews,
            timestamp=datetime.now(),
            summary=None,  # Could be generated from analysis results if needed
            analysis_results=analysis_results,
            context_results=context_results,
            original_content=original_content,
            analysis_errors=analysis_errors,
        )

    def _extract_reviews(self, graph_result: MultiAgentResult) -> List[ReviewResult]:
        """Extract review results from graph execution.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            List of ReviewResult objects
        """
        reviews = []

        for node_id, node_result in graph_result.results.items():
            # Skip non-review nodes
            if self._is_review_node(node_id):
                agent_state = node_result.result.state

                # Extract agent information
                agent_name = agent_state.get("agent_name", node_id)
                agent_role = agent_state.get("agent_role", "Reviewer")
                response = agent_state.get("response", "")
                error = agent_state.get("error")

                # Create ReviewResult
                review = ReviewResult(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    feedback=response,
                    timestamp=datetime.now(),
                    error=error,
                )
                reviews.append(review)

        return reviews

    def _extract_context_results(
        self, graph_result: MultiAgentResult
    ) -> List[ContextResult]:
        """Extract context results from graph execution.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            List of ContextResult objects
        """
        context_results = []

        for node_id, node_result in graph_result.results.items():
            # Skip non-context nodes
            if self._is_context_node(node_id):
                agent_state = node_result.result.state

                # Try to extract ContextResult from state
                if "response" in agent_state:
                    # Parse the response to extract context information
                    response = agent_state["response"]
                    context_result = self._parse_context_response(response)
                    if context_result:
                        context_results.append(context_result)

        return context_results

    def _extract_analysis_results(
        self, graph_result: MultiAgentResult
    ) -> List[AnalysisResult]:
        """Extract analysis results from graph execution.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            List of AnalysisResult objects
        """
        analysis_results = []

        for node_id, node_result in graph_result.results.items():
            # Skip non-analysis nodes
            if self._is_analysis_node(node_id):
                agent_state = node_result.result.state

                # Try to extract AnalysisResult from state
                if "response" in agent_state:
                    response = agent_state["response"]
                    analysis_result = self._parse_analysis_response(response)
                    if analysis_result:
                        analysis_results.append(analysis_result)

        return analysis_results

    def _extract_analysis_errors(self, graph_result: MultiAgentResult) -> List[str]:
        """Extract analysis errors from graph execution.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            List of error messages from analysis agents
        """
        errors = []

        for node_id, node_result in graph_result.results.items():
            if self._is_analysis_node(node_id):
                agent_state = node_result.result.state
                error = agent_state.get("error")
                if error:
                    errors.append(f"{node_id}: {error}")

        return errors

    def _determine_content(
        self, graph_result: MultiAgentResult, original_content: Optional[str]
    ) -> str:
        """Determine the content field for ConversationResult.

        Args:
            graph_result: Result from Strands graph execution
            original_content: Original content if available

        Returns:
            Content string for ConversationResult
        """
        # First try to get compiled content from document processor
        doc_processor_result = graph_result.results.get("document_processor")
        if doc_processor_result:
            doc_state = doc_processor_result.result.state
            doc_processor_data = doc_state.get("document_processor_result")
            if doc_processor_data and hasattr(doc_processor_data, "compiled_content"):
                return doc_processor_data.compiled_content

        # Fallback to original content or empty string
        return original_content or ""

    def _is_review_node(self, node_id: str) -> bool:
        """Check if a node is a review agent.

        Args:
            node_id: Node identifier

        Returns:
            True if this is a review agent node
        """
        # Review agents typically have names that don't start with special prefixes
        return not (
            node_id.startswith("document_processor")
            or node_id.startswith("contextualizer_")
            or node_id.startswith("analyzer_")
            or self._is_context_node(node_id)
            or self._is_analysis_node(node_id)
        )

    def _is_context_node(self, node_id: str) -> bool:
        """Check if a node is a context agent.

        Args:
            node_id: Node identifier

        Returns:
            True if this is a context agent node
        """
        return (
            node_id.startswith("contextualizer_")
            or "contextualizer" in node_id.lower()
            or "context" in node_id.lower()
        )

    def _is_analysis_node(self, node_id: str) -> bool:
        """Check if a node is an analysis agent.

        Args:
            node_id: Node identifier

        Returns:
            True if this is an analysis agent node
        """
        return (
            node_id.startswith("analyzer_")
            or "analyzer" in node_id.lower()
            or "analysis" in node_id.lower()
        )

    def _parse_context_response(self, response: str) -> Optional[ContextResult]:
        """Parse context agent response into ContextResult.

        Args:
            response: Raw response from context agent

        Returns:
            ContextResult object or None if parsing fails
        """
        try:
            # For now, create a simple ContextResult
            # In the future, this could parse structured responses
            return ContextResult(
                formatted_context=response,
                context_summary=(
                    response[:200] + "..." if len(response) > 200 else response
                ),
            )
        except Exception:
            return None

    def _parse_analysis_response(self, response: str) -> Optional[AnalysisResult]:
        """Parse analysis agent response into AnalysisResult.

        Args:
            response: Raw response from analysis agent

        Returns:
            AnalysisResult object or None if parsing fails
        """
        try:
            # For now, create a simple AnalysisResult
            # In the future, this could parse structured responses
            return AnalysisResult(
                synthesis=response,
            )
        except Exception:
            return None

    def get_execution_summary(self, graph_result: MultiAgentResult) -> Dict[str, Any]:
        """Get a summary of graph execution results.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            Dictionary with execution summary
        """
        total_nodes = len(graph_result.results)
        successful_nodes = sum(
            1
            for result in graph_result.results.values()
            if result.status == Status.COMPLETED
        )
        failed_nodes = total_nodes - successful_nodes

        # Count by type
        review_nodes = sum(
            1
            for node_id in graph_result.results.keys()
            if self._is_review_node(node_id)
        )
        context_nodes = sum(
            1
            for node_id in graph_result.results.keys()
            if self._is_context_node(node_id)
        )
        analysis_nodes = sum(
            1
            for node_id in graph_result.results.keys()
            if self._is_analysis_node(node_id)
        )

        return {
            "total_nodes": total_nodes,
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
            "review_agents": review_nodes,
            "context_agents": context_nodes,
            "analysis_agents": analysis_nodes,
            "overall_status": graph_result.status.value,
            "execution_time": getattr(graph_result, "execution_time", 0),
        }

    def extract_document_info(
        self, graph_result: MultiAgentResult
    ) -> Optional[Dict[str, Any]]:
        """Extract document processing information from graph result.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            Dictionary with document information or None
        """
        doc_processor_result = graph_result.results.get("document_processor")
        if not doc_processor_result:
            return None

        doc_state = doc_processor_result.result.state
        doc_processor_data = doc_state.get("document_processor_result")

        if not doc_processor_data:
            return None

        return {
            "document_type": getattr(doc_processor_data, "document_type", "unknown"),
            "document_count": len(getattr(doc_processor_data, "documents", [])),
            "has_manifest": getattr(doc_processor_data, "manifest_config", None)
            is not None,
            "validation_results": getattr(
                doc_processor_data, "validation_results", None
            ),
            "original_path": getattr(doc_processor_data, "original_path", None),
        }


# Convenience function for easy usage
def convert_graph_result_to_conversation(
    graph_result: MultiAgentResult,
    original_content: Optional[str] = None,
) -> ConversationResult:
    """Convert a Strands graph result to ConversationResult format.

    Args:
        graph_result: Result from Strands graph execution
        original_content: Original content that was processed

    Returns:
        ConversationResult compatible with existing code
    """
    converter = ResultConverter()
    return converter.convert_to_conversation_result(graph_result, original_content)
