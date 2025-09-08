"""
Result Converter for Strands Graph to ConversationResult.

This module provides utilities to convert Strands MultiAgentResult back to
ConversationResult format for backward compatibility with existing code.
"""

from datetime import datetime
from typing import Any

from strands.agent.agent_result import AgentResult

from strands.multiagent.base import MultiAgentResult, Status

from .analysis_agent import AnalysisResult
from .context_agent import ContextResult
from .data_models import ConversationResult, ReviewResult


class ResultConverter:
    """Converts Strands MultiAgentResult to ConversationResult format."""

    def __init__(self) -> None:
        """Initialize the result converter."""
        pass

    def convert_to_conversation_result(
        self,
        graph_result: MultiAgentResult,
        original_content: str | None = None,
    ) -> ConversationResult:
        """Convert MultiAgentResult to ConversationResult.

        Args:
            graph_result: Result from Strands graph execution
            original_content: Original content that was processed (if available)

        Returns:
            ConversationResult compatible with existing code
        """

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

    def _extract_reviews(self, graph_result: MultiAgentResult) -> list[ReviewResult]:
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
                # Check if result is MultiAgentResult
                if not isinstance(node_result.result, MultiAgentResult):
                    continue

                if node_id not in node_result.result.results:
                    continue

                nested_result = node_result.result.results[node_id].result
                if not isinstance(nested_result, AgentResult):
                    continue

                agent_state = nested_result.state

                # Extract agent information
                agent_name = agent_state.get("agent_name", node_id)
                agent_role = agent_state.get("agent_role", "Reviewer")

                # Try to get clean response from message content first, fallback to state
                try:
                    if nested_result.message and nested_result.message["content"]:
                        response = nested_result.message["content"][0]["text"]

                        # Check if response is a JSON/dict string and extract clean text
                        if isinstance(response, str):
                            parsed_response = self._try_parse_json_response(response)
                            if parsed_response:
                                response = parsed_response
                    else:
                        response = agent_state.get("response", "")
                except Exception:
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
    ) -> list[ContextResult]:
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
                # Check if result is MultiAgentResult
                if not isinstance(node_result.result, MultiAgentResult):
                    continue

                if node_id not in node_result.result.results:
                    continue

                nested_result = node_result.result.results[node_id].result
                if not isinstance(nested_result, AgentResult):
                    continue

                agent_state = nested_result.state

                # Try to get clean response from message content first, fallback to state
                try:
                    if nested_result.message and nested_result.message["content"]:
                        response = nested_result.message["content"][0]["text"]

                        # Check if response is a JSON/dict string and extract clean text
                        if isinstance(response, str):
                            parsed_response = self._try_parse_json_response(response)
                            if parsed_response:
                                response = parsed_response
                    else:
                        response = agent_state.get("response", "")
                except Exception:
                    response = agent_state.get("response", "")

                # Try to extract ContextResult from clean response
                if response:
                    context_result = self._parse_context_response(response)
                    if context_result:
                        context_results.append(context_result)

        return context_results

    def _extract_analysis_results(
        self, graph_result: MultiAgentResult
    ) -> list[AnalysisResult]:
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
                # Check if result is MultiAgentResult
                if not isinstance(node_result.result, MultiAgentResult):
                    continue

                if node_id not in node_result.result.results:
                    continue

                nested_result = node_result.result.results[node_id].result
                if not isinstance(nested_result, AgentResult):
                    continue

                agent_state = nested_result.state

                # Try to get clean response from message content first, fallback to state
                try:
                    if nested_result.message and nested_result.message["content"]:
                        response = nested_result.message["content"][0]["text"]

                        # Check if response is a JSON/dict string and extract clean text
                        if isinstance(response, str):
                            parsed_response = self._try_parse_json_response(response)
                            if parsed_response:
                                response = parsed_response
                    else:
                        response = agent_state.get("response", "")
                except Exception:
                    response = agent_state.get("response", "")

                # Try to extract AnalysisResult from clean response
                if response:
                    analysis_result = self._parse_analysis_response(response)
                    if analysis_result:
                        analysis_results.append(analysis_result)

        return analysis_results

    def _extract_analysis_errors(self, graph_result: MultiAgentResult) -> list[str]:
        """Extract analysis errors from graph execution.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            List of error messages from analysis agents
        """
        errors = []

        for node_id, node_result in graph_result.results.items():
            if self._is_analysis_node(node_id):
                # Check if result is MultiAgentResult
                if not isinstance(node_result.result, MultiAgentResult):
                    continue

                if node_id not in node_result.result.results:
                    continue

                nested_result = node_result.result.results[node_id].result
                if not isinstance(nested_result, AgentResult):
                    continue

                agent_state = nested_result.state
                error = agent_state.get("error")
                if error:
                    errors.append(f"{node_id}: {error}")

        return errors

    def _determine_content(
        self, graph_result: MultiAgentResult, original_content: str | None
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
            # Check if result is MultiAgentResult
            if isinstance(doc_processor_result.result, MultiAgentResult):
                if "document_processor" in doc_processor_result.result.results:
                    nested_result = doc_processor_result.result.results[
                        "document_processor"
                    ].result
                    if isinstance(nested_result, AgentResult):
                        doc_state = nested_result.state
                        doc_processor_data = doc_state.get("document_processor_result")
                        if doc_processor_data and hasattr(
                            doc_processor_data, "compiled_content"
                        ):
                            return str(doc_processor_data.compiled_content)

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

    def _try_parse_json_response(self, response: str) -> str | None:
        """
        Try to parse a response string as JSON/dict and extract clean text.
        Returns the extracted text if successful, None if not a JSON structure.
        """
        if not response.strip():
            return None

        # Try to parse as JSON/dict structure
        try:
            import ast
            import json

            parsed = None

            # First try json.loads (for proper JSON format)
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval for Python dict format
                try:
                    parsed = ast.literal_eval(response)
                except (ValueError, SyntaxError):
                    # Not a valid JSON/dict structure
                    return None

            # Check if it has the expected structure and extract clean text
            if isinstance(parsed, dict) and "content" in parsed:
                nested_content = parsed["content"]
                if isinstance(nested_content, list) and len(nested_content) > 0:
                    if (
                        isinstance(nested_content[0], dict)
                        and "text" in nested_content[0]
                    ):
                        return str(nested_content[0]["text"])

            # If structure doesn't match, return None (not a JSON response we care about)
            return None

        except Exception:
            # If any parsing fails, it's not a JSON structure we can handle
            return None

    def _parse_context_response(self, response: str) -> ContextResult | None:
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

    def _parse_analysis_response(self, response: str) -> AnalysisResult | None:
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

    def get_execution_summary(self, graph_result: MultiAgentResult) -> dict[str, Any]:
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
    ) -> dict[str, Any] | None:
        """Extract document processing information from graph result.

        Args:
            graph_result: Result from Strands graph execution

        Returns:
            Dictionary with document information or None
        """
        doc_processor_result = graph_result.results.get("document_processor")
        if not doc_processor_result:
            return None

        # Check if result is MultiAgentResult
        if not isinstance(doc_processor_result.result, MultiAgentResult):
            return None

        if "document_processor" not in doc_processor_result.result.results:
            return None

        nested_result = doc_processor_result.result.results["document_processor"].result
        if not isinstance(nested_result, AgentResult):
            return None

        doc_state = nested_result.state
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
    original_content: str | None = None,
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
