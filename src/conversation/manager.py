"""
Conversation Manager for orchestrating multi-agent reviews.

This module manages conversations between multiple review agents,
collecting and organizing their feedback using Strands Graph architecture.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..agents.analysis_agent import AnalysisAgent
from ..agents.context_agent import ContextAgent, ContextResult
from ..agents.data_models import ConversationResult
from ..agents.result_converter import ResultConverter
from ..agents.review_agent import ReviewAgent
from ..config.persona_loader import PersonaLoader
from ..logging.manager import LoggingManager
from .graph_builder import ReviewGraphBuilder


class ConversationManager:
    """Manages multi-agent review conversations using Strands Graph architecture."""

    def __init__(
        self,
        persona_loader: PersonaLoader | None = None,
        model_provider: str = "bedrock",
        model_config: dict[str, Any] | None = None,
        enable_analysis: bool = True,
    ):
        """Initialize the conversation manager.

        Args:
            persona_loader: Optional PersonaLoader instance
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
            enable_analysis: Whether to enable analysis of reviews using analyzer personas
        """
        self.persona_loader = persona_loader or PersonaLoader()
        self.model_provider = model_provider
        self.model_config = model_config or {}
        self.enable_analysis = enable_analysis

        # Initialize graph builder and result converter
        self.graph_builder = ReviewGraphBuilder(
            persona_loader=self.persona_loader,
            model_provider=self.model_provider,
            model_config=self.model_config,
            enable_analysis=self.enable_analysis,
        )
        self.result_converter = ResultConverter()

        # Keep legacy agent lists for backward compatibility
        self.agents: list[ReviewAgent] = self.graph_builder.review_agents
        self.context_agents: list[ContextAgent] = self.graph_builder.context_agents
        self.analysis_agents: list[AnalysisAgent] = self.graph_builder.analysis_agents

        # Get logging manager instance
        self.logging_manager = LoggingManager.get_instance()

    def get_available_agents(self) -> list[dict[str, Any]]:
        """Get information about available agents.

        Returns:
            List of agent information dictionaries
        """
        return [agent.get_info() for agent in self.agents]

    def get_available_contextualizers(self) -> list[dict[str, Any]]:
        """Get information about available contextualizer agents.

        Returns:
            List of contextualizer agent information dictionaries
        """
        return [agent.get_info() for agent in self.context_agents]

    def get_available_analyzers(self) -> list[dict[str, Any]]:
        """Get information about available analyzer agents.

        Returns:
            List of analyzer agent information dictionaries
        """
        return [agent.get_info() for agent in self.analysis_agents]

    async def run_review(
        self,
        content: str,
        context_data: str | None = None,
        selected_agents: list[str] | None = None,
    ) -> ConversationResult:
        """Run a review with selected agents using graph-based execution.

        Args:
            content: Content to review or directory path for multi-document review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        # Log the start of the review process
        self.logging_manager.log_conversation_event(
            "Starting review process",
            {
                "content_type": "file/directory"
                if Path(content).exists()
                else "direct_content",
                "content_preview": content[:100] + "..."
                if len(content) > 100
                else content,
                "has_context": context_data is not None,
                "selected_agents": selected_agents,
                "model_provider": self.model_provider,
            },
        )

        try:
            # Check if content is a directory path for multi-document review
            content_path = Path(content)
            if content_path.exists() and content_path.is_dir():
                self.logging_manager.log_conversation_event(
                    f"Processing directory: {content_path}"
                )
                result = await self._run_graph_based_review(
                    content_path, selected_agents
                )
            elif content_path.exists() and content_path.is_file():
                self.logging_manager.log_conversation_event(
                    f"Processing file: {content_path}"
                )
                result = await self._run_graph_based_review(
                    content_path, selected_agents
                )
            else:
                self.logging_manager.log_conversation_event("Processing direct content")
                # Direct content - use simple graph
                result = await self._run_simple_graph_review(content, selected_agents)

            # Log completion
            self.logging_manager.log_conversation_event(
                "Review process completed",
                {
                    "review_count": len(result.reviews),
                    "analysis_count": len(result.analysis_results)
                    if result.analysis_results
                    else 0,
                    "context_count": len(result.context_results)
                    if result.context_results
                    else 0,
                    "errors": len(result.analysis_errors)
                    if result.analysis_errors
                    else 0,
                },
            )

            return result

        except Exception as e:
            self.logging_manager.log_conversation_event(
                f"Review process failed: {str(e)}"
            )
            raise

    # ========================================
    # Graph-Based Execution Methods
    # ========================================

    async def _run_graph_based_review(
        self, content_path: Path, selected_agents: list[str] | None = None
    ) -> ConversationResult:
        """Run review using graph-based execution.

        Args:
            content_path: Path to file or directory to review
            selected_agents: Optional list of agent names to use

        Returns:
            ConversationResult with all reviews and analysis
        """
        try:
            # Build appropriate graph based on content type
            if content_path.is_dir():
                # Check for manifest
                manifest_path = content_path / "manifest.yaml"
                if manifest_path.exists():
                    manifest_config = self._load_manifest(manifest_path)
                    graph = self.graph_builder.build_manifest_driven_graph(
                        manifest_config, content_path
                    )
                else:
                    graph = self.graph_builder.build_standard_review_graph(
                        selected_reviewers=selected_agents
                    )
            else:
                # Single file
                graph = self.graph_builder.build_standard_review_graph(
                    selected_reviewers=selected_agents
                )

            # Execute graph asynchronously
            graph_result = await self.graph_builder.execute_graph(
                graph, str(content_path)
            )

            # Convert result back to ConversationResult
            return self.result_converter.convert_to_conversation_result(
                graph_result, original_content=str(content_path)
            )

        except Exception as e:
            print(f"❌ Async graph-based review failed: {e}")
            # Fallback to legacy implementation
            return await self._fallback_to_legacy_review(content_path, selected_agents)

    async def _run_simple_graph_review(
        self, content: str, selected_agents: list[str] | None = None
    ) -> ConversationResult:
        """Run review using simple graph for direct content input.

        Args:
            content: Direct content to review
            selected_agents: Optional list of agent names to use

        Returns:
            ConversationResult with all reviews and analysis
        """
        try:
            # Check if content looks like an error case (e.g., path that doesn't exist)
            if self._is_error_content(content):
                return ConversationResult(
                    content=content,
                    reviews=[],
                    timestamp=datetime.now(),
                    analysis_errors=["No valid content provided for review"],
                    input_source="Direct input",
                    manifest_path=None,
                )

            # Build simple graph for direct content
            graph = self.graph_builder.build_simple_review_graph(content)

            # Execute graph asynchronously
            graph_result = await self.graph_builder.execute_graph(graph, content)

            # Convert result back to ConversationResult
            return self.result_converter.convert_to_conversation_result(
                graph_result, original_content=content
            )

        except Exception as e:
            print(f"❌ Async simple graph review failed: {e}")
            # Fallback to legacy implementation
            return await self._fallback_to_legacy_simple_review(
                content, selected_agents
            )

    # ========================================
    # Legacy Support Methods
    # ========================================

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load and parse manifest file.

        Args:
            manifest_path: Path to manifest.yaml file

        Returns:
            Parsed manifest configuration dictionary
        """
        import yaml

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            return manifest or {}
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse manifest {manifest_path}: {e}")
            return {}

    # ========================================
    # Fallback Methods (Legacy Implementation)
    # ========================================

    async def _fallback_to_legacy_review(
        self, content_path: Path, selected_agents: list[str] | None = None
    ) -> ConversationResult:
        """Fallback to legacy review implementation."""
        print("🔄 Falling back to legacy review implementation")
        # For now, return a basic result - in a real implementation,
        # this would call the original legacy methods
        return ConversationResult(
            content=f"Legacy fallback for {content_path}",
            reviews=[],
            timestamp=datetime.now(),
            analysis_errors=[
                "Graph-based execution failed, legacy fallback not fully implemented"
            ],
            input_source=str(content_path),
            manifest_path=None,
        )

    async def _fallback_to_legacy_simple_review(
        self, content: str, selected_agents: list[str] | None = None
    ) -> ConversationResult:
        """Fallback to legacy simple review implementation."""
        print("🔄 Falling back to legacy simple review implementation")
        # For now, return a basic result - in a real implementation,
        # this would call the original legacy methods
        return ConversationResult(
            content=content,
            reviews=[],
            timestamp=datetime.now(),
            analysis_errors=[
                "Graph-based execution failed, legacy fallback not fully implemented"
            ],
            input_source="Direct input",
            manifest_path=None,
        )

    # ========================================
    # Legacy Methods (for backward compatibility)
    # ========================================

    def _load_agents(self) -> None:
        """Load all reviewer personas as review agents."""
        try:
            personas = self.persona_loader.load_reviewer_personas()
            self.agents = [
                ReviewAgent(
                    persona,
                    model_provider=self.model_provider,
                    model_config_override=self.model_config,
                )
                for persona in personas
            ]
            print(
                f"✅ Loaded {len(self.agents)} review agents with {self.model_provider} provider"
            )
        except Exception as e:
            print(f"❌ Error loading agents: {e}")
            self.agents = []

    def _load_contextualizers(self) -> None:
        """Load all contextualizer personas as context agents."""
        try:
            contextualizer_personas = self.persona_loader.load_contextualizer_personas()

            if contextualizer_personas:
                print(
                    f"✅ Loaded {len(contextualizer_personas)} contextualizer personas"
                )

                # Create context agents for all contextualizer personas
                self.context_agents = []
                for persona in contextualizer_personas:
                    context_agent = ContextAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.context_agents.append(context_agent)
                    print(f"✅ Created context agent: {persona.name}")
            else:
                print("ℹ️  No contextualizer personas found")
                self.context_agents = []

        except Exception as e:
            print(f"❌ Error loading contextualizers: {e}")
            self.context_agents = []

    def _load_analyzers(self) -> None:
        """Load all analyzer personas as analysis agents."""
        try:
            analyzer_personas = self.persona_loader.load_analyzer_personas()

            if analyzer_personas:
                print(f"✅ Loaded {len(analyzer_personas)} analyzer personas")

                # Create analysis agents for all analyzer personas
                self.analysis_agents = []
                for persona in analyzer_personas:
                    analysis_agent = AnalysisAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.analysis_agents.append(analysis_agent)
                    print(f"✅ Created analysis agent: {persona.name}")
            else:
                print("ℹ️  No analyzer personas found")
                self.analysis_agents = []

        except Exception as e:
            print(f"❌ Error loading analyzers: {e}")
            self.analysis_agents = []

    def _filter_agents(self, selected_agents: list[str] | None) -> list[ReviewAgent]:
        """Filter agents based on selected agent names."""
        if not selected_agents:
            return self.agents

        filtered_agents = []
        for agent_name in selected_agents:
            for agent in self.agents:
                if agent.persona.name.lower() == agent_name.lower():
                    filtered_agents.append(agent)
                    break

        return filtered_agents

    def _prepare_content_for_review(
        self, content: str, context_results: list[ContextResult]
    ) -> str:
        """Prepare content for review by adding context if available."""
        if not context_results:
            return content

        content_parts = [content]
        content_parts.append("\n\n## Context Information")
        content_parts.append("")

        for context_result in context_results:
            content_parts.append(context_result.formatted_context)

        return "\n".join(content_parts)

    def format_results(
        self,
        result: ConversationResult,
        include_content: bool = True,
        include_context: bool = True,
    ) -> str:
        """Format conversation results for display."""
        output_parts = []

        if include_content and result.content:
            output_parts.append("## Content")
            output_parts.append("")
            # Add timestamp at the top of content section
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_parts.append(f"*Generated: {timestamp}*")

            # Add input source information
            if result.input_source:
                if result.input_source == "Direct input":
                    output_parts.append(f"*Source: {result.input_source}*")
                else:
                    output_parts.append(f"*Source: {result.input_source}*")
                    if result.manifest_path:
                        output_parts.append(f"*Manifest: {result.manifest_path}*")

            output_parts.append("")
            output_parts.append(result.content)
            output_parts.append("")

        if result.reviews:
            output_parts.append("## Reviews")
            output_parts.append("")

            # Add table of contents if there are multiple reviews
            if len(result.reviews) > 1:
                output_parts.append("### Table of Contents")
                for review in result.reviews:
                    # Create anchor link from agent name
                    anchor = self._create_anchor_link(review.agent_name)
                    output_parts.append(f"- [{review.agent_name}](#{anchor})")
                output_parts.append("")
                output_parts.append("***")
                output_parts.append("")

            # Add reviews with separators
            for i, review in enumerate(result.reviews):
                # Add separator before each review except the first
                if i > 0:
                    output_parts.append("***")
                    output_parts.append("")

                output_parts.append(f"### {review.agent_name}")
                output_parts.append(f"*{review.agent_role}*")
                output_parts.append("")
                if review.error:
                    output_parts.append(f"❌ **Error:** {review.error}")
                else:
                    # Clean up any raw JSON that might be in the feedback
                    clean_feedback = self._clean_raw_json(review.feedback)
                    output_parts.append(clean_feedback)
                output_parts.append("")

        if include_context and result.context_results:
            output_parts.append("## Context")
            output_parts.append("")
            for context in result.context_results:
                output_parts.append(context.formatted_context)
                output_parts.append("")

        if result.analysis_results:
            output_parts.append("## Analysis")
            output_parts.append("")
            for analysis in result.analysis_results:
                output_parts.append("### Meta-Analysis Summary")
                output_parts.append(analysis.synthesis)

                if analysis.personal_statement_summary:
                    output_parts.append("")
                    output_parts.append("### Personal Statement Summary")
                    output_parts.append(analysis.personal_statement_summary)

                if analysis.key_themes:
                    output_parts.append("")
                    output_parts.append("### Key Themes")
                    for theme in analysis.key_themes:
                        output_parts.append(f"• {theme}")

                if analysis.priority_recommendations:
                    output_parts.append("")
                    output_parts.append("### Priority Recommendations")
                    for rec in analysis.priority_recommendations:
                        output_parts.append(f"• {rec}")

                output_parts.append("")

        if result.analysis_errors:
            output_parts.append("## Analysis Errors")
            output_parts.append("")
            for error in result.analysis_errors:
                output_parts.append(f"❌ {error}")
            output_parts.append("")

        return "\n".join(output_parts)

    def _is_error_content(self, content: str) -> bool:
        """Check if content appears to be an error case (like a non-existent path).

        Args:
            content: Content to check

        Returns:
            True if content appears to be an error case
        """
        content_stripped = content.strip()

        # Check if it looks like a path that doesn't exist
        if content_stripped.startswith("input/") and len(content_stripped) < 50:
            return True

        # Check for other error indicators
        error_indicators = [
            "input/nonexistent",
            "ERROR_NO_CONTENT",
        ]

        return any(indicator in content_stripped for indicator in error_indicators)

    def _clean_raw_json(self, text: str) -> str:
        """Clean up any raw JSON structures that might appear in agent responses.

        Args:
            text: The text that might contain raw JSON

        Returns:
            Clean text with JSON structures extracted
        """
        if not text or not isinstance(text, str):
            return text

        # Check if the text starts with a JSON structure
        text_stripped = text.strip()
        if text_stripped.startswith("{'role':") or text_stripped.startswith('{"role":'):
            try:
                # Try to parse the JSON structure and extract the actual text
                import ast
                import json

                # Try ast.literal_eval first (safer)
                try:
                    parsed = ast.literal_eval(text_stripped.replace("'", '"'))
                except:
                    # Fallback to json.loads
                    parsed = json.loads(text_stripped.replace("'", '"'))

                if isinstance(parsed, dict) and "content" in parsed:
                    nested_content = parsed["content"]
                    if isinstance(nested_content, list) and len(nested_content) > 0:
                        if (
                            isinstance(nested_content[0], dict)
                            and "text" in nested_content[0]
                        ):
                            return nested_content[0]["text"] or ""
            except:
                # If parsing fails, return the original text
                pass

        return text

    def _create_anchor_link(self, agent_name: str) -> str:
        """Create a markdown anchor link from an agent name.

        Args:
            agent_name: The agent name to convert to an anchor

        Returns:
            A markdown-compatible anchor link
        """
        # Convert to lowercase and replace spaces and special characters with hyphens
        anchor = agent_name.lower()
        anchor = anchor.replace(" ", "-")
        anchor = anchor.replace("(", "")
        anchor = anchor.replace(")", "")
        anchor = anchor.replace("&", "")
        anchor = anchor.replace(",", "")
        anchor = anchor.replace(".", "")
        anchor = anchor.replace("'", "")
        anchor = anchor.replace('"', "")
        # Remove multiple consecutive hyphens
        while "--" in anchor:
            anchor = anchor.replace("--", "-")
        # Remove leading/trailing hyphens
        anchor = anchor.strip("-")
        return anchor
