"""
Review Graph Builder for Strands Graph Architecture.

This module provides the ReviewGraphBuilder class that constructs Strands graphs
using our refactored agents that directly inherit from MultiAgentBase.
"""

from pathlib import Path
from typing import Any

from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentResult
from strands.multiagent.graph import Graph

from ..agents.analysis_agent import AnalysisAgent
from ..agents.context_agent import ContextAgent
from ..agents.review_agent import ReviewAgent
from ..config.persona_loader import PersonaConfig, PersonaLoader
from .document_processor import DocumentProcessorNode


class ReviewGraphBuilder:
    """Builder class for constructing review graphs using Strands architecture.

    This class creates graphs where agents can run in parallel, replacing the
    sequential orchestration in ConversationManager.
    """

    def __init__(
        self,
        persona_loader: PersonaLoader | None = None,
        model_provider: str = "bedrock",
        model_config: dict[str, Any] | None = None,
        enable_analysis: bool = True,
    ):
        """Initialize the review graph builder.

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

        # Load available agents
        self.review_agents = self._load_review_agents()
        self.context_agents = self._load_context_agents()
        self.analysis_agents = self._load_analysis_agents() if enable_analysis else []

    def _load_review_agents(self) -> list[ReviewAgent]:
        """Load all available review agents."""
        try:
            personas = self.persona_loader.load_reviewer_personas()
            agents = [
                ReviewAgent(
                    persona,
                    model_provider=self.model_provider,
                    model_config_override=self.model_config,
                )
                for persona in personas
            ]
            print(f"✅ Loaded {len(agents)} review agents")
            return agents
        except Exception as e:
            print(f"❌ Error loading review agents: {e}")
            return []

    def _load_context_agents(self) -> list[ContextAgent]:
        """Load all available context agents."""
        try:
            personas = self.persona_loader.load_contextualizer_personas()
            agents = [
                ContextAgent(
                    persona=persona,
                    model_provider=self.model_provider,
                    model_config=self.model_config,
                )
                for persona in personas
            ]
            print(f"✅ Loaded {len(agents)} context agents")
            return agents
        except Exception as e:
            print(f"❌ Error loading context agents: {e}")
            return []

    def _load_analysis_agents(self) -> list[AnalysisAgent]:
        """Load all available analysis agents."""
        try:
            personas = self.persona_loader.load_analyzer_personas()
            agents = [
                AnalysisAgent(
                    persona=persona,
                    model_provider=self.model_provider,
                    model_config=self.model_config,
                )
                for persona in personas
            ]
            print(f"✅ Loaded {len(agents)} analysis agents")
            return agents
        except Exception as e:
            print(f"❌ Error loading analysis agents: {e}")
            return []

    def build_standard_review_graph(
        self,
        selected_reviewers: list[str] | None = None,
        selected_contextualizers: list[str] | None = None,
        selected_analyzers: list[str] | None = None,
    ) -> Graph:
        """Build a standard review graph with document processing, context, reviews, and analysis.

        This creates a graph where:
        1. DocumentProcessor runs first
        2. Context agents run in parallel (depend on DocumentProcessor)
        3. Review agents run in parallel (depend on DocumentProcessor + Context agents)
        4. Analysis agents run in parallel (depend on Review agents)

        Args:
            selected_reviewers: Optional list of reviewer names to use (uses all if None)
            selected_contextualizers: Optional list of contextualizer names to use (uses all if None)
            selected_analyzers: Optional list of analyzer names to use (uses all if None)

        Returns:
            Built Strands Graph ready for execution
        """
        builder = GraphBuilder()

        # 1. Add document processor as entry point
        doc_processor = DocumentProcessorNode()
        builder.add_node(doc_processor, "document_processor")
        builder.set_entry_point("document_processor")

        # 2. Filter and add context agents (run in parallel)
        context_agents_to_use: list[ContextAgent] = self._filter_context_agents(
            selected_contextualizers
        )

        for context_agent in context_agents_to_use:
            builder.add_node(context_agent, context_agent.name)
            # Context agents depend on document processor
            builder.add_edge("document_processor", context_agent.name)

        # 3. Filter and add review agents (run in parallel)
        review_agents_to_use: list[ReviewAgent] = self._filter_review_agents(
            selected_reviewers
        )

        for review_agent in review_agents_to_use:
            builder.add_node(review_agent, review_agent.name)
            
            # If we have context agents, reviewers depend on them (sequential chain)
            # Otherwise, reviewers depend directly on document processor
            if context_agents_to_use:
                # Sequential: document_processor → contextualizer → reviewer
                for context_agent in context_agents_to_use:
                    builder.add_edge(context_agent.name, review_agent.name)
            else:
                # No contextualizer: document_processor → reviewer (direct)
                builder.add_edge("document_processor", review_agent.name)

        # 4. Add analysis agents (if enabled) - run in parallel
        if self.enable_analysis:
            analysis_agents_to_use = self._filter_analysis_agents(selected_analyzers)

            for analysis_agent in analysis_agents_to_use:
                builder.add_node(analysis_agent, analysis_agent.name)

                # Analysis agents depend on all review agents
                for review_agent in review_agents_to_use:
                    builder.add_edge(review_agent.name, analysis_agent.name)

        print(
            f"🏗️  Built graph with {len(review_agents_to_use)} reviewers, {len(context_agents_to_use)} contextualizers, {len(analysis_agents_to_use) if self.enable_analysis else 0} analyzers"
        )

        return builder.build()

    def build_manifest_driven_graph(
        self,
        manifest_config: dict[str, Any],
        directory_path: Path | None = None,
    ) -> Graph:
        """Build a review graph based on manifest configuration.

        Args:
            manifest_config: Parsed manifest configuration
            directory_path: Optional directory path for resolving relative paths

        Returns:
            Built Strands Graph configured according to manifest
        """
        builder = GraphBuilder()

        # 1. Add document processor as entry point
        doc_processor = DocumentProcessorNode()
        builder.add_node(doc_processor, "document_processor")
        builder.set_entry_point("document_processor")

        review_config = manifest_config.get("review_configuration", {})

        # 2. Load agents based on manifest specification
        selected_contextualizers: list[ContextAgent] = (
            self._load_contextualizers_from_manifest(review_config)
        )
        selected_reviewers: list[ReviewAgent] = self._load_reviewers_from_manifest(
            review_config
        )
        selected_analyzers: list[AnalysisAgent] = self._load_analyzers_from_manifest(
            review_config
        )

        # 3. Add context agents
        for context_agent in selected_contextualizers:
            builder.add_node(context_agent, context_agent.name)
            builder.add_edge("document_processor", context_agent.name)

        # 4. Add review agents with focus instructions if specified
        focus_config: dict[str, Any] = review_config.get("processed_focus", {})

        for review_agent in selected_reviewers:
            # Apply focus instructions to reviewer if available
            if focus_config and focus_config.get("focus_instructions"):
                review_agent = self._apply_focus_to_reviewer(review_agent, focus_config)

            builder.add_node(review_agent, review_agent.name)
            
            # If we have contextualizers, reviewers depend on them (sequential chain)
            # Otherwise, reviewers depend directly on document processor
            if selected_contextualizers:
                # Sequential: document_processor → contextualizer → reviewer
                for context_agent in selected_contextualizers:
                    builder.add_edge(context_agent.name, review_agent.name)
            else:
                # No contextualizer: document_processor → reviewer (direct)
                builder.add_edge("document_processor", review_agent.name)

        # 5. Add analysis agents
        for analysis_agent in selected_analyzers:
            builder.add_node(analysis_agent, analysis_agent.name)

            # Connect to all review agents
            for review_agent in selected_reviewers:
                builder.add_edge(review_agent.name, analysis_agent.name)

        print(
            f"🎯 Built manifest-driven graph with {len(selected_reviewers)} reviewers, {len(selected_contextualizers)} contextualizers, {len(selected_analyzers)} analyzers"
        )

        return builder.build()

    def build_simple_review_graph(self, content: str) -> Graph:
        """Build a simple graph for direct content review (no document processing).

        Args:
            content: Content to review directly

        Returns:
            Built Strands Graph for simple review
        """
        builder = GraphBuilder()

        # Use all available review agents as parallel entry points
        for review_agent in self.review_agents:
            builder.add_node(review_agent, review_agent.name)
            builder.set_entry_point(
                review_agent.name
            )  # Each reviewer is an entry point

        # Add analysis if enabled
        if self.enable_analysis and self.analysis_agents:
            for analysis_agent in self.analysis_agents:
                builder.add_node(analysis_agent, analysis_agent.name)

                # Connect to all reviewers
                for review_agent in self.review_agents:
                    builder.add_edge(review_agent.name, analysis_agent.name)

        print(
            f"🔧 Built simple graph with {len(self.review_agents)} reviewers, {len(self.analysis_agents) if self.enable_analysis else 0} analyzers"
        )

        return builder.build()

    def _filter_review_agents(
        self, selected_names: list[str] | None
    ) -> list[ReviewAgent]:
        """Filter review agents by name selection.

        Args:
            selected_names: List of agent names to include, or None for all

        Returns:
            Filtered list of ReviewAgent objects
        """
        if selected_names is None:
            return self.review_agents

        selected_lower = [name.lower() for name in selected_names]
        filtered = [
            agent
            for agent in self.review_agents
            if agent.persona.name.lower() in selected_lower
        ]

        if not filtered:
            print(f"⚠️  No review agents found matching: {selected_names}")
            print(
                f"Available agents: {[agent.persona.name for agent in self.review_agents]}"
            )
            return self.review_agents

        return filtered

    def _filter_context_agents(
        self, selected_names: list[str] | None
    ) -> list[ContextAgent]:
        """Filter context agents by name selection.

        Args:
            selected_names: List of agent names to include, or None for all

        Returns:
            Filtered list of ContextAgent objects
        """
        if selected_names is None:
            return self.context_agents

        selected_lower = [name.lower() for name in selected_names]
        filtered = [
            agent
            for agent in self.context_agents
            if agent.persona.name.lower() in selected_lower
        ]

        if not filtered:
            print(f"⚠️  No context agents found matching: {selected_names}")
            print(
                f"Available agents: {[agent.persona.name for agent in self.context_agents]}"
            )
            return self.context_agents

        return filtered

    def _filter_analysis_agents(
        self, selected_names: list[str] | None
    ) -> list[AnalysisAgent]:
        """Filter analysis agents by name selection.

        Args:
            selected_names: List of agent names to include, or None for all

        Returns:
            Filtered list of AnalysisAgent objects
        """
        if selected_names is None:
            return self.analysis_agents

        selected_lower = [name.lower() for name in selected_names]
        filtered = [
            agent
            for agent in self.analysis_agents
            if agent.persona.name.lower() in selected_lower
        ]

        if not filtered:
            print(f"⚠️  No analysis agents found matching: {selected_names}")
            print(
                f"Available agents: {[agent.persona.name for agent in self.analysis_agents]}"
            )
            return self.analysis_agents

        return filtered

    def _load_contextualizers_from_manifest(
        self, review_config: dict[str, Any]
    ) -> list[ContextAgent]:
        """Load contextualizers based on manifest configuration.

        Args:
            review_config: Review configuration section from manifest

        Returns:
            List of selected ContextAgent objects
        """
        try:
            selected_personas = self.persona_loader.load_contextualizers_from_manifest(
                review_config
            )
            agents = [
                ContextAgent(
                    persona=persona,
                    model_provider=self.model_provider,
                    model_config=self.model_config,
                )
                for persona in selected_personas
            ]
            return agents
        except Exception as e:
            print(f"⚠️  Error loading contextualizers from manifest: {e}")
            return self.context_agents  # Fallback to all available

    def _load_reviewers_from_manifest(
        self, review_config: dict[str, Any]
    ) -> list[ReviewAgent]:
        """Load reviewers based on manifest configuration.

        Args:
            review_config: Review configuration section from manifest

        Returns:
            List of selected ReviewAgent objects
        """
        try:
            selected_personas = self.persona_loader.load_reviewers_from_manifest(
                review_config
            )
            agents = [
                ReviewAgent(
                    persona,
                    model_provider=self.model_provider,
                    model_config_override=self.model_config,
                )
                for persona in selected_personas
            ]
            return agents
        except Exception as e:
            print(f"⚠️  Error loading reviewers from manifest: {e}")
            return self.review_agents  # Fallback to all available

    def _load_analyzers_from_manifest(
        self, review_config: dict[str, Any]
    ) -> list[AnalysisAgent]:
        """Load analyzers based on manifest configuration.

        Args:
            review_config: Review configuration section from manifest

        Returns:
            List of selected AnalysisAgent objects
        """
        if not self.enable_analysis:
            return []

        try:
            selected_personas = self.persona_loader.load_analyzers_from_manifest(
                review_config
            )
            agents = [
                AnalysisAgent(
                    persona=persona,
                    model_provider=self.model_provider,
                    model_config=self.model_config,
                )
                for persona in selected_personas
            ]
            return agents
        except Exception as e:
            print(f"⚠️  Error loading analyzers from manifest: {e}")
            return self.analysis_agents  # Fallback to all available

    def _apply_focus_to_reviewer(
        self, reviewer: ReviewAgent, focus_config: dict[str, Any]
    ) -> ReviewAgent:
        """Apply focus instructions to a reviewer persona.

        Args:
            reviewer: ReviewAgent to enhance
            focus_config: Focus configuration with instructions

        Returns:
            ReviewAgent with enhanced prompt template
        """
        focus_instructions = focus_config.get("focus_instructions", [])

        if not focus_instructions:
            return reviewer

        # Create enhanced prompt template
        focus_section = "\n\n## SPECIAL FOCUS AREAS FOR THIS REVIEW\n"
        focus_section += "\n".join(focus_instructions)
        focus_section += (
            "\n\nPlease pay particular attention to these focus areas in your review.\n"
        )

        # Create new persona with enhanced prompt
        enhanced_persona = PersonaConfig(
            name=reviewer.persona.name,
            role=reviewer.persona.role,
            goal=reviewer.persona.goal,
            backstory=reviewer.persona.backstory,
            prompt_template=reviewer.persona.prompt_template + focus_section,
            model_config=reviewer.persona.model_config,
        )

        # Create new agent with enhanced persona
        return ReviewAgent(
            enhanced_persona,
            model_provider=self.model_provider,
            model_config_override=self.model_config,
        )

    def get_available_agents_info(self) -> dict[str, list[dict[str, Any]]]:
        """Get information about all available agents.

        Returns:
            Dictionary with agent type as key and list of agent info as value
        """
        return {
            "reviewers": [
                {
                    "name": agent.persona.name,
                    "role": agent.persona.role,
                    "goal": agent.persona.goal,
                }
                for agent in self.review_agents
            ],
            "contextualizers": [
                {
                    "name": agent.persona.name,
                    "role": agent.persona.role,
                    "goal": agent.persona.goal,
                }
                for agent in self.context_agents
            ],
            "analyzers": [
                {
                    "name": agent.persona.name,
                    "role": agent.persona.role,
                    "goal": agent.persona.goal,
                }
                for agent in self.analysis_agents
            ],
        }

    # Note: Conditional edges were planned but Strands GraphBuilder doesn't support
    # add_conditional_edge method. All edges are currently unconditional.

    async def execute_graph(self, graph: Graph, input_data: Any) -> MultiAgentResult:
        """Execute a graph with input data.

        Args:
            graph: Built Strands Graph
            input_data: Input data to process

        Returns:
            MultiAgentResult with execution results
        """
        result = await graph.invoke_async(input_data)
        return result  # type: ignore[return-value]
