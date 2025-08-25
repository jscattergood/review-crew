"""
Conversation Manager for orchestrating multi-agent reviews.

This module manages conversations between multiple review agents,
collecting and organizing their feedback.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .review_agent import ReviewAgent
from .analysis_agent import AnalysisAgent, AnalysisResult
from .context_agent import ContextAgent, ContextResult
from ..config.persona_loader import PersonaLoader, PersonaConfig

import tiktoken


@dataclass
class ReviewResult:
    """Result from a single agent review."""

    agent_name: str
    agent_role: str
    feedback: str
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class ConversationResult:
    """Complete conversation result with all agent reviews."""

    content: str
    reviews: List[ReviewResult]
    timestamp: datetime
    summary: Optional[str] = None
    analysis_results: List[AnalysisResult] = None
    context_results: List[ContextResult] = None
    original_content: Optional[str] = None
    analysis_errors: List[str] = None  # Track analysis failures separately

    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.context_results is None:
            self.context_results = []
        if self.analysis_errors is None:
            self.analysis_errors = []


class ConversationManager:
    """Manages multi-agent review conversations."""

    def __init__(
        self,
        persona_loader: Optional[PersonaLoader] = None,
        model_provider: str = "bedrock",
        model_config: Optional[Dict[str, Any]] = None,
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
        self.agents: List[ReviewAgent] = []
        self.context_agents: List[ContextAgent] = []
        self.analysis_agents: List[AnalysisAgent] = []
        self._load_agents()
        self._load_contextualizers()

        if self.enable_analysis:
            self._load_analyzers()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        # Use cl100k_base encoding (used by GPT-4 and similar models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _truncate_to_token_limit(self, text: str, target_tokens: int) -> str:
        """Truncate text to the target number of tokens.
        
        Args:
            text: Text to truncate
            target_tokens: Target number of tokens
            
        Returns:
            Truncated text
        """
        if target_tokens <= 0:
            return ""
            
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        if len(tokens) <= target_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:target_tokens]
        return encoding.decode(truncated_tokens)

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

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get information about available agents.

        Returns:
            List of agent information dictionaries
        """
        return [agent.get_info() for agent in self.agents]

    def get_available_contextualizers(self) -> List[Dict[str, Any]]:
        """Get information about available contextualizer agents.

        Returns:
            List of contextualizer agent information dictionaries
        """
        return [agent.get_info() for agent in self.context_agents]

    def get_available_analyzers(self) -> List[Dict[str, Any]]:
        """Get information about available analyzer agents.

        Returns:
            List of analyzer agent information dictionaries
        """
        return [agent.get_info() for agent in self.analysis_agents]

    def run_review(
        self,
        content: str,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run a synchronous review with selected agents.

        Args:
            content: Content to review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        if not self.agents:
            raise ValueError(
                "No review agents available. Check your persona configurations."
            )

        # Process context with all contextualizers if provided and available
        context_results = []
        if context_data and self.context_agents:
            print(
                f"🔍 Processing context information with {len(self.context_agents)} contextualizers..."
            )
            for context_agent in self.context_agents:
                try:
                    context_result = context_agent.process_context(context_data)
                    if context_result:
                        context_results.append(context_result)
                        print(
                            f"  ✅ Context processed by {context_agent.persona.name}: {context_result.context_summary}"
                        )
                    else:
                        print(f"  ℹ️  {context_agent.persona.name} returned no context")
                except Exception as e:
                    print(
                        f"  ⚠️  Context processing failed for {context_agent.persona.name}: {e}"
                    )
        elif context_data:
            print("  ℹ️  Context data provided but no contextualizer personas available")

        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)

        print(f"🎭 Starting review with {len(agents_to_use)} agents...")

        reviews = []
        for agent in agents_to_use:
            print(f"  📝 {agent.persona.name} is reviewing...")

            try:
                # Prepare content for review (include context if available)
                content_to_review = self._prepare_content_for_review(
                    content, context_results
                )

                feedback = agent.review(content_to_review)
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback=feedback,
                    timestamp=datetime.now(),
                )
                reviews.append(review)
                print(f"  ✅ {agent.persona.name} completed review")

            except Exception as e:
                error_str = str(e)
                # Check if this is a context length error during review generation
                if "context length" in error_str.lower() and "4096" in error_str:
                    print(f"  ❌ {agent.persona.name} failed due to context length limit (input too large for model)")
                    print(f"      The model ran out of space before completing the review")
                    print(f"      Try: --max-context-length 8192 or use a larger model")
                    # Mark as failed with helpful error message
                    error_review = ReviewResult(
                        agent_name=agent.persona.name,
                        agent_role=agent.persona.role,
                        feedback="",
                        timestamp=datetime.now(),
                        error=f"Context length exceeded: {error_str}. The input content + context was too large for the 4096 token model limit. Any partial output shown in console was incomplete.",
                    )
                    reviews.append(error_review)
                else:
                    error_review = ReviewResult(
                        agent_name=agent.persona.name,
                        agent_role=agent.persona.role,
                        feedback="",
                        timestamp=datetime.now(),
                        error=error_str,
                    )
                    reviews.append(error_review)
                    print(f"  ❌ {agent.persona.name} failed: {e}")

        result = ConversationResult(
            content=content,
            reviews=reviews,
            context_results=context_results,
            original_content=None,  # No longer needed since we're not extracting
            timestamp=datetime.now(),
        )

        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agents:
            successful_reviews = [r for r in reviews if not r.error]
            if (
                len(successful_reviews) > 0
            ):  # Always run analysis if we have any successful reviews
                print(
                    f"🧠 Performing analysis with {len(self.analysis_agents)} analyzers..."
                )
                analysis_results = []

                # Convert ReviewResult objects to dictionaries for analysis
                review_dicts = [
                    {
                        "agent_name": r.agent_name,
                        "agent_role": r.agent_role,
                        "feedback": r.feedback,
                        "error": r.error,
                    }
                    for r in successful_reviews
                ]

                # Get context length from model config or use default
                max_context_length = self.model_config.get("max_context_length", None)

                analysis_errors = []
                for analysis_agent in self.analysis_agents:
                    try:
                        print(
                            f"  📊 Running analysis with {analysis_agent.persona.name}..."
                        )
                        analysis_result = analysis_agent.analyze(
                            review_dicts, max_context_length
                        )
                        analysis_results.append(analysis_result)
                        print(
                            f"  ✅ Analysis complete for {analysis_agent.persona.name}"
                        )
                    except Exception as e:
                        error_msg = f"{analysis_agent.persona.name}: {str(e)}"
                        analysis_errors.append(error_msg)
                        print(
                            f"  ⚠️  Analysis failed for {analysis_agent.persona.name}: {e}"
                        )

                result.analysis_results = analysis_results
                result.analysis_errors = analysis_errors
                
                if analysis_results:
                    print(
                        f"✅ Analysis complete! Ran {len(analysis_results)} successful analyzers"
                    )
                if analysis_errors:
                    print(
                        f"⚠️  {len(analysis_errors)} analyzer(s) failed but reviews are still available"
                    )
                    print("💡 Check the output for detailed analysis error information")

        print(
            f"🎉 Review complete! Collected {len([r for r in reviews if not r.error])} successful reviews"
        )
        return result

    async def run_review_async(
        self,
        content: str,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run an asynchronous review with selected agents.

        Args:
            content: Content to review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        if not self.agents:
            raise ValueError(
                "No review agents available. Check your persona configurations."
            )

        # Process context with all contextualizers if provided and available
        context_results = []
        if context_data and self.context_agents:
            print(
                f"🔍 Processing context information with {len(self.context_agents)} contextualizers..."
            )

            # Run all contextualizers concurrently
            context_tasks = []
            for context_agent in self.context_agents:
                task = self._process_context_with_agent_async(
                    context_agent, context_data
                )
                context_tasks.append(task)

            context_task_results = await asyncio.gather(
                *context_tasks, return_exceptions=True
            )

            # Process results
            for i, result in enumerate(context_task_results):
                context_agent = self.context_agents[i]
                if isinstance(result, Exception):
                    print(
                        f"  ⚠️  Context processing failed for {context_agent.persona.name}: {result}"
                    )
                elif result:
                    context_results.append(result)
                    print(
                        f"  ✅ Context processed by {context_agent.persona.name}: {result.context_summary}"
                    )
                else:
                    print(f"  ℹ️  {context_agent.persona.name} returned no context")
        elif context_data:
            print("  ℹ️  Context data provided but no contextualizer personas available")

        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)

        print(f"🎭 Starting async review with {len(agents_to_use)} agents...")

        # Prepare content for review
        content_to_review = self._prepare_content_for_review(content, context_results)

        # Run all reviews concurrently
        tasks = []
        for agent in agents_to_use:
            task = self._review_with_agent_async(agent, content_to_review)
            tasks.append(task)

        reviews = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_reviews = []
        for i, result in enumerate(reviews):
            agent = agents_to_use[i]

            if isinstance(result, Exception):
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=str(result),
                )
                print(f"  ❌ {agent.persona.name} failed: {result}")
            else:
                review = result
                print(f"  ✅ {agent.persona.name} completed review")

            processed_reviews.append(review)

        result = ConversationResult(
            content=content,
            reviews=processed_reviews,
            context_results=context_results,
            timestamp=datetime.now(),
        )

        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agents:
            successful_reviews = [r for r in processed_reviews if not r.error]
            if (
                len(successful_reviews) > 0
            ):  # Always run analysis if we have any successful reviews
                print(
                    f"🧠 Performing async analysis with {len(self.analysis_agents)} analyzers..."
                )

                # Convert ReviewResult objects to dictionaries for analysis
                review_dicts = [
                    {
                        "agent_name": r.agent_name,
                        "agent_role": r.agent_role,
                        "feedback": r.feedback,
                        "error": r.error,
                    }
                    for r in successful_reviews
                ]

                # Get context length from model config or use default
                max_context_length = self.model_config.get("max_context_length", None)

                # Run all analyzers concurrently
                analysis_tasks = []
                for analysis_agent in self.analysis_agents:
                    task = self._analyze_with_agent_async(
                        analysis_agent, review_dicts, max_context_length
                    )
                    analysis_tasks.append(task)

                analysis_task_results = await asyncio.gather(
                    *analysis_tasks, return_exceptions=True
                )

                # Process results
                analysis_results = []
                analysis_errors = []
                for i, analysis_result in enumerate(analysis_task_results):
                    analysis_agent = self.analysis_agents[i]
                    if isinstance(analysis_result, Exception):
                        error_msg = f"{analysis_agent.persona.name}: {str(analysis_result)}"
                        analysis_errors.append(error_msg)
                        print(
                            f"  ⚠️  Analysis failed for {analysis_agent.persona.name}: {analysis_result}"
                        )
                    else:
                        analysis_results.append(analysis_result)
                        print(
                            f"  ✅ Analysis complete for {analysis_agent.persona.name}"
                        )

                result.analysis_results = analysis_results
                result.analysis_errors = analysis_errors
                
                if analysis_results:
                    print(
                        f"✅ Async analysis complete! Ran {len(analysis_results)} successful analyzers"
                    )
                if analysis_errors:
                    print(
                        f"⚠️  {len(analysis_errors)} analyzer(s) failed but reviews are still available"
                    )
                    print("💡 Check the output for detailed analysis error information")

        print(
            f"🎉 Async review complete! Collected {len([r for r in processed_reviews if not r.error])} successful reviews"
        )
        return result

    async def _review_with_agent_async(
        self, agent: ReviewAgent, content: str
    ) -> ReviewResult:
        """Review content with a single agent asynchronously."""
        try:
            feedback = await agent.review_async(content)
            return ReviewResult(
                agent_name=agent.persona.name,
                agent_role=agent.persona.role,
                feedback=feedback,
                timestamp=datetime.now(),
            )
        except Exception as e:
            error_str = str(e)
            # Check if this is a context length error during review generation
            if "context length" in error_str.lower() and "4096" in error_str:
                print(f"  ❌ {agent.persona.name} failed due to context length limit (input too large for model)")
                print(f"      The model ran out of space before completing the review")
                print(f"      Try: --max-context-length 8192 or use a larger model")
                # Mark as failed with helpful error message
                return ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=f"Context length exceeded: {error_str}. The input content + context was too large for the 4096 token model limit. Any partial output shown in console was incomplete.",
                )
            else:
                return ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=error_str,
                )

    async def _process_context_with_agent_async(
        self, context_agent: ContextAgent, context_data: str
    ) -> Optional[ContextResult]:
        """Process context with a single context agent asynchronously."""
        try:
            return await context_agent.process_context_async(context_data)
        except Exception as e:
            print(
                f"  ⚠️  Context processing failed for {context_agent.persona.name}: {e}"
            )
            return None

    async def _analyze_with_agent_async(
        self,
        analysis_agent: AnalysisAgent,
        review_dicts: List[Dict[str, Any]],
        max_context_length: Optional[int],
    ) -> AnalysisResult:
        """Analyze reviews with a single analysis agent asynchronously."""
        return await analysis_agent.analyze_async(review_dicts, max_context_length)

    def _filter_agents(self, selected_agents: Optional[List[str]]) -> List[ReviewAgent]:
        """Filter agents based on selection.

        Args:
            selected_agents: List of agent names to include, or None for all

        Returns:
            Filtered list of ReviewAgent objects
        """
        if selected_agents is None:
            return self.agents

        # Filter by name (case-insensitive)
        selected_lower = [name.lower() for name in selected_agents]
        filtered = [
            agent
            for agent in self.agents
            if agent.persona.name.lower() in selected_lower
        ]

        if not filtered:
            print(f"⚠️  No agents found matching: {selected_agents}")
            print(f"Available agents: {[agent.persona.name for agent in self.agents]}")
            return self.agents

        return filtered

    def _prepare_content_for_review(
        self, content: str, context_results: List[ContextResult]
    ) -> str:
        """Prepare content for review by combining it with extracted context.

        Args:
            content: The cleaned content to review
            context_results: List of context extraction results

        Returns:
            Formatted content for review agents
        """
        if not context_results:
            return content

        # Format all context results for reviewers
        context_sections = []
        for i, context_result in enumerate(context_results, 1):
            # Find the corresponding context agent to format the context
            context_agent = None
            for agent in self.context_agents:
                if agent.persona and context_result.context_summary:
                    # Match by checking if the context summary matches
                    context_agent = agent
                    break

            if context_agent:
                context_section = context_agent.format_context_for_review(
                    context_result
                )
                context_sections.append(f"### Context {i}\n{context_section}")
            else:
                # Fallback formatting if no agent found
                context_sections.append(
                    f"### Context {i}\n## CONTEXT\n{context_result.formatted_context}\n\n---\n"
                )

        # Combine all context with content
        all_context = "\n".join(context_sections)
        combined_content = f"""{all_context}

## CONTENT TO REVIEW
{content}"""

        # Check if the combined content is too long and truncate if necessary
        return self._truncate_if_needed(combined_content, content, all_context)

    def _truncate_if_needed(self, combined_content: str, original_content: str, context_content: str) -> str:
        """Truncate context if the combined content is too long for the model.
        
        Args:
            combined_content: The full combined content (context + original content)
            original_content: The original content to review (must be preserved)
            context_content: The context content that can be truncated
            
        Returns:
            Truncated content if necessary, with warning logged
        """
        # Get max context length from model config, default to 4096 if not set
        max_context_length = self.model_config.get('max_context_length', 4096)
        
        # Reserve tokens for model response and prompt overhead
        response_buffer = 1000  # tokens for response
        prompt_overhead = 800   # tokens for system prompt and formatting
        
        # Available tokens for input content
        available_tokens = max_context_length - response_buffer - prompt_overhead
        
        # Count actual tokens in the combined content
        current_tokens = self._count_tokens(combined_content)
        
        print(f"📏 Token limit check: {current_tokens} tokens vs {available_tokens} available")
        
        # Check if truncation is needed
        if current_tokens <= available_tokens:
            return combined_content
        
        # Calculate how much context we need to truncate
        original_content_with_header = f"## CONTENT TO REVIEW\n{original_content}"
        original_tokens = self._count_tokens(original_content_with_header)
        context_budget_tokens = available_tokens - original_tokens
        
        if context_budget_tokens <= 0:
            # Original content itself is too long, warn but proceed
            print(f"⚠️  Warning: Original content ({original_tokens} tokens) exceeds available context budget")
            print("   Proceeding without context to avoid model errors")
            return original_content
        
        # Truncate context content to fit token budget
        context_tokens = self._count_tokens(context_content)
        if context_tokens > context_budget_tokens:
            # Binary search to find the right truncation point
            truncated_context = self._truncate_to_token_limit(context_content, context_budget_tokens - 50)  # -50 for truncation message
            truncation_msg = "\n\n[... Context truncated due to token limits ...]"
            truncated_context += truncation_msg
            
            final_tokens = self._count_tokens(truncated_context)
            print(f"⚠️  Warning: Context truncated from {context_tokens} to {final_tokens} tokens")
            print(f"   Available context budget: {context_budget_tokens} tokens, Model limit: {max_context_length} tokens")
            
            return f"""{truncated_context}

## CONTENT TO REVIEW
{original_content}"""
        
        return combined_content

    def format_results(
        self, result: ConversationResult, include_content: bool = True, include_context: bool = False
    ) -> str:
        """Format conversation results for display as clean markdown.

        Args:
            result: ConversationResult to format
            include_content: Whether to include the original content
            include_context: Whether to include the context results from contextualizers

        Returns:
            Formatted markdown string
        """
        output = []

        # Header
        output.append("# Review-Crew Analysis Results")
        output.append("")
        output.append(
            f"**Analysis completed:** {result.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}"
        )
        output.append("")

        if include_content:
            output.append("## Content Reviewed")
            output.append("")
            # Format content with proper markdown
            content_lines = result.content.split("\n")
            for line in content_lines:
                if line.strip():
                    output.append(line)
                else:
                    output.append("")
            output.append("")

        # Summary
        successful_reviews = [r for r in result.reviews if not r.error]
        failed_reviews = [r for r in result.reviews if r.error]

        output.append("## Summary")
        output.append("")
        output.append(f"- **Total Reviews:** {len(result.reviews)}")
        output.append(f"- **Successful:** {len(successful_reviews)} ✅")
        if failed_reviews:
            output.append(f"- **Failed:** {len(failed_reviews)} ❌")
        if result.context_results:
            output.append(
                f"- **Context Results:** {len(result.context_results)} contextualizers 🔍"
            )
        if result.analysis_results:
            output.append(
                f"- **Analysis Results:** {len(result.analysis_results)} analyzers 🧠"
            )
        if hasattr(result, 'analysis_errors') and result.analysis_errors:
            output.append(
                f"- **Analysis Errors:** {len(result.analysis_errors)} analyzers failed ⚠️"
            )
        output.append("")

        # Context Results Section
        if include_context and result.context_results:
            output.append("## Context Information")
            output.append("")
            output.append("The following context was processed by contextualizers and provided to reviewers:")
            output.append("")

            for i, context_result in enumerate(result.context_results, 1):
                # Try to find the corresponding context agent to get its name
                context_agent_name = f"Contextualizer {i}"
                if i <= len(self.context_agents) and self.context_agents[i - 1].persona:
                    context_agent_name = self.context_agents[i - 1].persona.name

                output.append(f"### {i}. {context_agent_name}")
                output.append("")
                output.append(f"**Summary:** {context_result.context_summary}")
                output.append("")
                output.append("**Formatted Context:**")
                output.append("")
                output.append("```")
                output.append(context_result.formatted_context)
                output.append("```")
                output.append("")
                if i < len(result.context_results):  # Don't add separator after last context
                    output.append("---")
                    output.append("")

        # Individual Reviews
        if successful_reviews:
            output.append("## Individual Reviews")
            output.append("")

            for i, review in enumerate(successful_reviews, 1):
                output.append(f"### {i}. {review.agent_name}")
                output.append(f"**Role:** {review.agent_role}")
                output.append("")

                # Extract clean text from feedback (handle both string and dict formats)
                clean_feedback = self._extract_clean_feedback(review.feedback)
                
                output.append(clean_feedback)
                output.append("")
                output.append("---")
                output.append("")

        # Analysis Section
        if result.analysis_results:
            output.append("## Analysis & Synthesis")
            output.append("")

            for i, analysis in enumerate(result.analysis_results, 1):
                # Try to get the analyzer name from the analysis or use a generic name
                analyzer_name = f"Analyzer {i}"
                # If we can identify the analyzer, use its name
                if hasattr(analysis, "analyzer_name"):
                    analyzer_name = analysis.analyzer_name
                elif i <= len(self.analysis_agents) and self.analysis_agents[i - 1].persona:
                    analyzer_name = self.analysis_agents[i - 1].persona.name

                output.append(f"### {analyzer_name}")
                output.append("")
                output.append(analysis.synthesis)
                output.append("")
                if i < len(
                    result.analysis_results
                ):  # Don't add separator after last analysis
                    output.append("---")
                    output.append("")

        # Failed Reviews
        if failed_reviews:
            output.append("## Failed Reviews")
            output.append("")
            for review in failed_reviews:
                output.append(f"- **{review.agent_name}:** {review.error}")
            output.append("")

        # Analysis Errors
        if hasattr(result, 'analysis_errors') and result.analysis_errors:
            output.append("## Analysis Errors")
            output.append("")
            output.append("The following analyzers failed but reviews were completed successfully:")
            output.append("")
            for error in result.analysis_errors:
                output.append(f"- **{error}**")
            output.append("")
            output.append("*Note: Reviews are still available above even though analysis failed.*")
            output.append("")

        return "\n".join(output)

    def _extract_clean_feedback(self, feedback) -> str:
        """Extract clean text from feedback, handling various formats."""
        import json
        import ast

        if isinstance(feedback, str):
            # Try to parse as JSON/dict if it looks like one
            if feedback.strip().startswith("{") and feedback.strip().endswith("}"):
                try:
                    # Try JSON first
                    parsed = json.loads(feedback)
                    return self._extract_clean_feedback(parsed)
                except json.JSONDecodeError:
                    try:
                        # Try literal_eval for Python dict format
                        parsed = ast.literal_eval(feedback)
                        return self._extract_clean_feedback(parsed)
                    except (ValueError, SyntaxError):
                        # If parsing fails, return as-is
                        return feedback
            else:
                # If it's already a plain string, return as-is
                return feedback
        elif isinstance(feedback, dict):
            # Handle dictionary format (like from API responses)
            if "role" in feedback and "content" in feedback:
                content = feedback["content"]
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "text" in content[0]:
                        return content[0]["text"]
                    else:
                        return str(content[0])
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)
            elif "text" in feedback:
                return feedback["text"]
            else:
                return str(feedback)
        else:
            # Fallback to string representation
            return str(feedback)

    # Removed college-specific supplemental context methods - analysis personas now handle context generation automatically
