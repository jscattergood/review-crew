"""
Context Agent for processing and formatting contextual information.

This agent is responsible for taking separate context information and
formatting it according to a contextualizer persona for use by review agents.
The agent is completely abstract and domain-agnostic.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent
from ..config.persona_loader import PersonaConfig, PersonaLoader


@dataclass
class ContextResult:
    """Result from context processing and formatting."""

    formatted_context: str
    context_summary: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ContextAgent(BaseAgent):
    """Agent that processes and formats contextual information using a contextualizer persona."""

    def __init__(
        self,
        persona: PersonaConfig,
        model_provider: str = "bedrock",
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the context agent.

        Args:
            persona: PersonaConfig object with contextualizer settings
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
        """
        # Initialize the base agent
        super().__init__(persona, model_provider, model_config)

    def process_context(self, context_data: str) -> Optional[ContextResult]:
        """Process and format contextual information using the contextualizer persona.

        Args:
            context_data: The raw context information to be processed and formatted

        Returns:
            ContextResult with formatted context, or None if no contextualizer available
        """
        # Agent is guaranteed to exist since we pass persona in constructor

        # Format the prompt with the context data
        prompt = self.persona.prompt_template.format(content=context_data)

        # Use the base agent's invoke method
        result = self.invoke(prompt, "context_processing")

        # Parse the structured response
        return self._parse_context_response(result)

    async def process_context_async(self, context_data: str) -> Optional[ContextResult]:
        """Asynchronously process and format contextual information.

        Args:
            context_data: The raw context information to be processed and formatted

        Returns:
            ContextResult with formatted context, or None if no contextualizer available
        """
        # Check if we received an error instead of content
        if context_data == "ERROR_NO_CONTENT":
            return ContextResult(
                formatted_context="No content was provided for context processing.",
                context_summary="No content available",
                timestamp=datetime.now(),
            )

        # Agent is guaranteed to exist since we pass persona in constructor

        # Format the prompt with the context data
        prompt = self.persona.prompt_template.format(content=context_data)

        # Use the base agent's invoke_async method
        result = await self.invoke_async_legacy(prompt, "context_processing_async")

        # Parse the structured response
        return self._parse_context_response(result)

    def _parse_context_response(self, response: str) -> ContextResult:
        """Parse the structured context processing response."""
        # Try to extract a context summary if the persona provides one
        context_summary = self._extract_section(response, "CONTEXT SUMMARY")

        # If no structured summary found, create a simple one
        if not context_summary:
            context_summary = "Context processed by contextualizer persona"

        # The full response is the formatted context
        formatted_context = response

        # Check if the response contains raw JSON structure and extract clean text
        if formatted_context.startswith("{'role':") or formatted_context.startswith(
            '{"role":'
        ):
            try:
                # Try to parse the JSON structure and extract the actual text
                import ast

                parsed = ast.literal_eval(formatted_context.replace("'", '"'))
                if isinstance(parsed, dict) and "content" in parsed:
                    nested_content = parsed["content"]
                    if isinstance(nested_content, list) and len(nested_content) > 0:
                        if (
                            isinstance(nested_content[0], dict)
                            and "text" in nested_content[0]
                        ):
                            formatted_context = nested_content[0]["text"]
            except:
                # If parsing fails, use the original response
                pass

        return ContextResult(
            formatted_context=formatted_context, context_summary=context_summary
        )

    def _extract_section(self, text: str, section_header: str) -> Optional[str]:
        """Extract a specific section from the structured response."""
        import re

        # Look for the section header (with ## prefix)
        pattern = rf"## {section_header}\s*\n(.*?)(?=\n\s*## |\n\s*---|\n\s*\n.*\S|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            content = match.group(1).strip()
            return content

        return None

    def format_context_for_review(self, context_result: ContextResult) -> str:
        """Format processed context for inclusion in review prompts.

        Args:
            context_result: The result from context processing

        Returns:
            Formatted context string for reviewers
        """
        return f"""## CONTEXT
{context_result.formatted_context}

---
"""

    def get_info(self) -> Dict[str, Any]:
        """Get information about this context agent."""
        return {
            "name": self.persona.name,
            "role": self.persona.role,
            "goal": self.persona.goal,
            "capabilities": [
                "Process contextual information",
                "Format context for reviewers",
                "Create context summaries",
            ],
        }

    async def invoke_async_graph(self, task, **kwargs):
        """Process task asynchronously for graph execution using specialized context logic.

        Args:
            task: Input content to process
            **kwargs: Additional arguments

        Returns:
            MultiAgentResult with context results
        """
        import time
        from strands.multiagent.base import MultiAgentResult, NodeResult, Status
        from strands.agent.agent_result import AgentResult
        from strands.telemetry.metrics import EventLoopMetrics
        from strands.types.content import ContentBlock, Message

        try:
            # Extract content from task using base agent logic
            content = self._extract_content_from_task(task)

            # Process using the specialized context method with timing
            start_time = time.time()
            context_result = await self.process_context_async(content)
            execution_time = time.time() - start_time

            # Format context result as text for the response
            response = f"**Context Summary:** {context_result.context_summary}\n\n**Formatted Context:** {context_result.formatted_context}"

            # Create agent result
            metrics = EventLoopMetrics()
            agent_result = AgentResult(
                stop_reason="end_turn",
                message=Message(
                    role="assistant", content=[ContentBlock(text=response)]
                ),
                metrics=metrics,
                state={
                    "agent_name": self.persona.name,
                    "agent_role": self.persona.role,
                    "response": response,
                    "context_result": context_result,
                },
            )

            # Return wrapped in MultiAgentResult
            return MultiAgentResult(
                status=Status.COMPLETED,
                results={
                    self.name: NodeResult(result=agent_result, status=Status.COMPLETED)
                },
                execution_time=execution_time,
                execution_count=1,
            )

        except Exception as e:
            # Handle errors gracefully
            return MultiAgentResult(
                status=Status.FAILED,
                results={
                    self.name: NodeResult(
                        result=AgentResult(
                            stop_reason="error",
                            message=Message(
                                role="assistant",
                                content=[
                                    ContentBlock(
                                        text=f"Context processing failed: {str(e)}"
                                    )
                                ],
                            ),
                            metrics=EventLoopMetrics(),
                            state={
                                "agent_name": self.persona.name,
                                "agent_role": self.persona.role,
                                "error": str(e),
                                "response": f"Context processing failed: {str(e)}",
                            },
                        ),
                        status=Status.FAILED,
                    )
                },
                execution_time=0.0,
                execution_count=1,
            )
