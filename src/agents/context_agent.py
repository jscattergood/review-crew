"""
Context Agent for processing and formatting contextual information.

This agent is responsible for taking separate context information and
formatting it according to a contextualizer persona for use by review agents.
The agent is completely abstract and domain-agnostic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from strands.multiagent.base import MultiAgentResult
from strands.types.content import ContentBlock

from ..config.persona_loader import PersonaConfig
from .base_agent import BaseAgent


@dataclass
class ContextResult:
    """Result from context processing and formatting."""

    formatted_context: str
    context_summary: str
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ContextAgent(BaseAgent):
    """Agent that processes and formats contextual information using a contextualizer persona."""

    def __init__(
        self,
        persona: PersonaConfig,
        model_provider: str = "bedrock",
        model_config: dict[str, Any] | None = None,
    ):
        """Initialize the context agent.

        Args:
            persona: PersonaConfig object with contextualizer settings
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
        """
        # Initialize the base agent
        super().__init__(persona, model_provider, model_config)

    async def process_context(self, context_data: str) -> ContextResult:
        """Process and format contextual information using the contextualizer persona.

        Args:
            context_data: The raw context information to be processed and formatted

        Returns:
            ContextResult with formatted context
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
        result = await self.invoke_async_legacy(prompt, "context_processing")

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

    def _extract_section(self, text: str, section_header: str) -> str | None:
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

    def get_info(self) -> dict[str, Any]:
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

    async def invoke_async_graph(
        self, task: str | list[ContentBlock], **kwargs: Any
    ) -> MultiAgentResult:
        """Process task asynchronously for graph execution using specialized context logic.

        Args:
            task: Input content to process
            **kwargs: Additional arguments

        Returns:
            MultiAgentResult with context results
        """
        import time

        from strands.agent.agent_result import AgentResult
        from strands.multiagent.base import MultiAgentResult, NodeResult, Status
        from strands.telemetry.metrics import EventLoopMetrics
        from strands.types.content import ContentBlock, Message

        try:
            # Extract formatted_context from the task
            # Strands converts upstream outputs to list[ContentBlock], never MultiAgentResult
            import re
            
            self.logger.info(f"[CONTEXT_AGENT] Task type: {type(task).__name__}")
            
            # Extract raw content from list WITHOUT stripping markers
            # (ContextAgent needs the markers to find the context)
            if isinstance(task, list):
                # Extract text from ContentBlocks
                text_parts = []
                for item in task:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                full_content = "\n".join(text_parts)
            else:
                # Fallback to base extraction (for str or other types)
                full_content = self._extract_content_from_task(task)
            
            # Now extract context from markers and also preserve the essays/content
            match = re.search(r'__CONTEXT_START__\n(.*?)\n__CONTEXT_END__\n\n(.*)', full_content, re.DOTALL)
            if match:
                formatted_context = match.group(1)
                essays_content = match.group(2)  # Preserve the essays
                self.logger.info(f"[CONTEXT_AGENT] Extracted formatted_context from markers, length: {len(formatted_context)}")
                self.logger.info(f"[CONTEXT_AGENT] Preserved essays content, length: {len(essays_content)}")
            else:
                # Fallback: use full content (for backwards compatibility with workflows without context files)
                formatted_context = full_content
                essays_content = ""  # No essays to preserve
                self.logger.info(f"[CONTEXT_AGENT] No markers found, using full content as fallback")
            
            # Process the context (not the essays) to create a summary
            start_time = time.time()
            context_result: ContextResult = await self.process_context(formatted_context)
            execution_time = int(time.time() - start_time)

            # Format response: Context summary + Original essays content
            # This ensures the reviewer gets BOTH the context summary AND the essays
            response = f"**Context Summary:** {context_result.context_summary}\n\n**Formatted Context:** {context_result.formatted_context}"
            
            # IMPORTANT: Pass through the original essays content
            if essays_content:
                response += f"\n\n{essays_content}"

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
                            stop_reason="end_turn",
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
                execution_time=0,
                execution_count=1,
            )
