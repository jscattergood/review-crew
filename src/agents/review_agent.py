"""
Review Agent implementation using Strands Agents.

This module wraps Strands Agent functionality with persona configurations
to create specialized review agents.
"""

from typing import Any
from strands.multiagent.base import MultiAgentResult
from strands.types.content import ContentBlock

from ..config.persona_loader import PersonaConfig
from .base_agent import BaseAgent


class ReviewAgent(BaseAgent):
    """A review agent that wraps Strands Agent with persona configuration."""

    def __init__(
        self,
        persona: PersonaConfig,
        model_provider: str | None = None,
        model_config_override: dict[str, Any] | None = None,
    ):
        """Initialize a review agent with a persona configuration.

        Args:
            persona: PersonaConfig object with agent settings
            model_provider: Optional model provider ('bedrock', 'lm_studio', 'ollama', etc.)
            model_config_override: Optional model configuration override
        """
        # Initialize the base agent
        super().__init__(persona, model_provider, model_config_override)

    async def review(self, content: str) -> str:
        """Review content using the configured persona.

        Args:
            content: The content to review

        Returns:
            Review feedback from the agent
        """
        # Check if we received an error instead of content
        if content == "ERROR_NO_CONTENT":
            return "No essay content was provided for review. Please submit your essay text for evaluation."

        # Check if tools are enabled for this persona
        tools_enabled = (
            hasattr(self, "persona_tools_config")
            and self.persona_tools_config
            and self.persona_tools_config.get("enabled", False)
        )

        if tools_enabled:
            # Enhanced review with objective analysis data
            # Extract clean essay content for tool analysis
            clean_content = self._extract_essay_content(content)
            analysis = self.get_content_analysis(clean_content)
            analysis_text = self.format_analysis_for_prompt(analysis)

            if analysis_text:
                # Include analysis data in the prompt
                enhanced_prompt = f"""
{self.persona.prompt_template.format(content=content)}

OBJECTIVE ANALYSIS DATA:
{analysis_text}

Use this objective data to inform your evaluation. The analysis provides precise measurements that enhance your expert assessment."""
                return await self.invoke_async_legacy(
                    enhanced_prompt, "enhanced_review"
                )

        # Standard review without tools
        prompt = self.persona.prompt_template.format(content=content)
        return await self.invoke_async_legacy(prompt, "review")

    async def invoke_async_graph(
        self, task: str | list[ContentBlock], **kwargs: Any
    ) -> MultiAgentResult:
        """Process task asynchronously for graph execution using specialized review logic.

        Args:
            task: Input content to process
            **kwargs: Additional arguments

        Returns:
            MultiAgentResult with review results
        """
        import time

        from strands.agent.agent_result import AgentResult
        from strands.multiagent.base import MultiAgentResult, NodeResult, Status
        from strands.telemetry.metrics import EventLoopMetrics
        from strands.types.content import ContentBlock, Message

        try:
            # Extract content from task using base agent logic
            content = self._extract_content_from_task(task)

            # Process using the specialized review method with timing
            start_time = time.time()
            response = await self.review(content)
            execution_time = int(time.time() - start_time)

            # Clean up any raw JSON that might have been generated in the response
            if response.startswith("{'role':") or response.startswith('{"role":'):
                try:
                    # Try to parse the JSON structure and extract the actual text
                    import ast

                    parsed = ast.literal_eval(response.replace("'", '"'))
                    if isinstance(parsed, dict) and "content" in parsed:
                        nested_content = parsed["content"]
                        if isinstance(nested_content, list) and len(nested_content) > 0:
                            if (
                                isinstance(nested_content[0], dict)
                                and "text" in nested_content[0]
                            ):
                                response = nested_content[0]["text"]
                except:
                    # If parsing fails, use the original response
                    pass

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
                                content=[ContentBlock(text=f"Review failed: {str(e)}")],
                            ),
                            metrics=EventLoopMetrics(),
                            state={
                                "agent_name": self.persona.name,
                                "agent_role": self.persona.role,
                                "error": str(e),
                                "response": f"Review failed: {str(e)}",
                            },
                        ),
                        status=Status.FAILED,
                    )
                },
                execution_time=0,
                execution_count=1,
            )
