"""
Review Agent implementation using Strands Agents.

This module wraps Strands Agent functionality with persona configurations
to create specialized review agents.
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from ..config.persona_loader import PersonaConfig


class ReviewAgent(BaseAgent):
    """A review agent that wraps Strands Agent with persona configuration."""

    def __init__(
        self,
        persona: PersonaConfig,
        model_provider: Optional[str] = None,
        model_config_override: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a review agent with a persona configuration.

        Args:
            persona: PersonaConfig object with agent settings
            model_provider: Optional model provider ('bedrock', 'lm_studio', 'ollama', etc.)
            model_config_override: Optional model configuration override
        """
        # Initialize the base agent
        super().__init__(persona, model_provider, model_config_override)

    def review(self, content: str) -> str:
        """Review content using the configured persona.

        Args:
            content: The content to review

        Returns:
            Review feedback from the agent
        """
        # Format the prompt with the content
        prompt = self.persona.prompt_template.format(content=content)

        # Use the base agent's invoke method which handles logging
        return self.invoke(prompt, "review")

    async def review_async(self, content: str) -> str:
        """Asynchronously review content using the configured persona.

        Args:
            content: The content to review

        Returns:
            Review feedback from the agent
        """
        # Format the prompt with the content
        prompt = self.persona.prompt_template.format(content=content)

        # Use the base agent's invoke_async method which handles logging
        return await self.invoke_async(prompt, "review_async")