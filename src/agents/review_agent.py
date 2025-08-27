"""
Review Agent implementation using Strands Agents.

This module wraps Strands Agent functionality with persona configurations
to create specialized review agents.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from strands import Agent
from ..config.persona_loader import PersonaConfig


class ReviewAgent:
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
        self.persona = persona
        self.model_provider = model_provider or "bedrock"  # Default to bedrock
        self.model_config_override = model_config_override or {}

        # Setup logging
        self._setup_agent_logging()

        # Create the Strands agent with persona configuration
        model = self._create_model()
        system_prompt = self._build_system_prompt()

        # Log the system prompt when the agent is created
        self._log_prompt(system_prompt, "system_prompt")

        self.agent = Agent(
            name=persona.name, model=model, system_prompt=system_prompt
        )

    def _setup_agent_logging(self):
        """Setup dedicated logging for this agent."""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create agent-specific logger
        agent_name = self.persona.name.replace(" ", "_").lower()
        self.logger_name = f"agent_{agent_name}"
        self.logger = logging.getLogger(self.logger_name)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # Create file handler for agent-specific log
            log_file = os.path.join(log_dir, f"{agent_name}_prompts.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)

    def _log_prompt(self, prompt: str, prompt_type: str = "review"):
        """Log the prompt to the dedicated agent log.
        
        Args:
            prompt: The prompt being sent to the agent
            prompt_type: Type of prompt (e.g., 'review', 'analysis')
        """
        try:
            # Log header with metadata
            header = f"[{prompt_type.upper()}] Prompt sent to {self.persona.name}"
            self.logger.info(header)
            
            # Log the clean prompt content that can be easily copy-pasted
            self.logger.info(prompt)
            
            # Log separator for readability
            self.logger.info("-" * 80)
        except Exception as e:
            # Fallback to print if logging fails
            print(f"Warning: Failed to log prompt for {self.persona.name}: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt from persona configuration."""
        return f"""You are a {self.persona.role}.

Your goal: {self.persona.goal}

Background: {self.persona.backstory}

Always provide constructive, specific feedback with actionable recommendations.
Be professional but thorough in your analysis."""

    def _create_model(self):
        """Create the appropriate model based on the provider."""
        model_config = self._get_model_config()

        if self.model_provider == "lm_studio":
            return self._create_lm_studio_model(model_config)
        elif self.model_provider == "ollama":
            return self._create_ollama_model(model_config)
        elif self.model_provider == "bedrock":
            return self._create_bedrock_model(model_config)
        else:
            # Default to bedrock
            return self._create_bedrock_model(model_config)

    def _create_lm_studio_model(self, config: Dict[str, Any]):
        """Create a model for LM Studio (OpenAI-compatible API)."""
        try:
            from strands.models.openai import OpenAIModel

            # LM Studio configuration for OpenAI-compatible API
            base_url = config.get("base_url", "http://localhost:1234/v1")
            model_id = config.get("model_id", "local-model")

            # Client arguments for the OpenAI client (for LM Studio)
            client_args = {
                "base_url": base_url,
                "api_key": "not-needed",  # LM Studio doesn't require API key
            }

            # Model configuration
            model_config = {
                "model_id": model_id,
            }

            # Add temperature and max_tokens if available
            if "temperature" in config:
                model_config["temperature"] = config["temperature"]
            if "max_tokens" in config:
                model_config["max_tokens"] = config["max_tokens"]

            print(f"✅ Creating LM Studio model with URL: {base_url}")
            return OpenAIModel(client_args=client_args, **model_config)

        except ImportError as e:
            print(f"⚠️  OpenAI model not available: {e}")
            print("💡 Install openai package: pip install openai")
            return None
        except Exception as e:
            print(f"⚠️  LM Studio model creation failed: {e}")
            print("💡 Make sure LM Studio is running at the specified URL.")
            return None

    def _create_ollama_model(self, config: Dict[str, Any]):
        """Create an Ollama model for local inference."""
        try:
            # Check if Ollama model is available
            from strands.models import OllamaModel

            ollama_config = {
                "model_id": config.get("model_id", "llama2"),  # Default Ollama model
                "base_url": config.get("base_url", "http://localhost:11434"),
            }

            # Add temperature and max_tokens if available
            if "temperature" in config:
                ollama_config["temperature"] = config["temperature"]
            if "max_tokens" in config:
                ollama_config["max_tokens"] = config["max_tokens"]

            return OllamaModel(**ollama_config)

        except ImportError:
            print(
                "⚠️  Ollama model not available. Install Ollama or use a different provider."
            )
            return None

    def _create_bedrock_model(self, config: Dict[str, Any]):
        """Create a Bedrock model for AWS inference."""
        try:
            from strands.models import BedrockModel

            return BedrockModel(**config)
        except ImportError:
            print("⚠️  Bedrock model not available.")
            return None

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from persona settings."""
        # Base configuration depends on provider
        if self.model_provider == "lm_studio":
            config = {
                "base_url": "http://localhost:1234/v1",  # LM Studio default
                "model_id": "local-model",  # Generic local model name
            }
        elif self.model_provider == "ollama":
            config = {
                "base_url": "http://localhost:11434",  # Ollama default
                "model_id": "llama2",  # Default Ollama model
            }
        else:  # bedrock
            config = {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"  # Default Bedrock model
            }

        # Apply persona model configuration
        if self.persona.model_config and isinstance(self.persona.model_config, dict):
            # Map persona config to model config
            if "temperature" in self.persona.model_config:
                config["temperature"] = self.persona.model_config["temperature"]
            if "max_tokens" in self.persona.model_config:
                config["max_tokens"] = self.persona.model_config["max_tokens"]
            # Allow overriding model_id and base_url from persona config
            if "model_id" in self.persona.model_config:
                config["model_id"] = self.persona.model_config["model_id"]
            if "base_url" in self.persona.model_config:
                config["base_url"] = self.persona.model_config["base_url"]

        # Apply any override configuration
        config.update(self.model_config_override)

        return config

    def review(self, content: str) -> str:
        """Review content using the configured persona.

        Args:
            content: The content to review

        Returns:
            Review feedback from the agent
        """
        # Format the prompt with the content
        prompt = self.persona.prompt_template.format(content=content)

        # Log the prompt to the dedicated agent log
        self._log_prompt(prompt, "review")

        # Get response from the Strands agent
        result = self.agent(prompt)

        # Extract the message content from the result
        if hasattr(result, "message"):
            return str(result.message)
        elif isinstance(result, dict):
            # Handle dictionary format
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and content:
                    return str(content[0])
                elif isinstance(content, str):
                    return content
            return str(result)
        else:
            return str(result)

    async def review_async(self, content: str) -> str:
        """Asynchronously review content using the configured persona.

        Args:
            content: The content to review

        Returns:
            Review feedback from the agent
        """
        # Format the prompt with the content
        prompt = self.persona.prompt_template.format(content=content)

        # Log the prompt to the dedicated agent log
        self._log_prompt(prompt, "review_async")

        # Get async response from the Strands agent
        result = await self.agent.invoke_async(prompt)

        # Extract the message content from the result
        if hasattr(result, "message"):
            return str(result.message)
        elif isinstance(result, dict):
            # Handle dictionary format
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and content:
                    return str(content[0])
                elif isinstance(content, str):
                    return content
            return str(result)
        else:
            return str(result)

    def get_info(self) -> Dict[str, Any]:
        """Get information about this review agent.

        Returns:
            Dictionary with agent information
        """
        # Safely get model config values
        temperature = 0.3
        max_tokens = 1500
        if self.persona.model_config and isinstance(self.persona.model_config, dict):
            temperature = self.persona.model_config.get("temperature", 0.3)
            max_tokens = self.persona.model_config.get("max_tokens", 1500)

        return {
            "name": self.persona.name,
            "role": self.persona.role,
            "goal": self.persona.goal,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
