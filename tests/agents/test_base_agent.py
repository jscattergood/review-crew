"""Tests for BaseAgent class context management and model configuration."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.base_agent import BaseAgent
from src.config.persona_loader import PersonaConfig


class TestBaseAgent:
    """Test cases for BaseAgent context management and model configuration."""

    @pytest.fixture
    def mock_persona_with_model_id(self):
        """Create a mock persona config with model_id."""
        persona = PersonaConfig(
            name="Test Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test prompt: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-thinking-2507",
                "temperature": 0.3,
                "max_tokens": 2000,
                "max_context_length": 32768,
            },
        )
        return persona

    @pytest.fixture
    def mock_persona_without_model_id(self):
        """Create a mock persona config without model_id."""
        persona = PersonaConfig(
            name="Test Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test prompt: {content}",
            model_config={"temperature": 0.5, "max_tokens": 1500},
        )
        return persona

    @pytest.fixture
    def mock_persona_standard_model(self):
        """Create a mock persona config with standard model."""
        persona = PersonaConfig(
            name="Test Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test prompt: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-2507",
                "temperature": 0.5,
                "max_tokens": 1500,
                "max_context_length": 8192,
            },
        )
        return persona

    @pytest.fixture
    def agent_with_reasoning_model(self, mock_persona_with_model_id):
        """Create a BaseAgent with reasoning model config."""
        agent = BaseAgent(
            persona=mock_persona_with_model_id, model_provider="lm_studio"
        )

        # Mock the agent property to avoid creating real models
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Test response")
        agent._agent = mock_agent

        return agent

    @pytest.fixture
    def agent_with_standard_model(self, mock_persona_standard_model):
        """Create a BaseAgent with standard model config."""
        agent = BaseAgent(
            persona=mock_persona_standard_model, model_provider="lm_studio"
        )

        # Mock the agent property to avoid creating real models
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Test response")
        agent._agent = mock_agent

        return agent

    @pytest.fixture
    def agent_without_model_id(self, mock_persona_without_model_id):
        """Create a BaseAgent without model_id (fallback behavior)."""
        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="lm_studio"
        )

        # Mock the agent property to avoid creating real models
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Test response")
        agent._agent = mock_agent

        return agent

    def test_get_max_context_length_with_explicit_config(
        self, agent_with_reasoning_model
    ):
        """Test get_max_context_length with explicit max_context_length in persona."""
        context_length = agent_with_reasoning_model.get_max_context_length()
        assert context_length == 32768

    def test_get_max_context_length_standard_model(self, agent_with_standard_model):
        """Test get_max_context_length with standard model."""
        context_length = agent_with_standard_model.get_max_context_length()
        assert context_length == 8192

    def test_get_max_context_length_reasoning_model_detection(
        self, mock_persona_without_model_id
    ):
        """Test automatic context length detection for reasoning model."""
        # Override model_config to have reasoning model without explicit context length
        mock_persona_without_model_id.model_config = {
            "model_id": "qwen/qwen3-4b-thinking-2507",
            "temperature": 0.3,
        }

        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="lm_studio"
        )
        context_length = agent.get_max_context_length()

        # Should detect reasoning model and use 32K context
        assert context_length == 32768

    def test_get_max_context_length_standard_model_detection(
        self, mock_persona_without_model_id
    ):
        """Test automatic context length detection for standard model."""
        # Override model_config to have standard model without explicit context length
        mock_persona_without_model_id.model_config = {
            "model_id": "qwen/qwen3-4b-2507",
            "temperature": 0.3,
        }

        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="lm_studio"
        )
        context_length = agent.get_max_context_length()

        # Should detect standard model and use 8K context
        assert context_length == 8192

    def test_get_max_context_length_lm_studio_fallback(self, agent_without_model_id):
        """Test fallback context length for LM Studio without model_id."""
        context_length = agent_without_model_id.get_max_context_length()

        # Should use LM Studio fallback (8K)
        assert context_length == 8192

    def test_get_max_context_length_claude_model(self, mock_persona_without_model_id):
        """Test context length detection for Claude model."""
        mock_persona_without_model_id.model_config = {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        }

        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="bedrock"
        )
        context_length = agent.get_max_context_length()

        # Should detect Claude and use 200K context
        assert context_length == 200000

    def test_get_max_context_length_gpt4_model(self, mock_persona_without_model_id):
        """Test context length detection for GPT-4 model."""
        mock_persona_without_model_id.model_config = {"model_id": "gpt-4-turbo"}

        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="lm_studio"
        )
        context_length = agent.get_max_context_length()

        # Should detect GPT-4 and use 128K context
        assert context_length == 128000

    def test_get_max_context_length_unknown_model(self, mock_persona_without_model_id):
        """Test fallback context length for unknown model."""
        mock_persona_without_model_id.model_config = {"model_id": "unknown-model-123"}

        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="lm_studio"
        )
        context_length = agent.get_max_context_length()

        # Should use conservative default (4K)
        assert context_length == 4096

    def test_count_tokens_with_tiktoken(self, agent_with_reasoning_model):
        """Test token counting with tiktoken available."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            # Mock tiktoken
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoding

            token_count = agent_with_reasoning_model._count_tokens("test text")

            assert token_count == 5
            mock_get_encoding.assert_called_once_with("cl100k_base")
            mock_encoding.encode.assert_called_once_with("test text")

    def test_count_tokens_without_tiktoken(self, agent_with_reasoning_model):
        """Test token counting fallback without tiktoken."""

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            token_count = agent_with_reasoning_model._count_tokens(
                "test text with 20 chars"
            )

            # Should use character-based fallback (20 chars / 4 = 5 tokens)
            assert token_count == 5

    def test_truncate_prompt_with_tiktoken(self, agent_with_reasoning_model):
        """Test prompt truncation with tiktoken."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            # Mock tiktoken
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]  # 10 tokens
            mock_encoding.decode.return_value = "truncated text"
            mock_get_encoding.return_value = mock_encoding

            truncated = agent_with_reasoning_model._truncate_prompt(
                "long text", max_tokens=5
            )

            assert "truncated text" in truncated
            assert (
                "[... content truncated due to context length limit ...]" in truncated
            )
            mock_encoding.encode.assert_called_once_with("long text")
            mock_encoding.decode.assert_called_once_with([1, 2, 3, 4, 5])

    def test_truncate_prompt_without_tiktoken(self, agent_with_reasoning_model):
        """Test prompt truncation fallback without tiktoken."""

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            long_text = "a" * 100  # 100 characters
            truncated = agent_with_reasoning_model._truncate_prompt(
                long_text, max_tokens=10
            )

            # Should truncate to approximately 40 characters (10 tokens * 4 chars/token)
            # Allow for small variation due to implementation details
            truncated_content = truncated.split("[... content truncated")[0]
            assert 38 <= len(truncated_content) <= 42
            assert (
                "[... content truncated due to context length limit ...]" in truncated
            )

    def test_truncate_prompt_no_truncation_needed(self, agent_with_reasoning_model):
        """Test prompt truncation when no truncation is needed."""

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            short_text = "short"
            result = agent_with_reasoning_model._truncate_prompt(
                short_text, max_tokens=10
            )

            # Should return original text
            assert result == short_text

    def test_should_chunk_content_true(self, agent_with_standard_model):
        """Test should_chunk_content returns True for large content."""

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("No module named 'tiktoken'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Standard model: 8192 context - 1000 response - 500 system = 6692 available tokens
            # Create content that exceeds this limit (using character fallback: 6692 * 4 = 26768 chars)
            large_content = (
                "a" * 30000
            )  # 30,000 characters, definitely exceeds available tokens

            should_chunk = agent_with_standard_model.should_chunk_content(large_content)
            assert should_chunk is True

    def test_should_chunk_content_false(self, agent_with_reasoning_model):
        """Test should_chunk_content returns False for small content."""
        small_content = "small content"

        should_chunk = agent_with_reasoning_model.should_chunk_content(small_content)
        assert should_chunk is False

    @pytest.mark.asyncio
    async def test_invoke_async_legacy_with_truncation(self, agent_with_standard_model):
        """Test invoke_async_legacy with content that needs truncation."""
        # Create content that exceeds context limit
        large_prompt = "word " * 3000  # Should exceed available tokens

        with patch.object(agent_with_standard_model, "_log_prompt") as mock_log:
            result = await agent_with_standard_model.invoke_async_legacy(large_prompt)

            # Should complete without error
            assert result == "Test response"

            # Should have logged the prompt (possibly truncated)
            mock_log.assert_called_once()
            logged_prompt = mock_log.call_args[0][0]

            # Logged prompt should be shorter than original if truncated
            assert len(logged_prompt) <= len(large_prompt)

    @pytest.mark.asyncio
    async def test_invoke_async_legacy_no_truncation(self, agent_with_reasoning_model):
        """Test invoke_async_legacy with content that doesn't need truncation."""
        normal_prompt = "normal sized prompt"

        with patch.object(agent_with_reasoning_model, "_log_prompt") as mock_log:
            result = await agent_with_reasoning_model.invoke_async_legacy(normal_prompt)

            # Should complete without error
            assert result == "Test response"

            # Should have logged the original prompt
            mock_log.assert_called_once()
            logged_prompt = mock_log.call_args[0][0]
            assert logged_prompt == normal_prompt

    def test_get_model_config_with_model_id(self, agent_with_reasoning_model):
        """Test _get_model_config includes model_id from persona."""
        config = agent_with_reasoning_model._get_model_config()

        assert config["model_id"] == "qwen/qwen3-4b-thinking-2507"
        assert config["temperature"] == 0.3
        assert config["max_tokens"] == 2000
        assert config["base_url"] == "http://localhost:1234/v1"

    def test_get_model_config_without_model_id(self, agent_without_model_id):
        """Test _get_model_config without model_id (LM Studio fallback)."""
        config = agent_without_model_id._get_model_config()

        # Should not have model_id for LM Studio fallback
        assert "model_id" not in config
        assert config["base_url"] == "http://localhost:1234/v1"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 1500

    def test_get_model_config_bedrock_provider(self, mock_persona_without_model_id):
        """Test _get_model_config with Bedrock provider."""
        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="bedrock"
        )
        config = agent._get_model_config()

        # Should have default Bedrock model
        assert config["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert "base_url" not in config  # Bedrock doesn't use base_url

    def test_get_model_config_ollama_provider(self, mock_persona_without_model_id):
        """Test _get_model_config with Ollama provider."""
        agent = BaseAgent(
            persona=mock_persona_without_model_id, model_provider="ollama"
        )
        config = agent._get_model_config()

        # Should have default Ollama model
        assert config["model_id"] == "llama2"
        assert config["base_url"] == "http://localhost:11434"

    def test_model_config_override(self, mock_persona_with_model_id):
        """Test model configuration override."""
        override_config = {"model_id": "override-model", "temperature": 0.8}

        agent = BaseAgent(
            persona=mock_persona_with_model_id,
            model_provider="lm_studio",
            model_config_override=override_config,
        )

        config = agent._get_model_config()

        # Override should take precedence
        assert config["model_id"] == "override-model"
        assert config["temperature"] == 0.8
        # Original persona config should still be present for non-overridden values
        assert config["max_tokens"] == 2000

    @pytest.fixture
    def mock_persona_with_tools(self):
        """Create a mock persona config with tools enabled."""
        persona = PersonaConfig(
            name="Test Agent with Tools",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test prompt: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-thinking-2507",
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            tools_config={
                "enabled": True,
                "analysis_types": ["metrics", "constraints", "structure"],
            },
        )
        return persona

    @pytest.fixture
    def agent_with_tools(self, mock_persona_with_tools):
        """Create a BaseAgent with tools enabled."""
        agent = BaseAgent(persona=mock_persona_with_tools, model_provider="lm_studio")

        # Mock the agent property
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Test response")
        agent._agent = mock_agent

        return agent

    def test_init_with_tools_config(self, agent_with_tools):
        """Test BaseAgent initialization with tools config."""
        assert agent_with_tools.persona_tools_config is not None
        assert agent_with_tools.persona_tools_config["enabled"] is True
        assert "metrics" in agent_with_tools.persona_tools_config["analysis_types"]

    def test_init_without_tools_config(self, agent_with_reasoning_model):
        """Test BaseAgent initialization without tools config."""
        assert agent_with_reasoning_model.persona_tools_config is None

    def test_tools_registration_with_config(self, agent_with_tools):
        """Test that tools are registered when tools_config is enabled."""
        # Mock the tools import to avoid actual tool loading in tests
        with patch.object(agent_with_tools, "_register_writing_tools") as mock_register:
            # Re-initialize to trigger tools registration
            agent_with_tools.__init__(
                agent_with_tools.persona, agent_with_tools.model_provider
            )
            mock_register.assert_called_once()

    def test_tools_not_registered_without_config(self, agent_with_reasoning_model):
        """Test that tools are not registered when tools_config is disabled and enable_tools=False."""
        with patch.object(
            agent_with_reasoning_model, "_register_writing_tools"
        ) as mock_register:
            # Re-initialize with enable_tools=False to check tools registration
            agent_with_reasoning_model.__init__(
                agent_with_reasoning_model.persona,
                agent_with_reasoning_model.model_provider,
                enable_tools=False,
            )
            mock_register.assert_not_called()
