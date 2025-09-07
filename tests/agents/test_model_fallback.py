"""Tests for model fallback behavior and enhanced logging."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
import io

from src.agents.base_agent import BaseAgent
from src.agents.review_agent import ReviewAgent
from src.agents.analysis_agent import AnalysisAgent
from src.config.persona_loader import PersonaConfig


class TestModelFallbackBehavior:
    """Test model fallback behavior for different providers."""

    @pytest.fixture
    def persona_without_model_id(self):
        """Create persona without model_id for fallback testing."""
        return PersonaConfig(
            name="Test Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test: {content}",
            model_config={
                "temperature": 0.5,
                "max_tokens": 1500
            }
        )

    @pytest.fixture
    def persona_with_reasoning_model(self):
        """Create persona with reasoning model."""
        return PersonaConfig(
            name="Test Reasoning Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-thinking-2507",
                "temperature": 0.3,
                "max_tokens": 2000,
                "max_context_length": 32768
            }
        )

    def test_lm_studio_fallback_no_model_id(self, persona_without_model_id):
        """Test LM Studio fallback behavior when no model_id specified."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
        config = agent._get_model_config()
        
        # Should not have model_id for LM Studio fallback
        assert "model_id" not in config
        assert config["base_url"] == "http://localhost:1234/v1"
        assert config["temperature"] == 0.5

    def test_bedrock_default_model(self, persona_without_model_id):
        """Test Bedrock uses default model when none specified."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="bedrock")
        config = agent._get_model_config()
        
        # Should have default Bedrock model
        assert config["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_ollama_default_model(self, persona_without_model_id):
        """Test Ollama uses default model when none specified."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="ollama")
        config = agent._get_model_config()
        
        # Should have default Ollama model
        assert config["model_id"] == "llama2"
        assert config["base_url"] == "http://localhost:11434"

    def test_lm_studio_model_creation_with_model_id(self, persona_with_reasoning_model):
        """Test LM Studio model creation with explicit model_id."""
        with patch('strands.models.openai.OpenAIModel') as mock_openai_model:
            agent = BaseAgent(persona=persona_with_reasoning_model, model_provider="lm_studio")
            
            # Mock the model creation to avoid actual API calls
            mock_model = Mock()
            mock_openai_model.return_value = mock_model
            
            model = agent._create_lm_studio_model(agent._get_model_config())
            
            # Should create model with specified model_id
            mock_openai_model.assert_called_once()
            call_kwargs = mock_openai_model.call_args[1]
            assert call_kwargs["model_id"] == "qwen/qwen3-4b-thinking-2507"

    def test_lm_studio_model_creation_without_model_id(self, persona_without_model_id):
        """Test LM Studio model creation without model_id (fallback)."""
        with patch('strands.models.openai.OpenAIModel') as mock_openai_model:
            agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
            
            # Mock the model creation to avoid actual API calls
            mock_model = Mock()
            mock_openai_model.return_value = mock_model
            
            model = agent._create_lm_studio_model(agent._get_model_config())
            
            # Should create model without model_id
            mock_openai_model.assert_called_once()
            call_kwargs = mock_openai_model.call_args[1]
            assert "model_id" not in call_kwargs

    def test_context_length_fallback_lm_studio(self, persona_without_model_id):
        """Test context length fallback for LM Studio without model_id."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
        context_length = agent.get_max_context_length()
        
        # Should use LM Studio fallback (8K)
        assert context_length == 8192

    def test_context_length_fallback_unknown_model(self, persona_without_model_id):
        """Test context length fallback for unknown model."""
        persona_without_model_id.model_config["model_id"] = "unknown-model"
        agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
        context_length = agent.get_max_context_length()
        
        # Should use conservative default (4K)
        assert context_length == 4096


class TestEnhancedLogging:
    """Test enhanced logging with model information."""

    @pytest.fixture
    def persona_with_model_info(self):
        """Create persona with complete model information."""
        return PersonaConfig(
            name="Test Logging Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-thinking-2507",
                "temperature": 0.3,
                "max_tokens": 2000,
                "max_context_length": 32768
            }
        )

    @pytest.fixture
    def persona_without_model_id(self):
        """Create persona without model_id for logging testing."""
        return PersonaConfig(
            name="Test Fallback Agent",
            role="Test Role",
            goal="Test goal",
            backstory="Test backstory",
            prompt_template="Test: {content}",
            model_config={
                "temperature": 0.5,
                "max_tokens": 1500
            }
        )

    def test_enhanced_logging_with_model_id(self, persona_with_model_info):
        """Test that logging includes model information."""
        agent = BaseAgent(persona=persona_with_model_info, model_provider="lm_studio")
        
        # Mock the logger to capture log messages
        with patch.object(agent, 'logger') as mock_logger:
            agent._log_prompt("test prompt", "test")
            
            # Check that model information was logged
            log_calls = mock_logger.info.call_args_list
            assert len(log_calls) >= 3  # Header, model info, config info
            
            # Find the model info log call
            model_info_call = None
            for call in log_calls:
                if "Model:" in str(call[0][0]):
                    model_info_call = call[0][0]
                    break
            
            assert model_info_call is not None
            assert "qwen/qwen3-4b-thinking-2507" in model_info_call
            assert "lm_studio" in model_info_call
            assert "32768" in model_info_call

    def test_enhanced_logging_without_model_id(self, persona_without_model_id):
        """Test that logging shows <default> for LM Studio without model_id."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
        
        # Mock the logger to capture log messages
        with patch.object(agent, 'logger') as mock_logger:
            agent._log_prompt("test prompt", "test")
            
            # Check that default model information was logged
            log_calls = mock_logger.info.call_args_list
            
            # Find the model info log call
            model_info_call = None
            for call in log_calls:
                if "Model:" in str(call[0][0]):
                    model_info_call = call[0][0]
                    break
            
            assert model_info_call is not None
            assert "<default>" in model_info_call
            assert "lm_studio" in model_info_call

    def test_enhanced_logging_temperature_and_tokens(self, persona_with_model_info):
        """Test that logging includes temperature and max_tokens."""
        agent = BaseAgent(persona=persona_with_model_info, model_provider="lm_studio")
        
        # Mock the logger to capture log messages
        with patch.object(agent, 'logger') as mock_logger:
            agent._log_prompt("test prompt", "test")
            
            # Check that temperature and tokens were logged
            log_calls = mock_logger.info.call_args_list
            
            # Find the config info log call
            config_info_call = None
            for call in log_calls:
                if "Temperature:" in str(call[0][0]):
                    config_info_call = call[0][0]
                    break
            
            assert config_info_call is not None
            assert "0.3" in config_info_call
            assert "2000" in config_info_call

    @patch('builtins.print')
    def test_console_output_with_model_id(self, mock_print, persona_with_model_info):
        """Test console output shows model creation with model_id."""
        agent = BaseAgent(persona=persona_with_model_info, model_provider="lm_studio")
        
        with patch('strands.models.openai.OpenAIModel'):
            agent._create_lm_studio_model(agent._get_model_config())
            
            # Check that console output was printed
            mock_print.assert_called()
            print_calls = [str(call) for call in mock_print.call_args_list]
            
            # Should show model creation with specific model
            model_creation_found = False
            for call in print_calls:
                if "Creating LM Studio model: qwen/qwen3-4b-thinking-2507" in call:
                    model_creation_found = True
                    break
            
            assert model_creation_found

    @patch('builtins.print')
    def test_console_output_without_model_id(self, mock_print, persona_without_model_id):
        """Test console output shows default model creation."""
        agent = BaseAgent(persona=persona_without_model_id, model_provider="lm_studio")
        
        with patch('strands.models.openai.OpenAIModel'):
            agent._create_lm_studio_model(agent._get_model_config())
            
            # Check that console output was printed
            mock_print.assert_called()
            print_calls = [str(call) for call in mock_print.call_args_list]
            
            # Should show default model creation
            default_creation_found = False
            for call in print_calls:
                if "Creating LM Studio model: <default>" in call:
                    default_creation_found = True
                    break
            
            assert default_creation_found

    def test_logging_with_different_providers(self, persona_with_model_info):
        """Test logging works correctly with different providers."""
        providers = ["lm_studio", "bedrock", "ollama"]
        
        for provider in providers:
            agent = BaseAgent(persona=persona_with_model_info, model_provider=provider)
            
            with patch.object(agent, 'logger') as mock_logger:
                agent._log_prompt("test prompt", "test")
                
                # Check that provider was logged
                log_calls = mock_logger.info.call_args_list
                
                # Find the model info log call
                model_info_call = None
                for call in log_calls:
                    if "Provider:" in str(call[0][0]):
                        model_info_call = call[0][0]
                        break
                
                assert model_info_call is not None
                assert provider in model_info_call


class TestAgentIntegrationWithModels:
    """Test agent integration with different model configurations."""

    @pytest.fixture
    def reasoning_persona(self):
        """Create persona configured for reasoning model."""
        return PersonaConfig(
            name="Reasoning Agent",
            role="Analysis Agent",
            goal="Complex analysis",
            backstory="Expert in reasoning",
            prompt_template="Analyze: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-thinking-2507",
                "temperature": 0.3,
                "max_tokens": 2000,
                "max_context_length": 32768
            }
        )

    @pytest.fixture
    def standard_persona(self):
        """Create persona configured for standard model."""
        return PersonaConfig(
            name="Standard Agent",
            role="Review Agent",
            goal="Basic review",
            backstory="Standard reviewer",
            prompt_template="Review: {content}",
            model_config={
                "model_id": "qwen/qwen3-4b-2507",
                "temperature": 0.5,
                "max_tokens": 1500,
                "max_context_length": 8192
            }
        )

    def test_review_agent_with_reasoning_model(self, reasoning_persona):
        """Test ReviewAgent with reasoning model configuration."""
        agent = ReviewAgent(persona=reasoning_persona, model_provider="lm_studio")
        
        # Mock the agent to avoid real model creation
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Reasoning response")
        agent._agent = mock_agent
        
        # Test configuration
        assert agent.get_max_context_length() == 32768
        config = agent._get_model_config()
        assert config["model_id"] == "qwen/qwen3-4b-thinking-2507"

    def test_review_agent_with_standard_model(self, standard_persona):
        """Test ReviewAgent with standard model configuration."""
        agent = ReviewAgent(persona=standard_persona, model_provider="lm_studio")
        
        # Mock the agent to avoid real model creation
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Standard response")
        agent._agent = mock_agent
        
        # Test configuration
        assert agent.get_max_context_length() == 8192
        config = agent._get_model_config()
        assert config["model_id"] == "qwen/qwen3-4b-2507"

    def test_analysis_agent_with_reasoning_model(self, reasoning_persona):
        """Test AnalysisAgent with reasoning model configuration."""
        agent = AnalysisAgent(persona=reasoning_persona, model_provider="lm_studio")
        
        # Mock the agent to avoid real model creation
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Analysis response")
        agent._agent = mock_agent
        
        # Test configuration
        assert agent.get_max_context_length() == 32768
        config = agent._get_model_config()
        assert config["model_id"] == "qwen/qwen3-4b-thinking-2507"

    @pytest.mark.asyncio
    async def test_context_length_validation_in_agents(self, standard_persona):
        """Test that agents validate context length automatically."""
        agent = ReviewAgent(persona=standard_persona, model_provider="lm_studio")
        
        # Mock the agent to avoid real model creation
        mock_agent = Mock()
        mock_agent.invoke_async = AsyncMock(return_value="Response")
        agent._agent = mock_agent
        
        # Test with large content that should trigger truncation
        large_content = "word " * 3000  # Should exceed 8192 - 1500 buffer
        
        with patch.object(agent, '_log_prompt') as mock_log:
            result = await agent.review(large_content)
            
            # Should complete successfully
            assert result == "Response"
            
            # Should have logged (possibly truncated) prompt
            mock_log.assert_called_once()
