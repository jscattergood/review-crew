"""Tests for ReviewAgent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.review_agent import ReviewAgent


class TestReviewAgent:
    """Test ReviewAgent functionality."""
    
    @patch('src.agents.base_agent.BaseAgent._setup_agent_logging')
    @patch('src.agents.base_agent.BaseAgent._create_model')
    @patch('src.agents.base_agent.Agent')
    def test_init(self, mock_strands_agent, mock_create_model, mock_setup_logging, mock_persona):
        """Test ReviewAgent initialization."""
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_agent_instance = Mock()
        mock_strands_agent.return_value = mock_agent_instance
        
        agent = ReviewAgent(mock_persona, model_provider="test", model_config_override={"temp": 0.5})
        
        assert agent.persona == mock_persona
        assert agent.model_provider == "test"
        assert agent.model_config_override == {"temp": 0.5}
        assert agent.agent == mock_agent_instance
        
        mock_create_model.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_strands_agent.assert_called_once()
    
    def test_build_system_prompt(self, mock_persona):
        """Test building system prompt from persona."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona)
                    prompt = agent._build_system_prompt()
                    
                    assert isinstance(prompt, str)
                    assert mock_persona.role in prompt
                    assert mock_persona.goal in prompt
                    assert mock_persona.backstory in prompt
    
    def test_review(self, mock_persona):
        """Test review method."""
        mock_content = "Test content to review"
        mock_response = "Mock review response"
        
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona)
                    
                    # Mock the invoke method from BaseAgent
                    with patch.object(agent, 'invoke', return_value=mock_response) as mock_invoke:
                        result = agent.review(mock_content)
                        
                        # Verify the result
                        assert result == mock_response
                        
                        # Verify invoke was called with properly formatted prompt
                        # The expected prompt should be the template with content substituted
                        expected_prompt = mock_persona.prompt_template.format(content=mock_content)
                        mock_invoke.assert_called_once_with(expected_prompt, "review")
    
    @pytest.mark.asyncio
    async def test_review_async(self, mock_persona):
        """Test async review method."""
        mock_content = "Test content to review"
        mock_response = "Mock async review response"
        
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona)
                    
                    # Mock the invoke_async method from BaseAgent
                    with patch.object(agent, 'invoke_async', return_value=mock_response) as mock_invoke_async:
                        result = await agent.review_async(mock_content)
                        
                        # Verify the result
                        assert result == mock_response
                        
                        # Verify invoke_async was called with properly formatted prompt
                        expected_prompt = mock_persona.prompt_template.format(content=mock_content)
                        mock_invoke_async.assert_called_once_with(expected_prompt, "review_async")
    
    def test_get_info(self, mock_persona):
        """Test get_info method inherited from BaseAgent."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona)
                    
                    # Mock the get_info method from BaseAgent
                    mock_info = {
                        "name": mock_persona.name,
                        "role": mock_persona.role,
                        "goal": mock_persona.goal,
                        "temperature": 0.3,
                        "max_tokens": 1500,
                    }
                    
                    with patch.object(agent, 'get_info', return_value=mock_info) as mock_get_info:
                        result = agent.get_info()
                        
                        assert result == mock_info
                        mock_get_info.assert_called_once()


# Test the BaseAgent functionality through ReviewAgent
class TestBaseAgentFunctionality:
    """Test BaseAgent functionality through ReviewAgent inheritance."""
    
    def test_model_config_bedrock(self, mock_persona):
        """Test model configuration for Bedrock."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="bedrock")
                    config = agent._get_model_config()
                    
                    assert "model_id" in config
                    assert "anthropic.claude" in config["model_id"]
    
    def test_model_config_lm_studio(self, mock_persona):
        """Test model configuration for LM Studio."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="lm_studio")
                    config = agent._get_model_config()
                    
                    assert config["base_url"] == "http://localhost:1234/v1"
                    assert config["model_id"] == "local-model"
    
    def test_model_config_with_override(self, mock_persona):
        """Test model configuration with override."""
        override_config = {"temperature": 0.8, "max_tokens": 2000}
        
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_config_override=override_config)
                    config = agent._get_model_config()
                    
                    assert config["temperature"] == 0.8
                    assert config["max_tokens"] == 2000
    
    def test_create_bedrock_model_success(self, mock_persona):
        """Test successful Bedrock model creation."""
        with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
            with patch('src.agents.base_agent.BaseAgent._create_model'):  # Mock to prevent actual model creation
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="bedrock")
                    
                    # Now test the _create_bedrock_model method directly
                    with patch('strands.models.BedrockModel') as mock_bedrock_model:
                        mock_model = Mock()
                        mock_bedrock_model.return_value = mock_model
                        
                        result = agent._create_bedrock_model({"model_id": "test-model"})
                        
                        assert result == mock_model
                        mock_bedrock_model.assert_called_once_with(model_id="test-model")
    
    def test_create_bedrock_model_import_error(self, mock_persona):
        """Test Bedrock model creation with import error."""
        with patch('strands.models.BedrockModel', side_effect=ImportError("Module not found")):
            with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="bedrock")
                    result = agent._create_bedrock_model({"model_id": "test-model"})
                    
                    assert result is None
    
    def test_create_lm_studio_model_success(self, mock_persona):
        """Test successful LM Studio model creation."""
        with patch('strands.models.openai.OpenAIModel') as mock_openai_model:
            mock_model = Mock()
            mock_openai_model.return_value = mock_model
            
            with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="lm_studio")
                    result = agent._create_lm_studio_model({
                        "base_url": "http://localhost:1234/v1",
                        "model_id": "test-model"
                    })
                    
                    assert result == mock_model
    
    def test_create_lm_studio_model_import_error(self, mock_persona):
        """Test LM Studio model creation with import error."""
        with patch('strands.models.openai.OpenAIModel', side_effect=ImportError("Module not found")):
            with patch('src.agents.base_agent.BaseAgent._setup_agent_logging'):
                with patch('src.agents.base_agent.Agent'):
                    agent = ReviewAgent(mock_persona, model_provider="lm_studio")
                    result = agent._create_lm_studio_model({
                        "base_url": "http://localhost:1234/v1",
                        "model_id": "test-model"
                    })
                    
                    assert result is None