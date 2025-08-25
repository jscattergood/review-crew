"""Tests for ReviewAgent."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.agents.review_agent import ReviewAgent


class TestReviewAgent:
    """Test ReviewAgent functionality."""
    
    @patch('src.agents.review_agent.ReviewAgent._create_model')
    @patch('src.agents.review_agent.Agent')
    def test_init(self, mock_strands_agent, mock_create_model, mock_persona):
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
        mock_strands_agent.assert_called_once_with(
            name=mock_persona.name,
            model=mock_model,
            system_prompt=agent._build_system_prompt()
        )
    
    def test_build_system_prompt(self, mock_persona):
        """Test building system prompt from persona."""
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent'):
                agent = ReviewAgent(mock_persona)
                prompt = agent._build_system_prompt()
                
                assert mock_persona.role in prompt
                assert mock_persona.goal in prompt
                assert mock_persona.backstory in prompt
                assert mock_persona.prompt_template in prompt
    
    @patch('src.agents.review_agent.ReviewAgent._get_model_config')
    @patch('src.agents.review_agent.ReviewAgent._create_bedrock_model')
    def test_create_model_bedrock(self, mock_create_bedrock, mock_get_config, mock_persona):
        """Test creating Bedrock model."""
        mock_config = {"model_id": "test-model"}
        mock_get_config.return_value = mock_config
        mock_model = Mock()
        mock_create_bedrock.return_value = mock_model
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona, model_provider="bedrock")
            model = agent._create_model()
            
            assert model == mock_model
            # Method is called during init and then again in test
            assert mock_create_bedrock.call_count == 2
            mock_create_bedrock.assert_called_with(mock_config)
    
    @patch('src.agents.review_agent.ReviewAgent._get_model_config')
    @patch('src.agents.review_agent.ReviewAgent._create_lm_studio_model')
    def test_create_model_lm_studio(self, mock_create_lm_studio, mock_get_config, mock_persona):
        """Test creating LM Studio model."""
        mock_config = {"base_url": "http://localhost:1234/v1"}
        mock_get_config.return_value = mock_config
        mock_model = Mock()
        mock_create_lm_studio.return_value = mock_model
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona, model_provider="lm_studio")
            model = agent._create_model()
            
            assert model == mock_model
            # Method is called during init and then again in test
            assert mock_create_lm_studio.call_count == 2
            mock_create_lm_studio.assert_called_with(mock_config)
    
    def test_get_model_config_bedrock(self, mock_persona):
        """Test getting model config for Bedrock."""
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent'):
                agent = ReviewAgent(mock_persona, model_provider="bedrock")
                config = agent._get_model_config()
                
                assert 'model_id' in config
                assert config['model_id'] == 'anthropic.claude-3-sonnet-20240229-v1:0'
                assert config['temperature'] == mock_persona.model_config['temperature']
                assert config['max_tokens'] == mock_persona.model_config['max_tokens']
    
    def test_get_model_config_lm_studio(self, mock_persona):
        """Test getting model config for LM Studio."""
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent'):
                agent = ReviewAgent(mock_persona, model_provider="lm_studio")
                config = agent._get_model_config()
                
                assert 'base_url' in config
                assert config['base_url'] == 'http://localhost:1234/v1'
                assert config['model_id'] == 'local-model'
                assert config['temperature'] == mock_persona.model_config['temperature']
                assert config['max_tokens'] == mock_persona.model_config['max_tokens']
    
    def test_get_model_config_with_override(self, mock_persona):
        """Test getting model config with override."""
        override = {"temperature": 0.8, "custom_param": "test"}
        
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent'):
                agent = ReviewAgent(mock_persona, model_config_override=override)
                config = agent._get_model_config()
                
                assert config['temperature'] == 0.8  # Override value
                assert config['custom_param'] == "test"  # New parameter
                assert config['max_tokens'] == mock_persona.model_config['max_tokens']  # Original value
    
    def test_review(self, mock_persona, sample_content):
        """Test synchronous review."""
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.message = "Test review response"
        mock_agent.return_value = mock_result
        
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent', return_value=mock_agent):
                agent = ReviewAgent(mock_persona)
                result = agent.review(sample_content)
                
                assert result == "Test review response"
                mock_agent.assert_called_once()
                # Check that the prompt was formatted with content
                call_args = mock_agent.call_args[0][0]
                assert sample_content in call_args
    
    @pytest.mark.asyncio
    async def test_review_async(self, mock_persona, sample_content):
        """Test asynchronous review."""
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.message = "Test async review response"
        mock_agent.invoke_async = AsyncMock(return_value=mock_result)
        
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent', return_value=mock_agent):
                agent = ReviewAgent(mock_persona)
                result = await agent.review_async(sample_content)
                
                assert result == "Test async review response"
                mock_agent.invoke_async.assert_called_once()
                # Check that the prompt was formatted with content
                call_args = mock_agent.invoke_async.call_args[0][0]
                assert sample_content in call_args
    
    def test_get_info(self, mock_persona):
        """Test getting agent information."""
        with patch('src.agents.review_agent.ReviewAgent._create_model'):
            with patch('src.agents.review_agent.Agent'):
                agent = ReviewAgent(mock_persona)
                info = agent.get_info()
                
                assert info['name'] == mock_persona.name
                assert info['role'] == mock_persona.role
                assert info['goal'] == mock_persona.goal
                assert info['temperature'] == mock_persona.model_config['temperature']
                assert info['max_tokens'] == mock_persona.model_config['max_tokens']
    
    @patch('strands.models.BedrockModel')
    def test_create_bedrock_model_success(self, mock_bedrock_model, mock_persona):
        """Test successful Bedrock model creation."""
        mock_model = Mock()
        mock_bedrock_model.return_value = mock_model
        config = {"model_id": "test-model"}
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona)
            result = agent._create_bedrock_model(config)
            
            assert result == mock_model
            # Method is called during init and then again in test
            assert mock_bedrock_model.call_count == 2
            mock_bedrock_model.assert_called_with(**config)
    
    @patch('strands.models.BedrockModel', side_effect=ImportError("Bedrock not available"))
    def test_create_bedrock_model_import_error(self, mock_bedrock_model, mock_persona):
        """Test Bedrock model creation with import error."""
        config = {"model_id": "test-model"}
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona)
            result = agent._create_bedrock_model(config)
            
            assert result is None
    
    @patch('strands.models.openai.OpenAIModel')
    def test_create_lm_studio_model_success(self, mock_openai_model, mock_persona):
        """Test successful LM Studio model creation."""
        mock_model = Mock()
        mock_openai_model.return_value = mock_model
        config = {"base_url": "http://localhost:1234/v1", "model_id": "local-model"}
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona)
            result = agent._create_lm_studio_model(config)
            
            assert result == mock_model
            mock_openai_model.assert_called_once()
    
    @patch('strands.models.openai.OpenAIModel', side_effect=ImportError("OpenAI not available"))
    def test_create_lm_studio_model_import_error(self, mock_openai_model, mock_persona):
        """Test LM Studio model creation with import error."""
        config = {"base_url": "http://localhost:1234/v1", "model_id": "local-model"}
        
        with patch('src.agents.review_agent.Agent'):
            agent = ReviewAgent(mock_persona)
            result = agent._create_lm_studio_model(config)
            
            assert result is None
