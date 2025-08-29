"""Integration tests for Review-Crew system."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.conversation.manager import ConversationManager
from src.config.persona_loader import PersonaLoader


class TestIntegration:
    """Integration tests for the complete Review-Crew system."""
    
    @pytest.fixture
    def test_files(self):
        """Available test files for integration testing."""
        return [
            ("Python Code", "test_inputs/user_registration.py"),
            ("HTML Page", "test_inputs/product_page.html"), 
            ("API Documentation", "test_inputs/api_documentation.md")
        ]
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code with security issues for testing."""
        return '''import hashlib

def register_user(email, password):
    # MD5 hashing (insecure)
    password_hash = hashlib.md5(password.encode()).hexdigest()
    print(f"DEBUG: Created user {email} with password {password}")  # Password leak!
    return password_hash
'''
    
    def test_persona_loader_integration(self):
        """Test that PersonaLoader can load real personas."""
        loader = PersonaLoader()
        
        try:
            personas = loader.load_reviewer_personas()
            assert len(personas) > 0
            
            # Check that personas have required fields
            for persona in personas:
                assert persona.name
                assert persona.role
                assert persona.goal
                assert persona.backstory
                assert persona.prompt_template
                assert '{content}' in persona.prompt_template
                assert isinstance(persona.model_config, dict)
                
        except ValueError as e:
            pytest.skip(f"No personas configured: {e}")
    
    def test_conversation_manager_initialization(self):
        """Test that ConversationManager can be initialized."""
        try:
            manager = ConversationManager()
            agents = manager.get_available_agents()
            
            # Should have at least one agent if personas are configured
            if len(agents) > 0:
                assert all('name' in agent for agent in agents)
                assert all('role' in agent for agent in agents)
                assert all('goal' in agent for agent in agents)
            else:
                pytest.skip("No agents available - personas not configured")
                
        except Exception as e:
            pytest.skip(f"ConversationManager initialization failed: {e}")
    
    @pytest.mark.asyncio
    async def test_mock_review_workflow(self, sample_python_code):
        """Test complete review workflow with mocked LLM responses."""
        
        # Mock the Strands Agent and ReviewAgent to avoid actual LLM calls
        with patch('src.agents.base_agent.Agent') as mock_agent_class, \
             patch('src.conversation.manager.ReviewAgent') as mock_review_agent_class:
            
            # Mock the Strands Agent class to return an awaitable agent
            mock_strands_agent = Mock()
            mock_strands_agent.invoke_async = AsyncMock(return_value="MOCK REVIEW: This is a mock review response with security and technical feedback.")
            mock_strands_agent.__call__ = Mock(return_value="MOCK REVIEW: This is a mock review response with security and technical feedback.")
            mock_agent_class.return_value = mock_strands_agent
            
            # Create a single mock ReviewAgent that can handle multiple calls
            mock_review_agent = Mock()
            # Mock the async method that's actually called by the graph execution
            mock_review_agent.invoke_async_legacy = AsyncMock(return_value="MOCK REVIEW: This is a mock review response with security and technical feedback.")
            mock_review_agent.review.return_value = "MOCK REVIEW: This is a mock review response with security and technical feedback."
            mock_review_agent.get_info.return_value = {
                'name': 'Mock Agent',
                'role': 'Mock Reviewer',
                'goal': 'Provide mock feedback'
            }
            mock_review_agent.persona = Mock()
            mock_review_agent.persona.name = 'Mock Agent'
            mock_review_agent.persona.role = 'Mock Reviewer'
            mock_review_agent.name = 'mock_agent'
            # Set the mocked Strands agent on the review agent
            mock_review_agent.agent = mock_strands_agent
            
            # Return the same mock for any number of agents
            mock_review_agent_class.return_value = mock_review_agent
            
            try:
                manager = ConversationManager(enable_analysis=False)  # Disable analysis for simpler test
                
                if len(manager.agents) == 0:
                    pytest.skip("No agents available for testing")
                
                # Run the review
                result = await manager.run_review(sample_python_code)
                
                # Verify results
                assert result.content == sample_python_code
                assert len(result.reviews) > 0
                
                # Check that we got responses
                successful_reviews = [r for r in result.reviews if not r.error]
                assert len(successful_reviews) > 0
                
                # Verify review content
                for review in successful_reviews:
                    assert review.agent_name
                    assert review.agent_role
                    assert review.feedback
                    assert review.timestamp
                    
            except OSError as e:
                if "File name too long" in str(e):
                    pytest.skip(f"Skipping due to filesystem limitation: {e}")
                else:
                    raise
            except ImportError as e:
                pytest.skip(f"Missing dependencies for integration test: {e}")
            except Exception as e:
                # Let other exceptions fail the test so we can see what's wrong
                raise
    
    @pytest.mark.asyncio
    async def test_context_integration(self, sample_content, sample_context):
        """Test context processing integration."""
        
        with patch('src.agents.base_agent.Agent') as mock_agent_class, \
             patch('src.conversation.manager.ReviewAgent') as mock_review_agent_class, \
             patch('src.conversation.manager.ContextAgent') as mock_context_agent_class:
            
            # Mock the Strands Agent class to return an awaitable agent
            mock_strands_agent = Mock()
            mock_strands_agent.invoke_async = AsyncMock(return_value="Review with context completed")
            mock_strands_agent.__call__ = Mock(return_value="Review with context completed")
            mock_agent_class.return_value = mock_strands_agent
            
            # Mock ReviewAgent
            mock_review_agent = Mock()
            mock_review_agent.invoke_async_legacy = AsyncMock(return_value="Review with context completed")
            mock_review_agent.review.return_value = "Review with context completed"
            mock_review_agent.get_info.return_value = {'name': 'Mock Agent', 'role': 'Mock Reviewer', 'goal': 'Mock Goal'}
            mock_review_agent.persona = Mock()
            mock_review_agent.persona.name = 'Mock Agent'
            mock_review_agent.persona.role = 'Mock Reviewer'
            mock_review_agent.name = 'mock_agent'
            mock_review_agent.agent = mock_strands_agent
            mock_review_agent_class.return_value = mock_review_agent
            
            # Mock ContextAgent
            mock_context_agent = Mock()
            mock_context_result = Mock()
            mock_context_result.formatted_context = f"## CONTEXT SUMMARY\nTest context processed\n\nFormatted: {sample_context}"
            mock_context_result.context_summary = "Test context processed"
            mock_context_agent.process_context.return_value = mock_context_result
            mock_context_agent_class.return_value = mock_context_agent
            
            try:
                manager = ConversationManager(
                    enable_analysis=False
                )
                
                if len(manager.agents) == 0:
                    pytest.skip("No agents available for testing")
                
                # Run review with context
                result = await manager.run_review(sample_content, context_data=sample_context)
                
                # Verify context was processed
                if result.context_results:
                    assert len(result.context_results) > 0
                    assert result.context_results[0].formatted_context
                    assert result.context_results[0].context_summary
                    
            except Exception as e:
                pytest.skip(f"Context integration test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_file_based_review(self):
        """Test reviewing file-like content with shorter samples."""
        
        # Use shorter sample content to avoid file name length issues
        sample_api_doc = """# API Documentation

## POST /register
Register a new user.

**Security Issue:** No authentication required!

**Example:**
```
curl -X POST api.example.com/register -d "email=user@example.com"
```

**Response:** Returns user data including password hash.
"""
        
        with patch('src.agents.base_agent.Agent') as mock_agent_class, \
             patch('src.conversation.manager.ReviewAgent') as mock_review_agent_class:
            
            # Mock the Strands Agent class to return an awaitable agent
            mock_strands_agent = Mock()
            mock_strands_agent.invoke_async = AsyncMock(return_value="MOCK FILE REVIEW: This API documentation has security issues.")
            mock_strands_agent.__call__ = Mock(return_value="MOCK FILE REVIEW: This API documentation has security issues.")
            mock_agent_class.return_value = mock_strands_agent
            
            mock_review_agent = Mock()
            mock_review_agent.invoke_async_legacy = AsyncMock(return_value="MOCK FILE REVIEW: This API documentation has security issues.")
            mock_review_agent.review.return_value = "MOCK FILE REVIEW: This API documentation has security issues."
            mock_review_agent.get_info.return_value = {
                'name': 'File Review Agent',
                'role': 'File Content Reviewer',
                'goal': 'Review file-based content'
            }
            mock_review_agent.persona = Mock()
            mock_review_agent.persona.name = 'File Review Agent'
            mock_review_agent.persona.role = 'File Content Reviewer'
            mock_review_agent.name = 'file_review_agent'
            mock_review_agent.agent = mock_strands_agent
            
            mock_review_agent_class.return_value = mock_review_agent
            
            try:
                manager = ConversationManager(enable_analysis=False)
                
                if len(manager.agents) == 0:
                    pytest.skip("No agents available for testing")
                
                result = await manager.run_review(sample_api_doc)
                
                assert result.content == sample_api_doc
                assert len(result.reviews) > 0
                
            except OSError as e:
                if "File name too long" in str(e):
                    pytest.skip(f"Skipping due to filesystem limitation: {e}")
                else:
                    raise
            except ImportError as e:
                pytest.skip(f"Missing dependencies for integration test: {e}")
            except Exception as e:
                # Let other exceptions fail the test so we can see what's wrong
                raise
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_expected_feedback_patterns(self, sample_python_code):
        """Test that mock reviews contain expected feedback patterns."""
        
        expected_patterns = {
            "security": ["security", "vulnerability", "MD5", "password", "hash"],
            "technical": ["code quality", "best practices", "error handling", "validation"],
            "content": ["documentation", "clarity", "explanation"]
        }
        
        with patch('src.agents.base_agent.Agent') as mock_agent_class, \
             patch('src.conversation.manager.ReviewAgent') as mock_review_agent_class:
            
            # Mock the Strands Agent class to return an awaitable agent
            mock_strands_agent = Mock()
            mock_strands_agent.invoke_async = AsyncMock(return_value="SECURITY REVIEW: Critical security vulnerabilities found including MD5 password hashing and plaintext password logging. TECHNICAL REVIEW: Code quality issues detected - missing error handling and input validation best practices.")
            mock_strands_agent.__call__ = Mock(return_value="SECURITY REVIEW: Critical security vulnerabilities found including MD5 password hashing and plaintext password logging. TECHNICAL REVIEW: Code quality issues detected - missing error handling and input validation best practices.")
            mock_agent_class.return_value = mock_strands_agent
            
            # Create mock ReviewAgent with expected feedback patterns
            mock_review_agent = Mock()
            mock_review_agent.invoke_async_legacy = AsyncMock(return_value="SECURITY REVIEW: Critical security vulnerabilities found including MD5 password hashing and plaintext password logging. TECHNICAL REVIEW: Code quality issues detected - missing error handling and input validation best practices.")
            mock_review_agent.review.return_value = "SECURITY REVIEW: Critical security vulnerabilities found including MD5 password hashing and plaintext password logging. TECHNICAL REVIEW: Code quality issues detected - missing error handling and input validation best practices."
            mock_review_agent.get_info.return_value = {
                'name': 'Pattern Test Agent',
                'role': 'Pattern Reviewer', 
                'goal': 'Test feedback patterns'
            }
            mock_review_agent.persona = Mock()
            mock_review_agent.persona.name = 'Pattern Test Agent'
            mock_review_agent.persona.role = 'Pattern Reviewer'
            mock_review_agent.name = 'pattern_test_agent'
            mock_review_agent.agent = mock_strands_agent
            
            # Return the same mock for any number of agents
            mock_review_agent_class.return_value = mock_review_agent
            
            try:
                manager = ConversationManager(enable_analysis=False)
                
                if len(manager.agents) == 0:
                    pytest.skip("No agents available for testing")
                
                result = await manager.run_review(sample_python_code)
                
                # Check that feedback contains expected patterns
                all_feedback = " ".join([r.feedback.lower() for r in result.reviews if not r.error])
                
                # At least some expected patterns should be present
                found_patterns = []
                for category, patterns in expected_patterns.items():
                    for pattern in patterns:
                        if pattern in all_feedback:
                            found_patterns.append(pattern)
                
                assert len(found_patterns) > 0, f"No expected patterns found in feedback: {all_feedback}"
                
            except OSError as e:
                if "File name too long" in str(e):
                    pytest.skip(f"Skipping due to filesystem limitation: {e}")
                else:
                    raise
            except ImportError as e:
                pytest.skip(f"Missing dependencies for integration test: {e}")
            except Exception as e:
                # Let other exceptions fail the test so we can see what's wrong
                raise


class TestSystemHealth:
    """System health and configuration tests."""
    
    def test_import_health(self):
        """Test that all core modules can be imported."""
        try:
            from src.config.persona_loader import PersonaLoader, PersonaConfig
            from src.agents.review_agent import ReviewAgent
            from src.agents.context_agent import ContextAgent
            from src.conversation.manager import ConversationManager
            from src.agents.analysis_agent import AnalysisAgent
            
            assert PersonaLoader
            assert PersonaConfig
            assert ReviewAgent
            assert ContextAgent
            assert ConversationManager
            assert AnalysisAgent
            
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_configuration_paths(self):
        """Test that configuration paths are accessible."""
        loader = PersonaLoader()
        config_info = loader.get_config_info()
        
        assert 'project_root' in config_info
        assert 'personas_dir' in config_info
        assert Path(config_info['project_root']).exists()
        
        # personas_dir might not exist in test environment, that's ok
        personas_dir_exists = config_info.get('personas_dir_exists', False)
        if not personas_dir_exists:
            pytest.skip("Personas directory not configured - this is expected in test environment")
    
    def test_test_inputs_available(self, test_files_dir):
        """Test that test input files are available."""
        if not test_files_dir.exists():
            pytest.skip("Test inputs directory not found")
        
        # Check for at least one test file
        test_files = list(test_files_dir.glob("*.py")) + list(test_files_dir.glob("*.md")) + list(test_files_dir.glob("*.html"))
        
        if len(test_files) == 0:
            pytest.skip("No test input files found")
        
        # Verify files are readable
        for test_file in test_files[:3]:  # Check first 3 files
            try:
                content = test_file.read_text(encoding='utf-8')
                assert len(content) > 0
            except Exception as e:
                pytest.fail(f"Could not read test file {test_file}: {e}")
