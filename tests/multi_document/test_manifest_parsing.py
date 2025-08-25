"""Tests for manifest parsing functionality."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, Mock

from src.agents.conversation_manager import ConversationManager
from src.config.persona_loader import PersonaLoader, PersonaConfig


class TestManifestParsing:
    """Test manifest parsing and reviewer selection."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ConversationManager for testing."""
        with patch('src.agents.conversation_manager.PersonaLoader') as mock_loader:
            # Mock the PersonaLoader to prevent it from loading actual personas
            mock_loader.return_value.load_reviewer_personas.return_value = []
            mock_loader.return_value.load_analyzer_personas.return_value = []
            mock_loader.return_value.load_contextualizer_personas.return_value = []
            
            manager = ConversationManager()
            return manager
    
    def test_load_manifest_valid_file(self, tmp_path, mock_manager):
        """Test loading a valid manifest file."""
        manifest_content = {
            "review_configuration": {
                "name": "Test Review",
                "reviewers": ["Content Reviewer"],
                "reviewer_categories": ["academic"]
            }
        }
        
        manifest_file = tmp_path / "manifest.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest_content, f)
        
        manager = mock_manager
        loaded_manifest = manager._load_manifest(manifest_file)
        
        assert loaded_manifest == manifest_content
        assert loaded_manifest["review_configuration"]["name"] == "Test Review"
    
    def test_load_manifest_invalid_file(self, tmp_path):
        """Test loading an invalid manifest file."""
        manifest_file = tmp_path / "invalid.yaml"
        manifest_file.write_text("invalid: yaml: content:")  # Invalid YAML
        
        manager = ConversationManager()
        loaded_manifest = manager._load_manifest(manifest_file)
        
        # Should return empty dict on error
        assert loaded_manifest == {}
    
    def test_load_manifest_nonexistent_file(self, tmp_path):
        """Test loading a non-existent manifest file."""
        manifest_file = tmp_path / "nonexistent.yaml"
        
        manager = ConversationManager()
        loaded_manifest = manager._load_manifest(manifest_file)
        
        # Should return empty dict on error
        assert loaded_manifest == {}
    
    @patch('src.config.persona_loader.PersonaLoader.load_reviewers_from_manifest')
    @patch('src.agents.conversation_manager.ConversationManager._run_single_document_review')
    def test_run_manifest_review_with_reviewers(self, mock_single_review, mock_load_reviewers):
        """Test running review with manifest-specified reviewers."""
        # Mock persona loading
        mock_persona = PersonaConfig(
            name="Test Reviewer",
            role="Test Role", 
            goal="Test Goal",
            backstory="Test Backstory",
            prompt_template="Test prompt with {content}",
            model_config={}
        )
        mock_load_reviewers.return_value = [mock_persona]
        
        # Mock single document review
        mock_result = Mock()
        mock_single_review.return_value = mock_result
        
        manager = ConversationManager()
        manifest_config = {
            "review_configuration": {
                "reviewers": ["Test Reviewer"]
            }
        }
        
        result = manager._run_manifest_review("test content", None, manifest_config)
        
        # Should load reviewers from manifest
        mock_load_reviewers.assert_called_once_with(manifest_config["review_configuration"])
        
        # Should run single document review
        mock_single_review.assert_called_once_with("test content", None, None)
        
        assert result == mock_result
    
    @patch('src.agents.conversation_manager.ConversationManager._run_single_document_review')
    def test_run_manifest_review_empty_manifest(self, mock_single_review):
        """Test running review with empty manifest falls back to standard review."""
        mock_result = Mock()
        mock_single_review.return_value = mock_result
        
        manager = ConversationManager()
        result = manager._run_manifest_review("test content", None, {})
        
        # Should fall back to standard single document review
        mock_single_review.assert_called_once_with("test content", None, None)
        assert result == mock_result
    
    @patch('src.config.persona_loader.PersonaLoader.load_reviewers_from_manifest')
    @patch('src.agents.conversation_manager.ConversationManager._run_single_document_review')
    def test_run_manifest_review_reviewer_loading_error(self, mock_single_review, mock_load_reviewers):
        """Test manifest review handles reviewer loading errors gracefully."""
        # Mock reviewer loading to raise an exception
        mock_load_reviewers.side_effect = Exception("Failed to load reviewers")
        
        mock_result = Mock()
        mock_single_review.return_value = mock_result
        
        manager = ConversationManager()
        manifest_config = {
            "review_configuration": {
                "reviewers": ["Invalid Reviewer"]
            }
        }
        
        result = manager._run_manifest_review("test content", None, manifest_config)
        
        # Should fall back to standard review despite error
        mock_single_review.assert_called_once_with("test content", None, None)
        assert result == mock_result


class TestPersonaLoaderManifestIntegration:
    """Test PersonaLoader manifest integration methods."""
    
    def test_load_reviewers_from_manifest_by_names(self):
        """Test loading reviewers by specific names from manifest."""
        with patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas') as mock_load_all:
            # Mock available personas
            persona1 = PersonaConfig(
                name="Content Reviewer",
                role="Content Specialist",
                goal="Review content",
                backstory="Content expert",
                prompt_template="Review: {content}",
                model_config={}
            )
            persona2 = PersonaConfig(
                name="Technical Reviewer", 
                role="Technical Specialist",
                goal="Review tech",
                backstory="Tech expert",
                prompt_template="Review: {content}",
                model_config={}
            )
            mock_load_all.return_value = [persona1, persona2]
            
            loader = PersonaLoader()
            manifest_config = {
                "reviewers": ["Content Reviewer"]
            }
            
            result = loader.load_reviewers_from_manifest(manifest_config)
            
            assert len(result) == 1
            assert result[0].name == "Content Reviewer"
    
    def test_load_reviewers_from_manifest_by_categories(self):
        """Test loading reviewers by categories from manifest."""
        with patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas_by_category') as mock_load_category:
            # Mock category-based loading
            persona1 = PersonaConfig(
                name="Academic Reviewer",
                role="Academic Specialist", 
                goal="Review academics",
                backstory="Academic expert",
                prompt_template="Review: {content}",
                model_config={}
            )
            mock_load_category.return_value = [persona1]
            
            loader = PersonaLoader()
            manifest_config = {
                "reviewer_categories": ["academic"]
            }
            
            result = loader.load_reviewers_from_manifest(manifest_config)
            
            mock_load_category.assert_called_once_with(["academic"])
            assert len(result) == 1
            assert result[0].name == "Academic Reviewer"
    
    def test_load_reviewers_from_manifest_mixed(self):
        """Test loading reviewers using both names and categories."""
        with patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas') as mock_load_all, \
             patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas_by_category') as mock_load_category:
            
            # Mock personas from names
            persona1 = PersonaConfig(
                name="Content Reviewer",
                role="Content Specialist",
                goal="Review content", 
                backstory="Content expert",
                prompt_template="Review: {content}",
                model_config={}
            )
            mock_load_all.return_value = [persona1]
            
            # Mock personas from categories  
            persona2 = PersonaConfig(
                name="Academic Reviewer",
                role="Academic Specialist",
                goal="Review academics",
                backstory="Academic expert", 
                prompt_template="Review: {content}",
                model_config={}
            )
            mock_load_category.return_value = [persona2]
            
            loader = PersonaLoader()
            manifest_config = {
                "reviewers": ["Content Reviewer"],
                "reviewer_categories": ["academic"]
            }
            
            result = loader.load_reviewers_from_manifest(manifest_config)
            
            # Should get both personas
            assert len(result) == 2
            names = [p.name for p in result]
            assert "Content Reviewer" in names
            assert "Academic Reviewer" in names
    
    def test_load_reviewers_from_manifest_deduplication(self):
        """Test that duplicate reviewers are removed."""
        with patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas') as mock_load_all, \
             patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas_by_category') as mock_load_category:
            
            # Same persona from both methods
            persona1 = PersonaConfig(
                name="Content Reviewer",
                role="Content Specialist", 
                goal="Review content",
                backstory="Content expert",
                prompt_template="Review: {content}",
                model_config={}
            )
            mock_load_all.return_value = [persona1]
            mock_load_category.return_value = [persona1]  # Same persona
            
            loader = PersonaLoader()
            manifest_config = {
                "reviewers": ["Content Reviewer"],
                "reviewer_categories": ["content"]
            }
            
            result = loader.load_reviewers_from_manifest(manifest_config)
            
            # Should deduplicate
            assert len(result) == 1
            assert result[0].name == "Content Reviewer"
