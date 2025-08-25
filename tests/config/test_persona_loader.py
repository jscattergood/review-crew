"""Tests for PersonaLoader."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.config.persona_loader import PersonaLoader, PersonaConfig


class TestPersonaLoader:
    """Test PersonaLoader functionality."""
    
    def test_init_default(self):
        """Test PersonaLoader initialization with defaults."""
        loader = PersonaLoader()
        assert loader.project_root.exists()
        assert "personas" in str(loader.personas_dir)
    
    def test_init_custom_paths(self, tmp_path):
        """Test PersonaLoader initialization with custom paths."""
        personas_dir = tmp_path / "custom_personas"
        personas_dir.mkdir()
        
        loader = PersonaLoader(project_root=tmp_path, personas_dir=personas_dir)
        assert loader.project_root == tmp_path
        assert loader.personas_dir == personas_dir
    
    def test_get_config_info(self):
        """Test getting configuration information."""
        loader = PersonaLoader()
        info = loader.get_config_info()
        
        assert 'project_root' in info
        assert 'personas_dir' in info
        assert 'personas_dir_exists' in info
        assert isinstance(info['personas_dir_exists'], bool)
    
    @patch('builtins.open', new_callable=mock_open, read_data="""
name: "Test Persona"
role: "Test Role"
goal: "Test goal"
backstory: "Test backstory"
prompt_template: "Test prompt: {content}"
model_config:
  temperature: 0.5
  max_tokens: 1000
""")
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_persona_success(self, mock_exists, mock_file):
        """Test successful persona loading."""
        loader = PersonaLoader()
        persona = loader.load_persona("test.yaml")
        
        assert persona.name == "Test Persona"
        assert persona.role == "Test Role"
        assert persona.goal == "Test goal"
        assert persona.backstory == "Test backstory"
        assert persona.prompt_template == "Test prompt: {content}"
        assert persona.model_config['temperature'] == 0.5
        assert persona.model_config['max_tokens'] == 1000
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_persona_file_not_found(self, mock_exists):
        """Test persona loading when file doesn't exist."""
        loader = PersonaLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_persona("nonexistent.yaml")
    
    @patch('builtins.open', new_callable=mock_open, read_data="invalid: yaml: content:")
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_persona_invalid_yaml(self, mock_exists, mock_file):
        """Test persona loading with invalid YAML."""
        loader = PersonaLoader()
        with pytest.raises(Exception):  # Should raise some parsing error
            loader.load_persona("invalid.yaml")
    
    def test_load_reviewer_personas_no_directory(self, tmp_path):
        """Test loading reviewer personas when directory doesn't exist."""
        loader = PersonaLoader(project_root=tmp_path, personas_dir=tmp_path / "nonexistent")
        
        with pytest.raises(ValueError, match="Personas directory not found"):
            loader.load_reviewer_personas()
    
    def test_load_reviewer_personas_no_reviewers_dir(self, tmp_path):
        """Test loading reviewer personas when reviewers subdirectory doesn't exist."""
        personas_dir = tmp_path / "personas"
        personas_dir.mkdir()
        
        loader = PersonaLoader(project_root=tmp_path, personas_dir=personas_dir)
        
        with pytest.raises(ValueError, match="Reviewers directory not found"):
            loader.load_reviewer_personas()
    
    @patch('src.config.persona_loader.PersonaLoader._load_personas_from_dir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_reviewer_personas_success(self, mock_exists, mock_load_personas, mock_persona):
        """Test successful reviewer personas loading."""
        mock_load_personas.return_value = [mock_persona]
        
        loader = PersonaLoader()
        personas = loader.load_reviewer_personas()
        
        assert len(personas) == 1
        assert personas[0] == mock_persona
    
    @patch('src.config.persona_loader.PersonaLoader._load_personas_from_dir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_analyzer_personas(self, mock_exists, mock_load_personas, mock_persona):
        """Test analyzer personas loading."""
        mock_load_personas.return_value = [mock_persona]
        
        loader = PersonaLoader()
        personas = loader.load_analyzer_personas()
        
        assert len(personas) == 1
        assert personas[0] == mock_persona
    
    @patch('src.config.persona_loader.PersonaLoader._load_personas_from_dir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_contextualizer_personas(self, mock_exists, mock_load_personas, mock_persona):
        """Test contextualizer personas loading."""
        mock_load_personas.return_value = [mock_persona]
        
        loader = PersonaLoader()
        personas = loader.load_contextualizer_personas()
        
        assert len(personas) == 1
        assert personas[0] == mock_persona
    
    @patch('src.config.persona_loader.PersonaLoader.load_reviewer_personas')
    @patch('src.config.persona_loader.PersonaLoader.load_analyzer_personas')
    @patch('src.config.persona_loader.PersonaLoader.load_contextualizer_personas')
    def test_load_all_persona_types(self, mock_contextualizers, mock_analyzers, mock_reviewers, mock_persona):
        """Test loading all persona types."""
        mock_reviewers.return_value = [mock_persona]
        mock_analyzers.return_value = [mock_persona]
        mock_contextualizers.return_value = [mock_persona]
        
        loader = PersonaLoader()
        all_personas = loader.load_all_persona_types()
        
        assert 'reviewers' in all_personas
        assert 'analyzers' in all_personas
        assert 'contextualizers' in all_personas
        assert len(all_personas['reviewers']) == 1
        assert len(all_personas['analyzers']) == 1
        assert len(all_personas['contextualizers']) == 1


class TestPersonaConfig:
    """Test PersonaConfig dataclass."""
    
    def test_persona_config_creation(self):
        """Test creating a PersonaConfig."""
        config = PersonaConfig(
            name="Test",
            role="Tester",
            goal="Test things",
            backstory="Testing background",
            prompt_template="Test: {content}",
            model_config={'temp': 0.5}
        )
        
        assert config.name == "Test"
        assert config.role == "Tester"
        assert config.goal == "Test things"
        assert config.backstory == "Testing background"
        assert config.prompt_template == "Test: {content}"
        assert config.model_config == {'temp': 0.5}
