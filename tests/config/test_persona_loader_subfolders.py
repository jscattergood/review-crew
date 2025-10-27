"""Tests for PersonaLoader sub-folder functionality."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.config.persona_loader import PersonaConfig, PersonaLoader


class TestPersonaLoaderSubfolders:
    """Test PersonaLoader sub-folder organization support."""

    def test_load_personas_from_flat_structure(self, tmp_path):
        """Test loading personas from flat directory structure (backwards compatibility)."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create persona files in flat structure
        persona1_content = """
name: "Content Reviewer"
role: "Content Specialist"
goal: "Review content quality"
backstory: "Expert in content strategy"
prompt_template: "Review this content: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "content_reviewer.yaml").write_text(persona1_content)

        persona2_content = """
name: "Technical Reviewer"  
role: "Technical Specialist"
goal: "Review technical aspects"
backstory: "Expert in technology"
prompt_template: "Review this tech: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "technical_reviewer.yaml").write_text(persona2_content)

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader._load_personas_from_dir(reviewers_dir)

        assert len(personas) == 2
        names = [p.name for p in personas]
        assert "Content Reviewer" in names
        assert "Technical Reviewer" in names

    def test_load_personas_from_subfolder_structure(self, tmp_path):
        """Test loading personas from organized sub-folder structure."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create sub-folders
        content_dir = reviewers_dir / "content"
        content_dir.mkdir()
        technical_dir = reviewers_dir / "technical"
        technical_dir.mkdir()

        # Create persona files in sub-folders
        content_persona = """
name: "Content Reviewer"
role: "Content Specialist" 
goal: "Review content quality"
backstory: "Expert in content strategy"
prompt_template: "Review this content: {content}"
model_config:
  temperature: 0.3
"""
        (content_dir / "content_reviewer.yaml").write_text(content_persona)

        tech_persona = """
name: "Technical Reviewer"
role: "Technical Specialist"
goal: "Review technical aspects" 
backstory: "Expert in technology"
prompt_template: "Review this tech: {content}"
model_config:
  temperature: 0.3
"""
        (technical_dir / "technical_reviewer.yaml").write_text(tech_persona)

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader._load_personas_from_dir(reviewers_dir)

        assert len(personas) == 2
        names = [p.name for p in personas]
        assert "Content Reviewer" in names
        assert "Technical Reviewer" in names

    def test_load_personas_mixed_structure(self, tmp_path):
        """Test loading personas from mixed flat and sub-folder structure."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create flat file
        flat_persona = """
name: "General Reviewer"
role: "General Specialist"
goal: "General review"
backstory: "General expert" 
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "general_reviewer.yaml").write_text(flat_persona)

        # Create sub-folder with file
        content_dir = reviewers_dir / "content"
        content_dir.mkdir()

        content_persona = """
name: "Content Reviewer"
role: "Content Specialist"
goal: "Review content"
backstory: "Content expert"
prompt_template: "Review content: {content}" 
model_config:
  temperature: 0.3
"""
        (content_dir / "content_reviewer.yaml").write_text(content_persona)

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader._load_personas_from_dir(reviewers_dir)

        assert len(personas) == 2
        names = [p.name for p in personas]
        assert "General Reviewer" in names
        assert "Content Reviewer" in names

    def test_load_reviewer_personas_by_category(self, tmp_path):
        """Test loading reviewers by specific categories."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create multiple categories
        academic_dir = reviewers_dir / "academic"
        academic_dir.mkdir()
        content_dir = reviewers_dir / "content"
        content_dir.mkdir()
        technical_dir = reviewers_dir / "technical"
        technical_dir.mkdir()

        # Create personas in each category
        academic_persona = """
name: "Academic Reviewer"
role: "Academic Specialist"
goal: "Review academic work"
backstory: "Academic expert"
prompt_template: "Review academic: {content}"
model_config:
  temperature: 0.3
"""
        (academic_dir / "academic_reviewer.yaml").write_text(academic_persona)

        content_persona = """
name: "Content Reviewer" 
role: "Content Specialist"
goal: "Review content"
backstory: "Content expert"
prompt_template: "Review content: {content}"
model_config:
  temperature: 0.3
"""
        (content_dir / "content_reviewer.yaml").write_text(content_persona)

        tech_persona = """
name: "Technical Reviewer"
role: "Technical Specialist" 
goal: "Review tech"
backstory: "Tech expert"
prompt_template: "Review tech: {content}"
model_config:
  temperature: 0.3
"""
        (technical_dir / "technical_reviewer.yaml").write_text(tech_persona)

        loader = PersonaLoader(personas_dir=tmp_path)

        # Test loading specific categories
        academic_personas = loader.load_reviewer_personas_by_category(["academic"])
        assert len(academic_personas) == 1
        assert academic_personas[0].name == "Academic Reviewer"

        content_personas = loader.load_reviewer_personas_by_category(["content"])
        assert len(content_personas) == 1
        assert content_personas[0].name == "Content Reviewer"

        # Test loading multiple categories
        multi_personas = loader.load_reviewer_personas_by_category(
            ["academic", "content"]
        )
        assert len(multi_personas) == 2
        names = [p.name for p in multi_personas]
        assert "Academic Reviewer" in names
        assert "Content Reviewer" in names

    def test_load_reviewer_personas_by_category_nonexistent(self, tmp_path):
        """Test loading reviewers by non-existent category."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader.load_reviewer_personas_by_category(["nonexistent"])

        assert len(personas) == 0

    def test_load_reviewer_personas_by_names(self, tmp_path):
        """Test loading reviewers by specific names."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        persona1_content = """
name: "Content Reviewer"
role: "Content Specialist"
goal: "Review content"
backstory: "Content expert"
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "content_reviewer.yaml").write_text(persona1_content)

        persona2_content = """
name: "Technical Reviewer"
role: "Technical Specialist"
goal: "Review tech"
backstory: "Tech expert" 
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "technical_reviewer.yaml").write_text(persona2_content)

        loader = PersonaLoader(personas_dir=tmp_path)

        # Test loading specific names
        selected_personas = loader.load_reviewer_personas_by_names(["Content Reviewer"])
        assert len(selected_personas) == 1
        assert selected_personas[0].name == "Content Reviewer"

        # Test loading multiple names
        multi_personas = loader.load_reviewer_personas_by_names(
            ["Content Reviewer", "Technical Reviewer"]
        )
        assert len(multi_personas) == 2
        names = [p.name for p in multi_personas]
        assert "Content Reviewer" in names
        assert "Technical Reviewer" in names

    def test_load_reviewer_personas_by_names_not_found(self, tmp_path):
        """Test loading reviewers by names that don't exist."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create at least one valid persona so load_reviewer_personas() doesn't fail
        valid_persona = """
name: "Valid Reviewer"
role: "Valid Specialist"
goal: "Valid review"
backstory: "Valid expert"
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "valid.yaml").write_text(valid_persona)

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader.load_reviewer_personas_by_names(["Nonexistent Reviewer"])

        assert len(personas) == 0

    def test_load_personas_with_file_errors(self, tmp_path):
        """Test loading personas handles file errors gracefully."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create valid persona
        valid_persona = """
name: "Valid Reviewer"
role: "Valid Specialist"
goal: "Valid review"
backstory: "Valid expert"
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "valid.yaml").write_text(valid_persona)

        # Create invalid persona
        (reviewers_dir / "invalid.yaml").write_text("invalid: yaml: content:")

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader._load_personas_from_dir(reviewers_dir)

        # Should load valid persona and skip invalid one
        assert len(personas) == 1
        assert personas[0].name == "Valid Reviewer"

    def test_load_personas_supports_yml_extension(self, tmp_path):
        """Test loading personas supports both .yaml and .yml extensions."""
        reviewers_dir = tmp_path / "reviewers"
        reviewers_dir.mkdir()

        # Create .yaml file
        yaml_persona = """
name: "YAML Reviewer"
role: "YAML Specialist" 
goal: "YAML review"
backstory: "YAML expert"
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "yaml_persona.yaml").write_text(yaml_persona)

        # Create .yml file
        yml_persona = """
name: "YML Reviewer"
role: "YML Specialist"
goal: "YML review" 
backstory: "YML expert"
prompt_template: "Review: {content}"
model_config:
  temperature: 0.3
"""
        (reviewers_dir / "yml_persona.yml").write_text(yml_persona)

        loader = PersonaLoader(personas_dir=tmp_path)
        personas = loader._load_personas_from_dir(reviewers_dir)

        assert len(personas) == 2
        names = [p.name for p in personas]
        assert "YAML Reviewer" in names
        assert "YML Reviewer" in names
