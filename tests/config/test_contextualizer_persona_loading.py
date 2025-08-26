"""
Test contextualizer manifest loading functionality.

Tests the new PersonaLoader methods for loading contextualizers based on manifest
configuration, including category-based and name-based selection.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from src.config.persona_loader import PersonaLoader, PersonaConfig


class TestContextualizerManifestLoading:
    """Test contextualizer loading from manifest configuration."""

    @pytest.fixture
    def temp_personas_dir(self):
        """Create temporary personas directory with test contextualizers."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create contextualizers directory structure
        contextualizers_dir = temp_dir / "contextualizers"
        contextualizers_dir.mkdir(parents=True)
        
        # Create some category directories
        business_dir = contextualizers_dir / "business"
        academic_dir = contextualizers_dir / "academic"
        business_dir.mkdir()
        academic_dir.mkdir()
        
        # Create test contextualizer files
        business_contextualizer = business_dir / "business_contextualizer.yaml"
        business_contextualizer.write_text("""
name: "Business Context Analyst"
role: "Business Context Specialist"
goal: "Provide business context and market analysis"
backstory: "Expert in business strategy and market analysis"
prompt_template: "Analyze the business context of: {content}"
model_config:
  temperature: 0.7
  max_tokens: 1500
""")
        
        academic_contextualizer = academic_dir / "academic_contextualizer.yaml"
        academic_contextualizer.write_text("""
name: "Academic Context Specialist"
role: "Academic Assessment Expert"
goal: "Provide academic context and institutional fit analysis"
backstory: "Expert in academic evaluation and institutional assessment"
prompt_template: "Analyze the academic context of: {content}"
model_config:
  temperature: 0.6
  max_tokens: 1200
""")
        
        # Create a contextualizer in the main directory (flat structure)
        main_contextualizer = contextualizers_dir / "general_contextualizer.yaml"
        main_contextualizer.write_text("""
name: "General Contextualizer"
role: "General Context Provider"
goal: "Provide general context analysis"
backstory: "Expert in general context analysis"
prompt_template: "Provide context for: {content}"
model_config:
  temperature: 0.5
  max_tokens: 1000
""")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_load_contextualizer_personas_by_names(self, temp_personas_dir):
        """Test loading contextualizers by specific names."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        # Test loading specific contextualizers by name
        names = ["Business Context Analyst", "General Contextualizer"]
        contextualizers = loader.load_contextualizer_personas_by_names(names)
        
        assert len(contextualizers) == 2
        contextualizer_names = [c.name for c in contextualizers]
        assert "Business Context Analyst" in contextualizer_names
        assert "General Contextualizer" in contextualizer_names

    def test_load_contextualizer_personas_by_names_not_found(self, temp_personas_dir):
        """Test loading contextualizers when some names don't exist."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        # Test with non-existent contextualizer
        names = ["Business Context Analyst", "Nonexistent Contextualizer"]
        contextualizers = loader.load_contextualizer_personas_by_names(names)
        
        # Should only return the found one
        assert len(contextualizers) == 1
        assert contextualizers[0].name == "Business Context Analyst"

    def test_load_contextualizer_personas_by_category(self, temp_personas_dir):
        """Test loading contextualizers by category."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        # Test loading from business category
        business_contextualizers = loader.load_contextualizer_personas_by_category(["business"])
        assert len(business_contextualizers) == 1
        assert business_contextualizers[0].name == "Business Context Analyst"
        
        # Test loading from academic category
        academic_contextualizers = loader.load_contextualizer_personas_by_category(["academic"])
        assert len(academic_contextualizers) == 1
        assert academic_contextualizers[0].name == "Academic Context Specialist"
        
        # Test loading from multiple categories
        multiple_contextualizers = loader.load_contextualizer_personas_by_category(["business", "academic"])
        assert len(multiple_contextualizers) == 2

    def test_load_contextualizer_personas_by_category_not_found(self, temp_personas_dir):
        """Test loading contextualizers from non-existent category."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        # Test with non-existent category
        contextualizers = loader.load_contextualizer_personas_by_category(["nonexistent"])
        assert len(contextualizers) == 0

    def test_load_contextualizers_from_manifest_categories_only(self, temp_personas_dir):
        """Test loading contextualizers from manifest with categories only."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {
            "contextualizer_categories": ["business", "academic"]
        }
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        assert len(contextualizers) == 2
        contextualizer_names = [c.name for c in contextualizers]
        assert "Business Context Analyst" in contextualizer_names
        assert "Academic Context Specialist" in contextualizer_names

    def test_load_contextualizers_from_manifest_names_only(self, temp_personas_dir):
        """Test loading contextualizers from manifest with names only."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {
            "contextualizers": ["General Contextualizer", "Business Context Analyst"]
        }
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        assert len(contextualizers) == 2
        contextualizer_names = [c.name for c in contextualizers]
        assert "General Contextualizer" in contextualizer_names
        assert "Business Context Analyst" in contextualizer_names

    def test_load_contextualizers_from_manifest_both_categories_and_names(self, temp_personas_dir):
        """Test loading contextualizers from manifest with both categories and names."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {
            "contextualizer_categories": ["business"],
            "contextualizers": ["General Contextualizer"]
        }
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        assert len(contextualizers) == 2
        contextualizer_names = [c.name for c in contextualizers]
        assert "Business Context Analyst" in contextualizer_names
        assert "General Contextualizer" in contextualizer_names

    def test_load_contextualizers_from_manifest_deduplication(self, temp_personas_dir):
        """Test that duplicate contextualizers are removed when specified in both ways."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {
            "contextualizer_categories": ["business"],  # This loads "Business Context Analyst"
            "contextualizers": ["Business Context Analyst"]  # This also tries to load the same one
        }
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        # Should only have one instance despite being specified twice
        assert len(contextualizers) == 1
        assert contextualizers[0].name == "Business Context Analyst"

    def test_load_contextualizers_from_manifest_empty_config(self, temp_personas_dir):
        """Test loading contextualizers from empty manifest config."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {}
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        assert len(contextualizers) == 0

    def test_load_contextualizers_from_manifest_no_contextualizer_config(self, temp_personas_dir):
        """Test loading contextualizers from manifest without contextualizer sections."""
        loader = PersonaLoader(personas_dir=temp_personas_dir)
        
        manifest_config = {
            "reviewer_categories": ["academic"],  # Other config, but no contextualizer config
            "analyzers": ["Some Analyzer"]
        }
        
        contextualizers = loader.load_contextualizers_from_manifest(manifest_config)
        assert len(contextualizers) == 0
