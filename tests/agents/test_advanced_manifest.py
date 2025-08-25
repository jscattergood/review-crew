"""
Tests for advanced manifest features.

Tests document relationships, review focus customization, output formatting,
and context file processing.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.agents.conversation_manager import ConversationManager


class TestAdvancedManifest:
    """Test advanced manifest processing features."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ConversationManager for testing."""
        with patch('src.agents.conversation_manager.PersonaLoader') as mock_loader:
            mock_loader.return_value.load_reviewer_personas.return_value = []
            mock_loader.return_value.load_analyzer_personas.return_value = []
            mock_loader.return_value.load_contextualizer_personas.return_value = []
            
            manager = ConversationManager()
            return manager

    def test_process_context_files(self, tmp_path, mock_manager):
        """Test processing of context files from manifest."""
        # Create test context file
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "requirements.md"
        context_file.write_text("# Requirements\nTest requirements content")
        
        # Create review config with context files
        review_config = {
            "documents": {
                "context_files": [
                    {
                        "path": "context/requirements.md",
                        "type": "requirements",
                        "weight": "high"
                    }
                ]
            }
        }
        
        manager = mock_manager
        context_files = manager._process_context_files(review_config, tmp_path)
        
        assert len(context_files) == 1
        assert context_files[0]["path"] == "context/requirements.md"
        assert context_files[0]["type"] == "requirements"
        assert context_files[0]["weight"] == "high"
        assert context_files[0]["loaded"] is True
        assert "Test requirements content" in context_files[0]["content"]

    def test_process_context_files_missing(self, tmp_path, mock_manager):
        """Test handling of missing context files."""
        review_config = {
            "documents": {
                "context_files": [
                    {
                        "path": "missing/file.md",
                        "type": "requirements",
                        "weight": "high"
                    }
                ]
            }
        }
        
        manager = mock_manager
        context_files = manager._process_context_files(review_config, tmp_path)
        
        assert len(context_files) == 0

    def test_process_document_relationships(self, mock_manager):
        """Test processing of document relationships."""
        review_config = {
            "documents": {
                "relationships": [
                    {
                        "source": "personal-statement.md",
                        "target": "supplemental-essay.md",
                        "type": "complements",
                        "note": "Should complement, not repeat"
                    },
                    {
                        "source": "activities.md",
                        "target": "personal-statement.md",
                        "type": "supports",
                        "note": "Provides evidence"
                    }
                ]
            }
        }
        
        manager = mock_manager
        relationships = manager._process_document_relationships(review_config)
        
        assert len(relationships) == 2
        assert relationships[0]["source"] == "personal-statement.md"
        assert relationships[0]["target"] == "supplemental-essay.md"
        assert relationships[0]["type"] == "complements"
        assert relationships[0]["note"] == "Should complement, not repeat"

    def test_process_review_focus(self, mock_manager):
        """Test processing of review focus configuration."""
        review_config = {
            "review_focus": {
                "primary_concerns": [
                    {
                        "concern": "Narrative coherence",
                        "weight": "critical",
                        "description": "Ensure consistent story"
                    }
                ],
                "secondary_concerns": [
                    {
                        "concern": "Writing quality",
                        "weight": "medium"
                    }
                ]
            }
        }
        
        manager = mock_manager
        focus_config = manager._process_review_focus(review_config)
        
        assert len(focus_config["primary_concerns"]) == 1
        assert len(focus_config["secondary_concerns"]) == 1
        assert len(focus_config["focus_instructions"]) == 2
        
        # Check instruction formatting
        instructions = focus_config["focus_instructions"]
        assert "🔴 CRITICAL" in instructions[0]
        assert "Narrative coherence" in instructions[0]
        assert "🔵 CONSIDER" in instructions[1]

    def test_process_output_configuration(self, mock_manager):
        """Test processing of output configuration."""
        review_config = {
            "output": {
                "format": "comprehensive",
                "include_sections": ["executive_summary", "document_reviews"],
                "exclude_sections": ["raw_content"],
                "summary_length": "detailed",
                "include_scores": True,
                "highlight_critical_issues": True
            }
        }
        
        manager = mock_manager
        output_config = manager._process_output_configuration(review_config)
        
        assert output_config["format"] == "comprehensive"
        assert "executive_summary" in output_config["include_sections"]
        assert "raw_content" in output_config["exclude_sections"]
        assert output_config["include_scores"] is True

    def test_build_enhanced_context(self, mock_manager):
        """Test building enhanced context from manifest data."""
        review_config = {
            "processed_context": [
                {
                    "path": "requirements.md",
                    "type": "requirements",
                    "weight": "high",
                    "content": "Test requirements",
                    "loaded": True
                }
            ],
            "processed_relationships": [
                {
                    "source": "doc1.md",
                    "target": "doc2.md",
                    "type": "complements",
                    "note": "Test relationship"
                }
            ],
            "processed_focus": {
                "focus_instructions": [
                    "🔴 CRITICAL: Focus on coherence"
                ]
            }
        }
        
        manager = mock_manager
        enhanced_context = manager._build_enhanced_context("Original context", review_config)
        
        assert "Original context" in enhanced_context
        assert "Context: requirements.md 🔴" in enhanced_context
        assert "Test requirements" in enhanced_context
        assert "Document Relationships" in enhanced_context
        assert "doc1.md complements doc2.md" in enhanced_context
        assert "Review Focus Instructions" in enhanced_context
        assert "🔴 CRITICAL: Focus on coherence" in enhanced_context

    def test_apply_focus_to_reviewers(self, mock_manager):
        """Test applying focus instructions to reviewer personas."""
        # Create mock reviewer persona
        mock_persona = MagicMock()
        mock_persona.name = "Test Reviewer"
        mock_persona.role = "Test Role"
        mock_persona.goal = "Test Goal"
        mock_persona.backstory = "Test Backstory"
        mock_persona.prompt_template = "Original prompt template"
        mock_persona.model_config = {}
        
        review_config = {
            "processed_focus": {
                "focus_instructions": [
                    "🔴 CRITICAL: Focus on coherence",
                    "🟡 HIGH PRIORITY: Check grammar"
                ]
            }
        }
        
        manager = mock_manager
        enhanced_reviewers = manager._apply_focus_to_reviewers([mock_persona], review_config)
        
        assert len(enhanced_reviewers) == 1
        enhanced_prompt = enhanced_reviewers[0].prompt_template
        assert "Original prompt template" in enhanced_prompt
        assert "SPECIAL FOCUS AREAS FOR THIS REVIEW" in enhanced_prompt
        assert "🔴 CRITICAL: Focus on coherence" in enhanced_prompt
        assert "🟡 HIGH PRIORITY: Check grammar" in enhanced_prompt

    def test_advanced_manifest_integration(self, tmp_path, mock_manager):
        """Test full advanced manifest processing integration."""
        # Create test directory structure
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "requirements.md").write_text("Test requirements")
        
        # Create advanced manifest
        manifest_content = {
            "review_configuration": {
                "name": "Advanced Test Review",
                "documents": {
                    "primary": "main.md",
                    "context_files": [
                        {
                            "path": "context/requirements.md",
                            "type": "requirements",
                            "weight": "high"
                        }
                    ],
                    "relationships": [
                        {
                            "source": "main.md",
                            "target": "support.md",
                            "type": "complements"
                        }
                    ]
                },
                "review_focus": {
                    "primary_concerns": [
                        {
                            "concern": "Test coherence",
                            "weight": "critical"
                        }
                    ]
                },
                "output": {
                    "format": "comprehensive",
                    "include_scores": True
                }
            }
        }
        
        manager = mock_manager
        enhanced_manifest = manager._process_advanced_manifest(manifest_content, tmp_path)
        
        # Verify all advanced features were processed
        review_config = enhanced_manifest["review_configuration"]
        assert "processed_context" in review_config
        assert "processed_relationships" in review_config
        assert "processed_focus" in review_config
        assert "processed_output" in review_config
        
        # Verify context file was loaded
        assert len(review_config["processed_context"]) == 1
        assert review_config["processed_context"][0]["loaded"] is True
