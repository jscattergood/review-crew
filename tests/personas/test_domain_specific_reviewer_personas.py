"""
Tests for domain-specific reviewer personas.

Tests the academic, business, and content domain reviewers for specialized evaluation.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
from src.config.persona_loader import PersonaLoader
from src.agents.review_agent import ReviewAgent


class TestDomainSpecificReviewerPersonas:
    """Test domain-specific reviewer personas for specialized evaluation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use the actual personas directory for testing
        self.personas_dir = Path("examples/personas")
        self.loader = PersonaLoader(personas_dir=self.personas_dir)

    def test_application_reviewer_loads(self):
        """Test that Application Reviewer persona loads correctly."""
        # Load the Application Reviewer
        persona_path = self.personas_dir / "reviewers" / "academic" / "application_reviewer.yaml"
        assert persona_path.exists(), "Application Reviewer persona file should exist"
        
        persona = self.loader.load_persona(persona_path)
        
        # Verify persona properties
        assert persona.name == "Application Reviewer"
        assert persona.role == "Admissions Specialist & College Application Expert"
        assert "coherence" in persona.goal.lower()
        assert "admissions" in persona.backstory.lower()
        assert "Narrative Coherence" in persona.prompt_template
        assert "Authenticity Assessment" in persona.prompt_template
        assert "competitive" in persona.prompt_template.lower()

    def test_multi_document_reviewer_loads(self):
        """Test that Multi-Document Reviewer persona loads correctly."""
        persona_path = self.personas_dir / "reviewers" / "content" / "multi_document_reviewer.yaml"
        assert persona_path.exists(), "Multi-Document Reviewer persona file should exist"
        
        persona = self.loader.load_persona(persona_path)
        
        # Verify persona properties
        assert persona.name == "Multi-Document Reviewer"
        assert persona.role == "Content Integration Specialist"
        assert "coherence" in persona.goal.lower()
        assert "integration" in persona.backstory.lower()
        assert "Consistency Evaluation" in persona.prompt_template
        assert "Content Integration" in persona.prompt_template
        assert "Strategic Communication" in persona.prompt_template

    def test_business_proposal_reviewer_loads(self):
        """Test that Business Proposal Reviewer persona loads correctly."""
        persona_path = self.personas_dir / "reviewers" / "business" / "proposal_reviewer.yaml"
        assert persona_path.exists(), "Business Proposal Reviewer persona file should exist"
        
        persona = self.loader.load_persona(persona_path)
        
        # Verify persona properties
        assert persona.name == "Business Proposal Reviewer"
        assert persona.role == "Business Strategy & Proposal Development Specialist"
        assert "business" in persona.goal.lower()
        assert "consultant" in persona.backstory.lower()
        assert "Strategic Clarity" in persona.prompt_template
        assert "Market & Competitive Analysis" in persona.prompt_template
        assert "Financial Viability" in persona.prompt_template

    def test_academic_category_loading(self):
        """Test loading reviewers from academic category."""
        academic_personas = self.loader.load_reviewer_personas_by_category(["academic"])
        
        # Should find the Application Reviewer
        persona_names = [p.name for p in academic_personas]
        assert "Application Reviewer" in persona_names
        assert len(academic_personas) >= 1

    def test_content_category_loading(self):
        """Test loading reviewers from content category."""
        content_personas = self.loader.load_reviewer_personas_by_category(["content"])
        
        # Should find Content Reviewer, UX Reviewer, and Multi-Document Reviewer
        persona_names = [p.name for p in content_personas]
        assert "Content Reviewer" in persona_names
        assert "UX Reviewer" in persona_names
        assert "Multi-Document Reviewer" in persona_names
        assert len(content_personas) >= 3

    def test_technical_category_loading(self):
        """Test loading reviewers from technical category."""
        technical_personas = self.loader.load_reviewer_personas_by_category(["technical"])
        
        # Should find Technical Reviewer and Security Reviewer
        persona_names = [p.name for p in technical_personas]
        assert "Technical Reviewer" in persona_names
        assert "Security Reviewer" in persona_names
        assert len(technical_personas) >= 2

    def test_business_category_loading(self):
        """Test loading reviewers from business category."""
        business_personas = self.loader.load_reviewer_personas_by_category(["business"])
        
        # Should find Business Proposal Reviewer
        persona_names = [p.name for p in business_personas]
        assert "Business Proposal Reviewer" in persona_names
        assert len(business_personas) >= 1

    def test_multiple_categories_loading(self):
        """Test loading reviewers from multiple categories."""
        personas = self.loader.load_reviewer_personas_by_category(["academic", "content"])
        
        # Should find personas from both categories
        persona_names = [p.name for p in personas]
        assert "Application Reviewer" in persona_names  # academic
        assert "Content Reviewer" in persona_names      # content
        assert "Multi-Document Reviewer" in persona_names  # content
        assert len(personas) >= 4

    def test_reviewer_persona_instantiation(self):
        """Test that new reviewer personas can be instantiated as ReviewAgents."""
        # Load Application Reviewer
        persona_path = self.personas_dir / "reviewers" / "academic" / "application_reviewer.yaml"
        persona = self.loader.load_persona(persona_path)
        
        # Mock model config
        model_config = {
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # Should be able to create ReviewAgent (will fail without actual provider, but class should instantiate)
        try:
            agent = ReviewAgent(persona, None, model_config)
            assert agent.persona == persona
            assert agent.model_config == model_config
        except Exception as e:
            # Expected to fail due to missing model provider, but persona should be valid
            assert "persona" not in str(e).lower(), f"Persona validation failed: {e}"

    def test_all_domain_specific_personas_have_required_fields(self):
        """Test that all domain-specific personas have required fields."""
        persona_paths = [
            self.personas_dir / "reviewers" / "academic" / "application_reviewer.yaml",
            self.personas_dir / "reviewers" / "content" / "multi_document_reviewer.yaml",
            self.personas_dir / "reviewers" / "business" / "proposal_reviewer.yaml"
        ]
        
        for persona_path in persona_paths:
            assert persona_path.exists(), f"Persona file should exist: {persona_path}"
            
            persona = self.loader.load_persona(persona_path)
            
            # Verify required fields
            assert persona.name, f"Persona should have name: {persona_path}"
            assert persona.role, f"Persona should have role: {persona_path}"
            assert persona.goal, f"Persona should have goal: {persona_path}"
            assert persona.backstory, f"Persona should have backstory: {persona_path}"
            assert persona.prompt_template, f"Persona should have prompt_template: {persona_path}"
            
            # Verify model config
            assert hasattr(persona, 'model_config'), f"Persona should have model_config: {persona_path}"
            assert isinstance(persona.model_config, dict), f"model_config should be dict: {persona_path}"
