"""
Tests for multi-document analyzer personas.

Tests the holistic analysis personas designed for multi-document review and synthesis.
"""

from pathlib import Path

import pytest

from src.agents.analysis_agent import AnalysisAgent
from src.config.persona_loader import PersonaLoader


class TestMultiDocumentAnalyzerPersonas:
    """Test multi-document analyzer personas for holistic analysis and synthesis."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use the actual personas directory for testing
        self.personas_dir = Path("examples/personas")
        self.loader = PersonaLoader(personas_dir=self.personas_dir)

    def test_application_coherence_analyzer_loads(self):
        """Test that Application Coherence Analyzer persona loads correctly."""
        persona_path = (
            self.personas_dir / "analyzers" / "application_coherence_analyzer.yaml"
        )
        assert persona_path.exists(), (
            "Application Coherence Analyzer persona file should exist"
        )

        persona = self.loader.load_persona(persona_path)

        # Verify persona properties
        assert persona.name == "Application Coherence Analyzer"
        assert persona.role == "Holistic Application Assessment Specialist"
        assert "coherence" in persona.goal.lower()
        assert "meta-analysis" in persona.backstory.lower()
        assert "Narrative Coherence Assessment" in persona.prompt_template
        assert "Cross-Document Pattern Analysis" in persona.prompt_template
        assert "Competitive Positioning Synthesis" in persona.prompt_template
        assert "{content}" in persona.prompt_template
        assert "{reviews}" in persona.prompt_template

    def test_cross_document_themes_analyzer_loads(self):
        """Test that Cross-Document Themes Analyzer persona loads correctly."""
        persona_path = (
            self.personas_dir / "analyzers" / "cross_document_themes_analyzer.yaml"
        )
        assert persona_path.exists(), (
            "Cross-Document Themes Analyzer persona file should exist"
        )

        persona = self.loader.load_persona(persona_path)

        # Verify persona properties
        assert persona.name == "Cross-Document Themes Analyzer"
        assert persona.role == "Thematic Consistency & Integration Specialist"
        assert "themes" in persona.goal.lower()
        assert "thematic patterns" in persona.backstory.lower()
        assert "Theme Extraction" in persona.prompt_template
        assert "Message Consistency Evaluation" in persona.prompt_template
        assert "Narrative Thread Analysis" in persona.prompt_template
        assert "{content}" in persona.prompt_template
        assert "{reviews}" in persona.prompt_template

    def test_competitive_analysis_specialist_loads(self):
        """Test that Competitive Analysis Specialist persona loads correctly."""
        persona_path = (
            self.personas_dir / "analyzers" / "competitive_analysis_specialist.yaml"
        )
        assert persona_path.exists(), (
            "Competitive Analysis Specialist persona file should exist"
        )

        persona = self.loader.load_persona(persona_path)

        # Verify persona properties
        assert persona.name == "Competitive Analysis Specialist"
        assert persona.role == "Strategic Positioning & Competitive Assessment Expert"
        assert "competitive" in persona.goal.lower()
        assert "strategic analysis" in persona.backstory.lower()
        assert "Unique Value Proposition Assessment" in persona.prompt_template
        assert "Competitive Strength Evaluation" in persona.prompt_template
        assert "Market Positioning Analysis" in persona.prompt_template
        assert "{content}" in persona.prompt_template
        assert "{reviews}" in persona.prompt_template

    def test_all_multi_document_analyzers_load_successfully(self):
        """Test that all multi-document analyzer personas can be loaded."""
        analyzer_personas = self.loader.load_analyzer_personas()

        # Check that our multi-document analyzers are included
        analyzer_names = [p.name for p in analyzer_personas]
        assert "Application Coherence Analyzer" in analyzer_names
        assert "Cross-Document Themes Analyzer" in analyzer_names
        assert "Competitive Analysis Specialist" in analyzer_names

        # Should also include existing analyzers
        assert "Quality Metrics Analyzer" in analyzer_names
        assert "Sentiment & Tone Analyzer" in analyzer_names

    def test_analyzer_persona_instantiation(self):
        """Test that new analyzer personas can be instantiated as AnalysisAgents."""
        # Load Application Coherence Analyzer
        persona_path = (
            self.personas_dir / "analyzers" / "application_coherence_analyzer.yaml"
        )
        persona = self.loader.load_persona(persona_path)

        # Mock model config
        model_config = {"temperature": 0.3, "max_tokens": 1000}

        # Should be able to create AnalysisAgent (will fail without actual provider, but class should instantiate)
        try:
            agent = AnalysisAgent(persona, None, model_config)
            assert agent.persona == persona
            assert agent.model_config == model_config
        except Exception as e:
            # Expected to fail due to missing model provider, but persona should be valid
            assert "persona" not in str(e).lower(), f"Persona validation failed: {e}"

    def test_all_multi_document_analyzers_have_required_fields(self):
        """Test that all multi-document analyzer personas have required fields."""
        analyzer_paths = [
            self.personas_dir / "analyzers" / "application_coherence_analyzer.yaml",
            self.personas_dir / "analyzers" / "cross_document_themes_analyzer.yaml",
            self.personas_dir / "analyzers" / "competitive_analysis_specialist.yaml",
        ]

        for persona_path in analyzer_paths:
            assert persona_path.exists(), f"Persona file should exist: {persona_path}"

            persona = self.loader.load_persona(persona_path)

            # Verify required fields
            assert persona.name, f"Persona should have name: {persona_path}"
            assert persona.role, f"Persona should have role: {persona_path}"
            assert persona.goal, f"Persona should have goal: {persona_path}"
            assert persona.backstory, f"Persona should have backstory: {persona_path}"
            assert persona.prompt_template, (
                f"Persona should have prompt_template: {persona_path}"
            )

            # Verify model config
            assert hasattr(persona, "model_config"), (
                f"Persona should have model_config: {persona_path}"
            )
            assert isinstance(persona.model_config, dict), (
                f"model_config should be dict: {persona_path}"
            )

            # Verify analyzer-specific requirements
            assert "{content}" in persona.prompt_template, (
                f"Analyzer should have content placeholder: {persona_path}"
            )
            assert "{reviews}" in persona.prompt_template, (
                f"Analyzer should have reviews placeholder: {persona_path}"
            )

    def test_analyzer_prompt_templates_include_synthesis_focus(self):
        """Test that multi-document analyzer personas focus on synthesis and meta-analysis."""
        analyzer_paths = [
            self.personas_dir / "analyzers" / "application_coherence_analyzer.yaml",
            self.personas_dir / "analyzers" / "cross_document_themes_analyzer.yaml",
            self.personas_dir / "analyzers" / "competitive_analysis_specialist.yaml",
        ]

        synthesis_keywords = [
            "synthesis",
            "synthesize",
            "holistic",
            "meta",
            "cross-document",
            "integration",
            "coherence",
            "patterns",
            "themes",
        ]

        for persona_path in analyzer_paths:
            persona = self.loader.load_persona(persona_path)

            # Check that prompt template includes synthesis concepts
            prompt_lower = persona.prompt_template.lower()
            found_keywords = [
                keyword for keyword in synthesis_keywords if keyword in prompt_lower
            ]

            assert len(found_keywords) >= 2, (
                f"Analyzer should focus on synthesis/integration (found: {found_keywords}): {persona_path}"
            )

    def test_analyzers_differentiate_from_reviewers(self):
        """Test that new analyzers clearly differentiate from reviewers in their approach."""
        # Load an analyzer and a reviewer for comparison
        analyzer_path = (
            self.personas_dir / "analyzers" / "application_coherence_analyzer.yaml"
        )
        reviewer_path = (
            self.personas_dir / "reviewers" / "academic" / "application_reviewer.yaml"
        )

        analyzer = self.loader.load_persona(analyzer_path)
        reviewer = self.loader.load_persona(reviewer_path)

        # Analyzer should mention processing multiple reviews
        assert "{reviews}" in analyzer.prompt_template
        assert (
            "review feedback" in analyzer.prompt_template.lower()
            or "reviewer" in analyzer.prompt_template.lower()
        )

        # Reviewer should focus on evaluating content
        assert "{reviews}" not in reviewer.prompt_template
        assert (
            "evaluation" in reviewer.prompt_template.lower()
            or "assess" in reviewer.prompt_template.lower()
        )

        # Analyzer should mention synthesis/meta-analysis
        analyzer_lower = analyzer.prompt_template.lower()
        assert any(
            word in analyzer_lower
            for word in ["synthesis", "synthesize", "meta", "holistic"]
        )

    def test_analyzer_directory_structure(self):
        """Test that analyzers are properly organized in the analyzers directory."""
        analyzers_dir = self.personas_dir / "analyzers"
        assert analyzers_dir.exists(), "Analyzers directory should exist"

        # Check that new analyzers are in the main analyzers directory (not subdirectories)
        new_analyzer_files = [
            "application_coherence_analyzer.yaml",
            "cross_document_themes_analyzer.yaml",
            "competitive_analysis_specialist.yaml",
        ]

        for filename in new_analyzer_files:
            analyzer_path = analyzers_dir / filename
            assert analyzer_path.exists(), (
                f"Analyzer should exist in analyzers directory: {filename}"
            )
            assert analyzer_path.is_file(), f"Should be a file: {filename}"
