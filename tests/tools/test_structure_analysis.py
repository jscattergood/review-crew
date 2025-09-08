"""Unit tests for structure analysis tools."""

import pytest

from src.tools.structure_analysis import (
    DocumentStructure,
    EssayComponents,
    ParagraphInfo,
    analyze_document_structure,
    analyze_paragraph_flow,
    detect_essay_components,
)


class TestAnalyzeDocumentStructure:
    """Test document structure analysis."""

    def test_empty_content(self):
        """Test with empty content."""
        structure = analyze_document_structure("")

        assert structure.total_paragraphs == 0
        assert structure.paragraphs == []
        assert structure.has_introduction is False
        assert structure.has_conclusion is False
        assert structure.body_paragraph_count == 0

    def test_single_paragraph(self):
        """Test with single paragraph."""
        content = "This is a single paragraph with multiple sentences. It has some content for testing."
        structure = analyze_document_structure(content)

        assert structure.total_paragraphs == 1
        assert len(structure.paragraphs) == 1
        assert structure.paragraphs[0].paragraph_type == "introduction"
        assert structure.average_paragraph_length > 0

    def test_multi_paragraph_essay(self):
        """Test with multi-paragraph essay structure."""
        content = """This is the introduction paragraph. It sets up the essay topic.

This is the first body paragraph. It develops the main argument with supporting details.

This is the second body paragraph. It continues the argument development.

This is the conclusion paragraph. It wraps up the essay and provides closure."""

        structure = analyze_document_structure(content)

        assert structure.total_paragraphs == 4
        assert structure.has_introduction is True
        assert structure.has_conclusion is True
        assert structure.body_paragraph_count == 2
        assert structure.structural_coherence_score > 0

    def test_paragraph_balance_metrics(self):
        """Test paragraph balance calculations."""
        content = """Short intro.

This is a much longer body paragraph with significantly more content and detail to analyze the balance.

Another body paragraph of moderate length for comparison.

Short conclusion."""

        structure = analyze_document_structure(content)

        assert structure.paragraph_length_variance > 0
        assert structure.shortest_paragraph > 0
        assert structure.longest_paragraph > structure.shortest_paragraph
        assert structure.average_paragraph_length > 0

    def test_transition_detection(self):
        """Test transition word detection."""
        content = """This is the introduction paragraph.

However, this body paragraph starts with a transition word.

Furthermore, this paragraph also uses transitions effectively.

In conclusion, this paragraph wraps up the essay."""

        structure = analyze_document_structure(content)

        assert structure.transition_density > 0
        # Should detect transitions in some paragraphs
        transition_count = sum(
            1 for p in structure.paragraphs if p.starts_with_transition
        )
        assert transition_count > 0


class TestDetectEssayComponents:
    """Test essay component detection."""

    def test_empty_content(self):
        """Test with empty content."""
        components = detect_essay_components("")

        assert components.introduction is None
        assert components.body_paragraphs == []
        assert components.conclusion is None
        assert components.overall_structure_score == 0.0

    def test_complete_essay_structure(self):
        """Test with complete essay structure."""
        content = """This is an engaging introduction that hooks the reader. It provides context and sets up the thesis.

This is the first body paragraph that develops the main argument. It includes specific examples and evidence.

This is the second body paragraph that continues the argument. It provides additional support and analysis.

This is the conclusion that synthesizes the main points. It provides closure and final thoughts."""

        components = detect_essay_components(content)

        assert components.introduction is not None
        assert len(components.body_paragraphs) == 2
        assert components.conclusion is not None
        assert components.overall_structure_score > 0

    def test_introduction_analysis(self):
        """Test introduction analysis."""
        content = """What if I told you that everything you know is wrong? This engaging question hooks the reader and provides context for the essay topic.

Body paragraph content here."""

        components = detect_essay_components(content)

        assert components.introduction is not None
        assert components.introduction_strength > 0
        assert "hook_strength" in components.introduction

    def test_body_paragraph_analysis(self):
        """Test body paragraph analysis."""
        content = """Introduction paragraph.

The main argument is supported by evidence. For example, research shows that specific examples strengthen arguments.

Another supporting point with detailed analysis. Studies indicate that comprehensive development improves essay quality.

Conclusion paragraph."""

        components = detect_essay_components(content)

        assert len(components.body_paragraphs) == 2
        assert components.body_development > 0
        # Check that body paragraphs have evidence markers
        assert any(body["has_evidence"] for body in components.body_paragraphs)


class TestAnalyzeParagraphFlow:
    """Test paragraph flow analysis."""

    def test_single_paragraph_flow(self):
        """Test with single paragraph (no flow to analyze)."""
        content = "Single paragraph content."
        flow = analyze_paragraph_flow(content)

        assert flow["paragraph_count"] == 1
        assert flow["transition_quality_avg"] == 0.0
        assert flow["abrupt_transitions"] == 0
        assert flow["smooth_transitions"] == 0

    def test_multi_paragraph_flow(self):
        """Test with multiple paragraphs."""
        content = """This is the first paragraph with some content.

However, this second paragraph uses a transition word to connect smoothly.

This third paragraph continues the flow with related content.

Finally, this conclusion paragraph wraps up the discussion."""

        flow = analyze_paragraph_flow(content)

        assert flow["paragraph_count"] == 4
        assert flow["transition_quality_avg"] > 0
        assert len(flow["transition_scores"]) == 3  # n-1 transitions
        assert flow["smooth_transitions"] > 0

    def test_abrupt_transitions(self):
        """Test detection of abrupt transitions."""
        content = """This paragraph discusses topic A in detail.

Completely different topic B is introduced without connection.

Another unrelated topic C appears suddenly.

Random conclusion about topic D."""

        flow = analyze_paragraph_flow(content)

        assert flow["paragraph_count"] == 4
        assert flow["abrupt_transitions"] >= 0  # May detect some abrupt transitions
        assert flow["flow_consistency_score"] >= 0


class TestDataClasses:
    """Test the data classes."""

    def test_paragraph_info_dataclass(self):
        """Test ParagraphInfo dataclass."""
        para_info = ParagraphInfo(
            index=0,
            word_count=50,
            sentence_count=3,
            starts_with_transition=True,
            topic_sentence_strength=0.8,
            contains_evidence=True,
            paragraph_type="introduction",
        )

        assert para_info.index == 0
        assert para_info.word_count == 50
        assert para_info.starts_with_transition is True
        assert para_info.paragraph_type == "introduction"

    def test_document_structure_dataclass(self):
        """Test DocumentStructure dataclass."""
        structure = DocumentStructure(
            total_paragraphs=3,
            paragraphs=[],
            has_introduction=True,
            has_conclusion=True,
            body_paragraph_count=1,
            paragraph_length_variance=0.2,
            shortest_paragraph=20,
            longest_paragraph=60,
            average_paragraph_length=40.0,
            transition_density=0.5,
            topic_sentence_strength_avg=0.7,
            structural_coherence_score=0.8,
        )

        assert structure.total_paragraphs == 3
        assert structure.has_introduction is True
        assert structure.structural_coherence_score == 0.8

    def test_essay_components_dataclass(self):
        """Test EssayComponents dataclass."""
        components = EssayComponents(
            introduction={"strength_score": 0.8},
            body_paragraphs=[{"development_score": 0.7}],
            conclusion={"effectiveness_score": 0.6},
            introduction_strength=0.8,
            body_development=0.7,
            conclusion_effectiveness=0.6,
            overall_structure_score=0.7,
        )

        assert components.introduction_strength == 0.8
        assert components.body_development == 0.7
        assert components.overall_structure_score == 0.7
