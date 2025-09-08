"""
Document structure analysis tools.

These tools provide objective analysis of document organization and flow
that LLMs cannot reliably assess with precision.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParagraphInfo:
    """Information about a single paragraph."""

    index: int
    word_count: int
    sentence_count: int
    starts_with_transition: bool
    topic_sentence_strength: float  # 0-1 score
    contains_evidence: bool
    paragraph_type: str  # 'introduction', 'body', 'conclusion', 'transition'


@dataclass
class DocumentStructure:
    """Comprehensive document structure analysis."""

    total_paragraphs: int
    paragraphs: list[ParagraphInfo]

    # Structure components
    has_introduction: bool
    has_conclusion: bool
    body_paragraph_count: int

    # Balance metrics
    paragraph_length_variance: float  # Lower = more balanced
    shortest_paragraph: int  # word count
    longest_paragraph: int  # word count
    average_paragraph_length: float

    # Flow analysis
    transition_density: float  # transitions per paragraph
    topic_sentence_strength_avg: float
    structural_coherence_score: float  # 0-1 overall score


@dataclass
class EssayComponents:
    """Detected essay components and their quality."""

    introduction: dict[str, Any] | None = None
    body_paragraphs: list[dict[str, Any]] | None = None
    conclusion: dict[str, Any] | None = None

    # Component quality scores
    introduction_strength: float = 0.0  # 0-1
    body_development: float = 0.0  # 0-1
    conclusion_effectiveness: float = 0.0  # 0-1

    # Overall essay structure score
    overall_structure_score: float = 0.0  # 0-1


def analyze_document_structure(content: str) -> DocumentStructure:
    """
    Analyze document structure with objective metrics.

    Provides precise structural analysis that LLMs cannot reliably perform:
    - Exact paragraph counts and lengths (LLMs approximate)
    - Quantified balance metrics (LLMs give subjective assessments)
    - Objective flow measurements (LLMs provide vague feedback)

    Critical for college essays where structure significantly impacts scores.

    Args:
        content: Document content to analyze

    Returns:
        DocumentStructure with comprehensive structural metrics

    Example:
        >>> structure = analyze_document_structure(essay_content)
        >>> structure.paragraph_length_variance
        0.23  # Low variance = well-balanced paragraphs
        >>> structure.transition_density
        1.4   # Average transitions per paragraph
    """
    if not content or not content.strip():
        return DocumentStructure(
            total_paragraphs=0,
            paragraphs=[],
            has_introduction=False,
            has_conclusion=False,
            body_paragraph_count=0,
            paragraph_length_variance=0.0,
            shortest_paragraph=0,
            longest_paragraph=0,
            average_paragraph_length=0.0,
            transition_density=0.0,
            topic_sentence_strength_avg=0.0,
            structural_coherence_score=0.0,
        )

    # Extract paragraphs
    paragraphs_text = _extract_paragraphs(content)
    if not paragraphs_text:
        paragraphs_text = [content]  # Treat as single paragraph

    # Analyze each paragraph
    paragraphs_info = []
    for i, para_text in enumerate(paragraphs_text):
        para_info = _analyze_paragraph(
            i, para_text, i == 0, i == len(paragraphs_text) - 1
        )
        paragraphs_info.append(para_info)

    # Calculate structure metrics
    total_paragraphs = len(paragraphs_info)
    word_counts = [p.word_count for p in paragraphs_info]

    # Identify components
    has_introduction = (
        paragraphs_info[0].paragraph_type == "introduction"
        if paragraphs_info
        else False
    )
    has_conclusion = (
        paragraphs_info[-1].paragraph_type == "conclusion" if paragraphs_info else False
    )
    body_paragraph_count = sum(1 for p in paragraphs_info if p.paragraph_type == "body")

    # Balance metrics
    if word_counts:
        avg_length = sum(word_counts) / len(word_counts)
        variance = sum((count - avg_length) ** 2 for count in word_counts) / len(
            word_counts
        )
        paragraph_length_variance = variance / (avg_length**2) if avg_length > 0 else 0
        shortest_paragraph = min(word_counts)
        longest_paragraph = max(word_counts)
    else:
        avg_length = 0
        paragraph_length_variance = 0
        shortest_paragraph = 0
        longest_paragraph = 0

    # Flow metrics
    transitions_per_para = sum(1 for p in paragraphs_info if p.starts_with_transition)
    transition_density = transitions_per_para / max(total_paragraphs, 1)

    topic_strength_avg = sum(p.topic_sentence_strength for p in paragraphs_info) / max(
        total_paragraphs, 1
    )

    # Overall coherence score (0-1)
    coherence_score = _calculate_structural_coherence(
        paragraphs_info, has_introduction, has_conclusion
    )

    return DocumentStructure(
        total_paragraphs=total_paragraphs,
        paragraphs=paragraphs_info,
        has_introduction=has_introduction,
        has_conclusion=has_conclusion,
        body_paragraph_count=body_paragraph_count,
        paragraph_length_variance=paragraph_length_variance,
        shortest_paragraph=shortest_paragraph,
        longest_paragraph=longest_paragraph,
        average_paragraph_length=avg_length,
        transition_density=transition_density,
        topic_sentence_strength_avg=topic_strength_avg,
        structural_coherence_score=coherence_score,
    )


def detect_essay_components(content: str) -> EssayComponents:
    """
    Detect and analyze essay components (introduction, body, conclusion).

    Provides objective component identification that LLMs often misidentify.
    Essential for essay structure feedback.

    Args:
        content: Essay content to analyze

    Returns:
        EssayComponents with detected components and quality scores
    """
    paragraphs_text = _extract_paragraphs(content)
    if not paragraphs_text:
        return EssayComponents(body_paragraphs=[])

    components = EssayComponents(body_paragraphs=[])

    # Analyze introduction (first paragraph)
    if paragraphs_text:
        intro_analysis = _analyze_introduction(paragraphs_text[0])
        components.introduction = intro_analysis
        components.introduction_strength = intro_analysis.get("strength_score", 0.0)

    # Analyze body paragraphs (middle paragraphs)
    if len(paragraphs_text) > 2:
        body_paragraphs = paragraphs_text[1:-1]
    elif len(paragraphs_text) == 2:
        body_paragraphs = [paragraphs_text[1]]
    else:
        body_paragraphs = []

    body_analyses = []
    for i, body_para in enumerate(body_paragraphs):
        body_analysis = _analyze_body_paragraph(body_para, i)
        body_analyses.append(body_analysis)

    components.body_paragraphs = body_analyses
    components.body_development = _calculate_body_development_score(body_analyses)

    # Analyze conclusion (last paragraph, if different from intro)
    if len(paragraphs_text) > 1:
        conclusion_analysis = _analyze_conclusion(paragraphs_text[-1])
        components.conclusion = conclusion_analysis
        components.conclusion_effectiveness = conclusion_analysis.get(
            "effectiveness_score", 0.0
        )

    # Calculate overall structure score
    components.overall_structure_score = _calculate_overall_structure_score(components)

    return components


def analyze_paragraph_flow(content: str) -> dict[str, Any]:
    """
    Analyze paragraph-to-paragraph flow and transitions.

    Provides objective flow metrics that LLMs cannot quantify precisely.

    Args:
        content: Document content to analyze

    Returns:
        Dictionary with flow analysis metrics
    """
    paragraphs_text = _extract_paragraphs(content)
    if len(paragraphs_text) < 2:
        return {
            "paragraph_count": len(paragraphs_text),
            "transition_quality_avg": 0.0,
            "flow_consistency_score": 0.0,
            "abrupt_transitions": 0,
            "smooth_transitions": 0,
        }

    transition_scores = []
    abrupt_count = 0
    smooth_count = 0

    for i in range(len(paragraphs_text) - 1):
        current_para = paragraphs_text[i]
        next_para = paragraphs_text[i + 1]

        transition_score = _analyze_paragraph_transition(current_para, next_para)
        transition_scores.append(transition_score)

        if transition_score < 0.3:
            abrupt_count += 1
        elif transition_score > 0.7:
            smooth_count += 1

    avg_transition_quality = sum(transition_scores) / len(transition_scores)
    flow_consistency = (
        1.0 - (max(transition_scores) - min(transition_scores))
        if transition_scores
        else 0.0
    )

    return {
        "paragraph_count": len(paragraphs_text),
        "transition_quality_avg": avg_transition_quality,
        "flow_consistency_score": flow_consistency,
        "abrupt_transitions": abrupt_count,
        "smooth_transitions": smooth_count,
        "transition_scores": transition_scores,
    }


# Helper functions


def _extract_paragraphs(content: str) -> list[str]:
    """Extract paragraphs from content."""
    # Split on double newlines or single newlines with significant indentation
    paragraphs = re.split(r"\n\s*\n|\n\s{4,}", content)
    return [p.strip() for p in paragraphs if p.strip()]


def _analyze_paragraph(
    index: int, text: str, is_first: bool, is_last: bool
) -> ParagraphInfo:
    """Analyze a single paragraph."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Determine paragraph type
    if is_first:
        para_type = "introduction"
    elif is_last:
        para_type = "conclusion"
    else:
        para_type = "body"

    # Check for transition words at start
    starts_with_transition = _starts_with_transition(text)

    # Analyze topic sentence strength
    topic_strength = _analyze_topic_sentence_strength(sentences[0] if sentences else "")

    # Check for evidence/examples
    contains_evidence = _contains_evidence_markers(text)

    return ParagraphInfo(
        index=index,
        word_count=len(words),
        sentence_count=len(sentences),
        starts_with_transition=starts_with_transition,
        topic_sentence_strength=topic_strength,
        contains_evidence=contains_evidence,
        paragraph_type=para_type,
    )


def _starts_with_transition(text: str) -> bool:
    """Check if paragraph starts with transition word/phrase."""
    transition_starters = [
        "however",
        "furthermore",
        "moreover",
        "additionally",
        "meanwhile",
        "consequently",
        "therefore",
        "thus",
        "nevertheless",
        "on the other hand",
        "in contrast",
        "similarly",
        "likewise",
        "for example",
        "for instance",
        "first",
        "second",
        "third",
        "finally",
        "in conclusion",
        "to summarize",
    ]

    text_start = text.lower().strip()
    return any(text_start.startswith(transition) for transition in transition_starters)


def _analyze_topic_sentence_strength(sentence: str) -> float:
    """Analyze the strength of a topic sentence (0-1 score)."""
    if not sentence:
        return 0.0

    score = 0.5  # Base score

    # Positive indicators
    if len(sentence.split()) >= 8:  # Substantial length
        score += 0.1
    if any(
        word in sentence.lower()
        for word in ["because", "since", "due to", "as a result"]
    ):
        score += 0.1  # Causal reasoning
    if sentence.count(",") >= 1:  # Complex structure
        score += 0.1

    # Negative indicators
    if sentence.lower().startswith(("i ", "my ", "me ")):
        score -= 0.1  # Too personal for topic sentence
    if len(sentence.split()) < 5:  # Too short
        score -= 0.2

    return max(0.0, min(1.0, score))


def _contains_evidence_markers(text: str) -> bool:
    """Check if paragraph contains evidence or example markers."""
    evidence_markers = [
        "for example",
        "for instance",
        "such as",
        "including",
        "specifically",
        "according to",
        "research shows",
        "studies indicate",
        "data reveals",
        "statistics show",
        "evidence suggests",
        "as demonstrated by",
    ]

    text_lower = text.lower()
    return any(marker in text_lower for marker in evidence_markers)


def _analyze_introduction(intro_text: str) -> dict[str, Any]:
    """Analyze introduction paragraph quality."""
    words = intro_text.split()
    sentences = re.split(r"[.!?]+", intro_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Hook strength (first sentence engagement)
    hook_strength = _analyze_hook_strength(sentences[0] if sentences else "")

    # Context provision
    provides_context = _provides_context(intro_text)

    # Thesis presence
    has_thesis = _has_thesis_statement(sentences[-1] if sentences else "")

    # Calculate overall strength
    strength_score = (
        hook_strength + (0.3 if provides_context else 0) + (0.4 if has_thesis else 0)
    ) / 1.7

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "hook_strength": hook_strength,
        "provides_context": provides_context,
        "has_thesis": has_thesis,
        "strength_score": min(1.0, strength_score),
    }


def _analyze_body_paragraph(body_text: str, index: int) -> dict[str, Any]:
    """Analyze body paragraph quality."""
    words = body_text.split()
    sentences = re.split(r"[.!?]+", body_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Topic sentence analysis
    topic_sentence_strength = _analyze_topic_sentence_strength(
        sentences[0] if sentences else ""
    )

    # Evidence and examples
    has_evidence = _contains_evidence_markers(body_text)

    # Development depth
    development_score = min(1.0, len(words) / 100)  # Normalize to 100 words = 1.0

    return {
        "index": index,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "topic_sentence_strength": topic_sentence_strength,
        "has_evidence": has_evidence,
        "development_score": development_score,
    }


def _analyze_conclusion(conclusion_text: str) -> dict[str, Any]:
    """Analyze conclusion paragraph effectiveness."""
    words = conclusion_text.split()
    sentences = re.split(r"[.!?]+", conclusion_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Synthesis indicators
    synthesizes = _contains_synthesis_markers(conclusion_text)

    # Forward-looking elements
    forward_looking = _contains_forward_looking_elements(conclusion_text)

    # Avoids repetition
    avoids_repetition = not _contains_repetitive_phrases(conclusion_text)

    # Calculate effectiveness
    effectiveness_score = (
        (0.4 if synthesizes else 0)
        + (0.3 if forward_looking else 0)
        + (0.3 if avoids_repetition else 0)
    )

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "synthesizes": synthesizes,
        "forward_looking": forward_looking,
        "avoids_repetition": avoids_repetition,
        "effectiveness_score": effectiveness_score,
    }


def _analyze_hook_strength(first_sentence: str) -> float:
    """Analyze the strength of an opening hook (0-1 score)."""
    if not first_sentence:
        return 0.0

    score = 0.3  # Base score

    # Strong hook indicators
    if first_sentence.endswith("?"):  # Question
        score += 0.3
    if any(
        word in first_sentence.lower() for word in ["imagine", "picture", "consider"]
    ):
        score += 0.2  # Engaging language
    if '"' in first_sentence:  # Quote
        score += 0.2
    if any(
        word in first_sentence.lower()
        for word in ["never", "always", "everyone", "no one"]
    ):
        score += 0.1  # Bold statements

    # Weak hook indicators
    if first_sentence.lower().startswith("in today's society"):
        score -= 0.3  # Cliché
    if first_sentence.lower().startswith("since the beginning of time"):
        score -= 0.3  # Cliché

    return max(0.0, min(1.0, score))


def _provides_context(intro_text: str) -> bool:
    """Check if introduction provides sufficient context."""
    context_indicators = [
        "background",
        "history",
        "context",
        "situation",
        "currently",
        "today",
        "recent",
        "modern",
        "contemporary",
        "society",
    ]

    text_lower = intro_text.lower()
    return any(indicator in text_lower for indicator in context_indicators)


def _has_thesis_statement(last_sentence: str) -> bool:
    """Check if sentence appears to be a thesis statement."""
    if not last_sentence or len(last_sentence.split()) < 8:
        return False

    thesis_indicators = [
        "will",
        "should",
        "must",
        "argue",
        "demonstrate",
        "show",
        "prove",
        "examine",
        "explore",
        "analyze",
        "because",
    ]

    sentence_lower = last_sentence.lower()
    return any(indicator in sentence_lower for indicator in thesis_indicators)


def _contains_synthesis_markers(text: str) -> bool:
    """Check for synthesis/summary language in conclusion."""
    synthesis_markers = [
        "in conclusion",
        "to summarize",
        "in summary",
        "overall",
        "ultimately",
        "in the end",
        "therefore",
        "thus",
        "hence",
    ]

    text_lower = text.lower()
    return any(marker in text_lower for marker in synthesis_markers)


def _contains_forward_looking_elements(text: str) -> bool:
    """Check for forward-looking elements in conclusion."""
    forward_markers = [
        "future",
        "will",
        "should",
        "must",
        "need to",
        "going forward",
        "next",
        "continue",
        "further",
        "additional",
        "more",
    ]

    text_lower = text.lower()
    return any(marker in text_lower for marker in forward_markers)


def _contains_repetitive_phrases(text: str) -> bool:
    """Check for repetitive conclusion phrases (clichés)."""
    repetitive_phrases = [
        "in conclusion",
        "to conclude",
        "in summary",
        "to summarize",
        "as i have shown",
        "as stated above",
        "as mentioned earlier",
    ]

    text_lower = text.lower()
    return any(phrase in text_lower for phrase in repetitive_phrases)


def _calculate_body_development_score(body_analyses: list[dict[str, Any]]) -> float:
    """Calculate overall body paragraph development score."""
    if not body_analyses:
        return 0.0

    total_score = 0.0
    for analysis in body_analyses:
        para_score = (
            analysis["topic_sentence_strength"] * 0.4
            + (0.3 if analysis["has_evidence"] else 0)
            + analysis["development_score"] * 0.3
        )
        total_score += para_score

    return total_score / len(body_analyses)


def _calculate_overall_structure_score(components: EssayComponents) -> float:
    """Calculate overall essay structure score."""
    intro_weight = 0.25
    body_weight = 0.5
    conclusion_weight = 0.25

    score = (
        components.introduction_strength * intro_weight
        + components.body_development * body_weight
        + components.conclusion_effectiveness * conclusion_weight
    )

    return score


def _calculate_structural_coherence(
    paragraphs: list[ParagraphInfo], has_intro: bool, has_conclusion: bool
) -> float:
    """Calculate overall structural coherence score (0-1)."""
    if not paragraphs:
        return 0.0

    score = 0.0

    # Component presence (0.4 max)
    if has_intro:
        score += 0.2
    if has_conclusion:
        score += 0.2

    # Paragraph balance (0.3 max)
    if len(paragraphs) >= 3:  # Minimum structure
        score += 0.1

    word_counts = [p.word_count for p in paragraphs]
    if word_counts:
        avg_length = sum(word_counts) / len(word_counts)
        balance_score = 1.0 - (max(word_counts) - min(word_counts)) / max(avg_length, 1)
        score += balance_score * 0.2

    # Transition usage (0.3 max)
    transition_ratio = sum(1 for p in paragraphs if p.starts_with_transition) / len(
        paragraphs
    )
    score += min(0.3, transition_ratio * 0.6)  # Cap at 0.3

    return min(1.0, score)


def _analyze_paragraph_transition(current_para: str, next_para: str) -> float:
    """Analyze transition quality between paragraphs (0-1 score)."""
    # Get last sentence of current paragraph
    current_sentences = re.split(r"[.!?]+", current_para)
    current_last = current_sentences[-2] if len(current_sentences) > 1 else ""

    # Get first sentence of next paragraph
    next_sentences = re.split(r"[.!?]+", next_para)
    next_first = next_sentences[0] if next_sentences else ""

    score = 0.3  # Base score

    # Transition word in next paragraph start
    if _starts_with_transition(next_para):
        score += 0.4

    # Thematic connection (simple keyword overlap)
    current_words = set(current_last.lower().split())
    next_words = set(next_first.lower().split())
    overlap = len(current_words & next_words)
    if overlap > 0:
        score += min(0.3, overlap * 0.1)

    return min(1.0, score)
