"""
Academic writing analysis tools specifically designed for college applications.

These tools provide specialized analysis for college essays, personal statements,
and academic writing that requires objective assessment of quality factors.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from .logging_utils import log_tool_execution


@dataclass
class EssayStrengthAnalysis:
    """Analysis of essay strength and impact."""

    # Narrative elements
    has_compelling_story: bool
    personal_details_count: int
    specific_examples_count: int

    # Voice and authenticity
    personal_voice_strength: float  # 0-1
    authenticity_score: float  # 0-1
    generic_language_percentage: float

    # Impact factors
    memorable_moments_count: int
    emotional_resonance_score: float  # 0-1
    uniqueness_score: float  # 0-1

    # Overall assessment
    admissions_strength_score: float  # 0-1


@dataclass
class ClicheAnalysis:
    """Analysis of cliché usage in writing."""

    total_cliches_found: int
    cliche_density: float  # Clichés per 100 words
    cliche_categories: dict[str, int]  # Category -> count
    specific_cliches: list[dict[str, Any]]  # Detailed cliché instances

    # Severity assessment
    severity_score: float  # 0-1 (1 = very clichéd)
    admissions_risk_level: str  # 'low', 'medium', 'high'


@dataclass
class PersonalVoiceAnalysis:
    """Analysis of personal voice and authenticity."""

    # Voice indicators
    first_person_usage: float  # Percentage of sentences
    personal_anecdotes_count: int
    specific_details_count: int

    # Authenticity markers
    conversational_tone_score: float  # 0-1
    vulnerability_indicators: int
    reflection_depth_score: float  # 0-1

    # Voice strength
    voice_consistency_score: float  # 0-1
    authenticity_score: float  # 0-1
    generic_vs_personal_ratio: float


@log_tool_execution("essay_strength_analysis")
def analyze_essay_strength(content: str) -> EssayStrengthAnalysis:
    """
    Analyze essay strength for college admissions impact.

    Provides objective metrics for essay quality factors that admissions officers
    value but LLMs cannot reliably assess:
    - Specific vs. generic content (LLMs miss subtle distinctions)
    - Narrative structure strength (LLMs give vague assessments)
    - Memorable moment identification (LLMs cannot predict memorability)

    Critical for college essays where strength determines admission outcomes.

    Args:
        content: Essay content to analyze

    Returns:
        EssayStrengthAnalysis with comprehensive strength metrics

    Example:
        >>> analysis = analyze_essay_strength(essay_content)
        >>> analysis.admissions_strength_score
        0.78  # Strong essay likely to stand out
        >>> analysis.specific_examples_count
        4     # Good use of concrete examples
    """
    if not content or not content.strip():
        return EssayStrengthAnalysis(
            has_compelling_story=False,
            personal_details_count=0,
            specific_examples_count=0,
            personal_voice_strength=0.0,
            authenticity_score=0.0,
            generic_language_percentage=100.0,
            memorable_moments_count=0,
            emotional_resonance_score=0.0,
            uniqueness_score=0.0,
            admissions_strength_score=0.0,
        )

    # Analyze narrative elements
    has_story = _has_compelling_narrative(content)
    personal_details = _count_personal_details(content)
    specific_examples = _count_specific_examples(content)

    # Analyze voice and authenticity
    voice_strength = _analyze_personal_voice_strength(content)
    authenticity = _calculate_authenticity_score(content)
    generic_percentage = _calculate_generic_language_percentage(content)

    # Analyze impact factors
    memorable_moments = _count_memorable_moments(content)
    emotional_resonance = _calculate_emotional_resonance(content)
    uniqueness = _calculate_uniqueness_score(content)

    # Calculate overall admissions strength
    admissions_strength = _calculate_admissions_strength(
        has_story,
        personal_details,
        specific_examples,
        voice_strength,
        authenticity,
        generic_percentage,
        memorable_moments,
        emotional_resonance,
        uniqueness,
    )

    return EssayStrengthAnalysis(
        has_compelling_story=has_story,
        personal_details_count=personal_details,
        specific_examples_count=specific_examples,
        personal_voice_strength=voice_strength,
        authenticity_score=authenticity,
        generic_language_percentage=generic_percentage,
        memorable_moments_count=memorable_moments,
        emotional_resonance_score=emotional_resonance,
        uniqueness_score=uniqueness,
        admissions_strength_score=admissions_strength,
    )


@log_tool_execution("cliche_detection")
def detect_cliches(content: str) -> ClicheAnalysis:
    """
    Detect clichés and overused phrases that hurt college application essays.

    Provides precise cliché identification that LLMs often miss or misidentify:
    - Exact phrase matching (LLMs approximate and miss variations)
    - Categorized cliché types (LLMs don't systematically categorize)
    - Density calculations (LLMs cannot count precisely)

    Critical because clichés are automatic red flags for admissions officers.

    Args:
        content: Text content to analyze for clichés

    Returns:
        ClicheAnalysis with detailed cliché detection results

    Example:
        >>> analysis = detect_cliches(essay_content)
        >>> analysis.total_cliches_found
        7
        >>> analysis.admissions_risk_level
        'high'  # Too many clichés detected
    """
    if not content:
        return ClicheAnalysis(
            total_cliches_found=0,
            cliche_density=0.0,
            cliche_categories={},
            specific_cliches=[],
            severity_score=0.0,
            admissions_risk_level="low",
        )

    # Load cliché database
    cliche_database = _get_college_essay_cliches()

    # Find all clichés
    found_cliches = []
    category_counts = {}

    content_lower = content.lower()

    for category, cliches in cliche_database.items():
        category_counts[category] = 0

        for cliche_info in cliches:
            phrase = str(cliche_info["phrase"])
            severity = str(cliche_info["severity"])

            # Count occurrences
            count = content_lower.count(phrase.lower())
            if count > 0:
                category_counts[category] += count
                found_cliches.append(
                    {
                        "phrase": phrase,
                        "category": category,
                        "severity": severity,
                        "count": count,
                        "impact": cliche_info.get("impact", "Reduces originality"),
                    }
                )

    # Calculate metrics
    total_cliches = sum(category_counts.values())
    word_count = len(content.split())
    cliche_density = (total_cliches / max(word_count, 1)) * 100

    # Calculate severity score
    severity_score = min(1.0, cliche_density / 5.0)  # 5% density = max severity

    # Determine risk level
    if cliche_density >= 3.0:
        risk_level = "high"
    elif cliche_density >= 1.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return ClicheAnalysis(
        total_cliches_found=total_cliches,
        cliche_density=cliche_density,
        cliche_categories=category_counts,
        specific_cliches=found_cliches,
        severity_score=severity_score,
        admissions_risk_level=risk_level,
    )


def analyze_personal_voice(content: str) -> PersonalVoiceAnalysis:
    """
    Analyze personal voice strength and authenticity in writing.

    Provides objective voice analysis that LLMs cannot reliably perform:
    - Quantified personal vs. generic language (LLMs give subjective assessments)
    - Specific authenticity markers (LLMs miss subtle indicators)
    - Voice consistency measurement (LLMs cannot track consistency precisely)

    Essential for college essays where authentic personal voice is crucial.

    Args:
        content: Text content to analyze

    Returns:
        PersonalVoiceAnalysis with detailed voice strength metrics
    """
    if not content:
        return PersonalVoiceAnalysis(
            first_person_usage=0.0,
            personal_anecdotes_count=0,
            specific_details_count=0,
            conversational_tone_score=0.0,
            vulnerability_indicators=0,
            reflection_depth_score=0.0,
            voice_consistency_score=0.0,
            authenticity_score=0.0,
            generic_vs_personal_ratio=1.0,
        )

    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Analyze first-person usage
    first_person_count = sum(1 for s in sentences if _contains_first_person(s))
    first_person_usage = (first_person_count / max(len(sentences), 1)) * 100

    # Count personal elements
    anecdotes_count = _count_personal_anecdotes(content)
    specific_details = _count_specific_details(content)

    # Analyze tone and authenticity
    conversational_score = _calculate_conversational_tone(content)
    vulnerability_count = _count_vulnerability_indicators(content)
    reflection_score = _calculate_reflection_depth(content)

    # Calculate voice consistency
    voice_consistency = _calculate_voice_consistency(sentences)

    # Calculate overall authenticity
    authenticity = _calculate_voice_authenticity(
        first_person_usage,
        anecdotes_count,
        specific_details,
        conversational_score,
        vulnerability_count,
        reflection_score,
    )

    # Calculate generic vs. personal ratio
    generic_markers = _count_generic_language_markers(content)
    personal_markers = anecdotes_count + specific_details + vulnerability_count
    generic_ratio = generic_markers / max(personal_markers, 1)

    return PersonalVoiceAnalysis(
        first_person_usage=first_person_usage,
        personal_anecdotes_count=anecdotes_count,
        specific_details_count=specific_details,
        conversational_tone_score=conversational_score,
        vulnerability_indicators=vulnerability_count,
        reflection_depth_score=reflection_score,
        voice_consistency_score=voice_consistency,
        authenticity_score=authenticity,
        generic_vs_personal_ratio=generic_ratio,
    )


# Helper functions


def _has_compelling_narrative(content: str) -> bool:
    """Check if content contains a compelling narrative structure."""
    narrative_indicators = [
        # Story progression
        "when i",
        "as i",
        "after i",
        "before i",
        "while i",
        # Temporal markers
        "first",
        "then",
        "next",
        "finally",
        "eventually",
        # Scene setting
        "remember",
        "moment",
        "day",
        "time",
        "experience",
        # Conflict/resolution
        "challenge",
        "problem",
        "struggle",
        "overcome",
        "learned",
    ]

    content_lower = content.lower()
    indicator_count = sum(
        1 for indicator in narrative_indicators if indicator in content_lower
    )

    return indicator_count >= 3  # Minimum threshold for narrative structure


def _count_personal_details(content: str) -> int:
    """Count specific personal details that make writing memorable."""
    detail_patterns = [
        r"\b\d+\s*(years?|months?|weeks?|days?)\b",  # Specific time periods
        r"\b\d+\s*(dollars?|cents?)\b",  # Specific amounts
        r"\b\d+\s*(miles?|feet|inches?)\b",  # Specific measurements
        r"\b\d+\s*(people|students?|friends?)\b",  # Specific quantities
        r'"[^"]*"',  # Direct quotes
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Proper names (simplified)
    ]

    count = 0
    for pattern in detail_patterns:
        matches = re.findall(pattern, content)
        count += len(matches)

    return count


def _count_specific_examples(content: str) -> int:
    """Count specific examples and concrete instances."""
    example_markers = [
        "for example",
        "for instance",
        "such as",
        "like when",
        "specifically",
        "in particular",
        "one time",
        "once",
    ]

    content_lower = content.lower()
    return sum(1 for marker in example_markers if marker in content_lower)


def _analyze_personal_voice_strength(content: str) -> float:
    """Analyze strength of personal voice (0-1 score)."""
    sentences = re.split(r"[.!?]+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    score = 0.0

    # First-person perspective
    first_person_count = sum(1 for s in sentences if _contains_first_person(s))
    first_person_ratio = first_person_count / len(sentences)
    score += min(0.3, first_person_ratio * 0.6)

    # Personal anecdotes
    anecdote_count = _count_personal_anecdotes(content)
    score += min(0.2, anecdote_count * 0.05)

    # Conversational elements
    conversational_score = _calculate_conversational_tone(content)
    score += conversational_score * 0.3

    # Vulnerability/authenticity
    vulnerability_count = _count_vulnerability_indicators(content)
    score += min(0.2, vulnerability_count * 0.04)

    return min(1.0, score)


def _calculate_authenticity_score(content: str) -> float:
    """Calculate authenticity score based on multiple factors."""
    # Specific details vs. generic statements
    specific_count = _count_specific_details(content)
    generic_count = _count_generic_language_markers(content)

    specificity_ratio = specific_count / max(generic_count, 1)
    specificity_score = min(1.0, specificity_ratio / 2.0)

    # Personal reflection indicators
    reflection_score = _calculate_reflection_depth(content)

    # Vulnerability indicators
    vulnerability_count = _count_vulnerability_indicators(content)
    vulnerability_score = min(1.0, vulnerability_count * 0.1)

    # Combined authenticity score
    return specificity_score * 0.4 + reflection_score * 0.4 + vulnerability_score * 0.2


def _calculate_generic_language_percentage(content: str) -> float:
    """Calculate percentage of generic/clichéd language."""
    words = content.split()
    if not words:
        return 100.0

    generic_phrases = [
        "in today's society",
        "since the beginning of time",
        "throughout history",
        "in conclusion",
        "as a result",
        "it is important to note",
        "plays a vital role",
        "is extremely important",
        "has always been",
    ]

    generic_count = 0
    content_lower = content.lower()

    for phrase in generic_phrases:
        generic_count += content_lower.count(phrase) * len(phrase.split())

    return (generic_count / len(words)) * 100


def _count_memorable_moments(content: str) -> int:
    """Count potentially memorable moments in the narrative."""
    memorable_indicators = [
        "realized",
        "discovered",
        "understood",
        "learned",
        "moment",
        "suddenly",
        "finally",
        "breakthrough",
        "turning point",
        "epiphany",
        "revelation",
    ]

    content_lower = content.lower()
    return sum(1 for indicator in memorable_indicators if indicator in content_lower)


def _calculate_emotional_resonance(content: str) -> float:
    """Calculate emotional resonance score (0-1)."""
    emotion_words = [
        # Positive emotions
        "joy",
        "happy",
        "excited",
        "proud",
        "grateful",
        "amazed",
        # Negative emotions
        "sad",
        "frustrated",
        "angry",
        "disappointed",
        "worried",
        "scared",
        # Complex emotions
        "conflicted",
        "overwhelmed",
        "determined",
        "hopeful",
        "nervous",
    ]

    content_lower = content.lower()
    emotion_count = sum(1 for emotion in emotion_words if emotion in content_lower)

    # Normalize by content length
    words = content.split()
    emotion_density = emotion_count / max(len(words), 1)

    return min(1.0, emotion_density * 50)  # Scale to 0-1


def _calculate_uniqueness_score(content: str) -> float:
    """Calculate uniqueness/originality score (0-1)."""
    # Check for unique elements
    unique_elements = 0

    # Specific numbers/dates
    if re.search(r"\b\d{4}\b", content):  # Years
        unique_elements += 1
    if re.search(r"\b\d+\.\d+\b", content):  # Decimals
        unique_elements += 1

    # Quotes
    if '"' in content:
        unique_elements += 1

    # Proper nouns (simplified detection)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", content)
    if len(proper_nouns) > 5:
        unique_elements += 1

    # Technical terms or specialized vocabulary
    if any(
        word in content.lower() for word in ["algorithm", "hypothesis", "methodology"]
    ):
        unique_elements += 1

    return min(1.0, unique_elements * 0.2)


def _calculate_admissions_strength(
    has_story: bool,
    personal_details: int,
    specific_examples: int,
    voice_strength: float,
    authenticity: float,
    generic_percentage: float,
    memorable_moments: int,
    emotional_resonance: float,
    uniqueness: float,
) -> float:
    """Calculate overall admissions strength score."""
    score = 0.0

    # Narrative strength (25%)
    if has_story:
        score += 0.15
    score += min(0.1, specific_examples * 0.02)

    # Voice and authenticity (35%)
    score += voice_strength * 0.2
    score += authenticity * 0.15

    # Originality (25%)
    generic_penalty = (generic_percentage / 100) * 0.15
    score -= generic_penalty
    score += uniqueness * 0.1

    # Impact factors (15%)
    score += min(0.1, memorable_moments * 0.02)
    score += emotional_resonance * 0.05

    return max(0.0, min(1.0, score))


def _get_college_essay_cliches() -> dict[str, list[dict[str, Any]]]:
    """Get database of college essay clichés categorized by type."""
    return {
        "openings": [
            {
                "phrase": "in today's society",
                "severity": "high",
                "impact": "Immediate red flag",
            },
            {
                "phrase": "since the beginning of time",
                "severity": "high",
                "impact": "Clichéd opening",
            },
            {
                "phrase": "throughout history",
                "severity": "medium",
                "impact": "Generic start",
            },
            {"phrase": "imagine if", "severity": "medium", "impact": "Overused hook"},
        ],
        "conclusions": [
            {
                "phrase": "in conclusion",
                "severity": "high",
                "impact": "Weak conclusion marker",
            },
            {"phrase": "to conclude", "severity": "high", "impact": "Formulaic ending"},
            {
                "phrase": "in summary",
                "severity": "medium",
                "impact": "Repetitive conclusion",
            },
            {
                "phrase": "as you can see",
                "severity": "medium",
                "impact": "Obvious statement",
            },
        ],
        "generic_phrases": [
            {
                "phrase": "plays a vital role",
                "severity": "high",
                "impact": "Corporate speak",
            },
            {
                "phrase": "is extremely important",
                "severity": "medium",
                "impact": "Vague importance",
            },
            {
                "phrase": "has always been",
                "severity": "medium",
                "impact": "Overgeneralization",
            },
            {
                "phrase": "it goes without saying",
                "severity": "medium",
                "impact": "Redundant phrase",
            },
        ],
        "admissions_specific": [
            {
                "phrase": "ever since i was little",
                "severity": "high",
                "impact": "Overused personal start",
            },
            {
                "phrase": "i have always wanted to",
                "severity": "high",
                "impact": "Generic aspiration",
            },
            {
                "phrase": "my passion for",
                "severity": "medium",
                "impact": "Overused passion claim",
            },
            {
                "phrase": "i learned so much",
                "severity": "medium",
                "impact": "Vague learning claim",
            },
        ],
    }


def _contains_first_person(sentence: str) -> bool:
    """Check if sentence contains first-person perspective."""
    first_person_words = ["i ", "my ", "me ", "myself", "mine"]
    sentence_lower = sentence.lower()
    return any(word in sentence_lower for word in first_person_words)


def _count_personal_anecdotes(content: str) -> int:
    """Count personal anecdotes and stories."""
    anecdote_markers = [
        "when i",
        "i remember",
        "one day",
        "last year",
        "during",
        "i was",
        "i had",
        "i found",
        "i discovered",
        "i realized",
    ]

    content_lower = content.lower()
    return sum(1 for marker in anecdote_markers if marker in content_lower)


def _count_specific_details(content: str) -> int:
    """Count specific, concrete details."""
    # Numbers, dates, names, quotes, specific locations
    detail_count = 0

    # Numbers
    detail_count += len(re.findall(r"\b\d+\b", content))

    # Quotes
    detail_count += content.count('"')

    # Specific time references
    time_words = ["monday", "tuesday", "january", "february", "morning", "afternoon"]
    content_lower = content.lower()
    detail_count += sum(1 for word in time_words if word in content_lower)

    return detail_count


def _calculate_conversational_tone(content: str) -> float:
    """Calculate conversational tone score (0-1)."""
    conversational_markers = [
        # Contractions
        "'m ",
        "'re ",
        "'ve ",
        "'ll ",
        "'d ",
        "n't ",
        # Informal words
        "really",
        "pretty",
        "quite",
        "sort of",
        "kind of",
        # Questions
        "?",
    ]

    marker_count = 0
    for marker in conversational_markers:
        marker_count += content.count(marker)

    words = content.split()
    conversational_density = marker_count / max(len(words), 1)

    return min(1.0, conversational_density * 20)


def _count_vulnerability_indicators(content: str) -> int:
    """Count indicators of vulnerability and openness."""
    vulnerability_words = [
        "struggled",
        "failed",
        "mistake",
        "wrong",
        "difficult",
        "scared",
        "nervous",
        "worried",
        "confused",
        "uncertain",
        "admitted",
        "confess",
        "honest",
        "vulnerable",
        "insecure",
    ]

    content_lower = content.lower()
    return sum(1 for word in vulnerability_words if word in content_lower)


def _calculate_reflection_depth(content: str) -> float:
    """Calculate depth of reflection and introspection (0-1)."""
    reflection_markers = [
        "realized",
        "understood",
        "learned",
        "discovered",
        "recognized",
        "began to see",
        "came to understand",
        "it dawned on me",
        "looking back",
        "in retrospect",
        "now i know",
    ]

    content_lower = content.lower()
    reflection_count = sum(
        1 for marker in reflection_markers if marker in content_lower
    )

    # Normalize by content length
    sentences = re.split(r"[.!?]+", content)
    reflection_density = reflection_count / max(len(sentences), 1)

    return min(1.0, reflection_density * 3)


def _calculate_voice_consistency(sentences: list[str]) -> float:
    """Calculate consistency of voice throughout the text (0-1)."""
    if len(sentences) < 2:
        return 1.0

    # Check consistency of first-person usage
    first_person_usage = [_contains_first_person(s) for s in sentences]
    consistency_score = sum(first_person_usage) / len(first_person_usage)

    # Penalize extreme inconsistency (switching back and forth)
    if 0.2 < consistency_score < 0.8:
        consistency_score *= 0.7  # Penalty for mixed perspective

    return consistency_score


def _calculate_voice_authenticity(
    first_person_usage: float,
    anecdotes: int,
    details: int,
    conversational: float,
    vulnerability: int,
    reflection: float,
) -> float:
    """Calculate overall voice authenticity score."""
    # Normalize first-person usage (optimal around 60-80%)
    fp_score = 1.0 - abs(70 - first_person_usage) / 70

    # Anecdotes and details (more is better, up to a point)
    content_score = min(1.0, (anecdotes + details) * 0.05)

    # Conversational tone (moderate is best)
    tone_score = conversational if conversational < 0.7 else (1.0 - conversational)

    # Vulnerability (some is good, too much may be concerning)
    vuln_score = min(1.0, vulnerability * 0.1) if vulnerability <= 5 else 0.5

    # Combine scores
    return (
        fp_score * 0.3
        + content_score * 0.3
        + tone_score * 0.2
        + vuln_score * 0.1
        + reflection * 0.1
    )


def _count_generic_language_markers(content: str) -> int:
    """Count generic language markers."""
    generic_markers = [
        "very",
        "really",
        "quite",
        "extremely",
        "incredibly",
        "amazing",
        "awesome",
        "great",
        "good",
        "nice",
        "important",
        "significant",
        "meaningful",
        "valuable",
    ]

    content_lower = content.lower()
    return sum(content_lower.count(marker) for marker in generic_markers)
