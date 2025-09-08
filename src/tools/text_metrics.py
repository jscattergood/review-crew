"""
Text metrics and constraint validation tools.

These tools provide precise, deterministic analysis of text properties that LLMs
cannot reliably measure. Essential for writing with word/character limits.
"""

import re
import string
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TextMetrics:
    """Comprehensive text metrics."""

    # Basic counts
    word_count: int
    character_count: int
    character_count_no_spaces: int
    sentence_count: int
    paragraph_count: int

    # Advanced metrics
    average_words_per_sentence: float
    average_sentences_per_paragraph: float
    vocabulary_diversity: float  # Unique words / total words

    # Readability
    flesch_kincaid_grade: float
    flesch_reading_ease: float

    # Writing style indicators
    passive_voice_percentage: float
    transition_words_count: int
    complex_sentences_percentage: float


@dataclass
class ConstraintValidation:
    """Word/character constraint validation results."""

    word_count: int
    character_count: int

    # Constraint checking
    word_limit: int | None = None
    character_limit: int | None = None

    # Validation results
    within_word_limit: bool = True
    within_character_limit: bool = True
    words_over_under: int = 0  # Positive = over, negative = under
    characters_over_under: int = 0

    # Usage percentages
    word_usage_percentage: float = 0.0
    character_usage_percentage: float = 0.0

    # Recommendations
    needs_trimming: bool = False
    needs_expansion: bool = False
    optimal_range: bool = True


def get_text_metrics(content: str) -> TextMetrics:
    """
    Get comprehensive text metrics for writing analysis.

    This tool provides precise measurements that LLMs cannot reliably calculate:
    - Exact word/character counts (LLMs have ±5-15% error rate)
    - Objective readability scores (LLMs give subjective assessments)
    - Quantified style metrics (LLMs provide vague descriptions)

    Args:
        content: Text content to analyze

    Returns:
        TextMetrics object with comprehensive measurements

    Example:
        >>> metrics = get_text_metrics("This is a test sentence. Another sentence here.")
        >>> metrics.word_count
        9
        >>> metrics.sentence_count
        2
    """
    if not content or not content.strip():
        return TextMetrics(
            word_count=0,
            character_count=0,
            character_count_no_spaces=0,
            sentence_count=0,
            paragraph_count=0,
            average_words_per_sentence=0.0,
            average_sentences_per_paragraph=0.0,
            vocabulary_diversity=0.0,
            flesch_kincaid_grade=0.0,
            flesch_reading_ease=0.0,
            passive_voice_percentage=0.0,
            transition_words_count=0,
            complex_sentences_percentage=0.0,
        )

    # Basic counts
    words = _extract_words(content)
    word_count = len(words)
    character_count = len(content)
    character_count_no_spaces = len(content.replace(" ", ""))

    sentences = _extract_sentences(content)
    sentence_count = len(sentences)

    paragraphs = _extract_paragraphs(content)
    paragraph_count = len(paragraphs)

    # Averages
    avg_words_per_sentence = word_count / max(sentence_count, 1)
    avg_sentences_per_paragraph = sentence_count / max(paragraph_count, 1)

    # Vocabulary diversity (unique words / total words)
    unique_words = {word.lower() for word in words}
    vocabulary_diversity = len(unique_words) / max(word_count, 1)

    # Readability scores
    flesch_kincaid = _calculate_flesch_kincaid(content, words, sentences)
    flesch_ease = _calculate_flesch_reading_ease(content, words, sentences)

    # Style analysis
    passive_voice_pct = _calculate_passive_voice_percentage(sentences)
    transition_count = _count_transition_words(content)
    complex_sentences_pct = _calculate_complex_sentences_percentage(sentences)

    return TextMetrics(
        word_count=word_count,
        character_count=character_count,
        character_count_no_spaces=character_count_no_spaces,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        average_words_per_sentence=avg_words_per_sentence,
        average_sentences_per_paragraph=avg_sentences_per_paragraph,
        vocabulary_diversity=vocabulary_diversity,
        flesch_kincaid_grade=flesch_kincaid,
        flesch_reading_ease=flesch_ease,
        passive_voice_percentage=passive_voice_pct,
        transition_words_count=transition_count,
        complex_sentences_percentage=complex_sentences_pct,
    )


def _strip_markdown_formatting(text: str) -> str:
    """
    Strip markdown formatting from text for accurate character counting.

    Removes:
    - _italics_ and *italics*
    - **bold** and __bold__
    - ***bold italics***
    - `code`
    - [link text](url) -> link text

    Args:
        text: Text with potential markdown formatting

    Returns:
        Text with markdown formatting removed
    """
    import re

    # Remove bold and italics (order matters - do bold italics first)
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", text)  # ***bold italics***
    text = re.sub(r"___([^_]+)___", r"\1", text)  # ___bold italics___
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italics*
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italics_

    # Remove code formatting
    text = re.sub(r"`([^`]+)`", r"\1", text)  # `code`

    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text

    return text


def analyze_short_answers(content: str, character_limit: int) -> dict[str, Any]:
    """
    Analyze individual short answers against character limits.

    Parses Q&A format content and checks each answer individually
    against the specified character limit. Strips markdown formatting
    before counting characters.

    Args:
        content: Content with questions and answers
        character_limit: Character limit per individual answer

    Returns:
        Dictionary with individual answer analysis
    """
    import re

    results: dict[str, Any] = {
        "answers": [],
        "total_answers": 0,
        "answers_within_limit": 0,
        "answers_over_limit": 0,
        "character_limit": character_limit,
    }

    # Split content into lines and parse Q&A pairs
    lines = content.split("\n")
    current_question = None
    current_answer = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a question line (starts with *)
        if line.startswith("*"):
            # Save previous answer if exists
            if current_question and current_answer:
                # Strip markdown formatting before counting characters
                clean_answer = _strip_markdown_formatting(current_answer)
                char_count = len(clean_answer)
                within_limit = char_count <= character_limit

                results["answers"].append(
                    {
                        "question": current_question,
                        "answer": current_answer,  # Keep original with formatting for display
                        "clean_answer": clean_answer,  # Cleaned version for reference
                        "character_count": char_count,
                        "within_limit": within_limit,
                        "over_by": max(0, char_count - character_limit),
                    }
                )

                if within_limit:
                    results["answers_within_limit"] += 1
                else:
                    results["answers_over_limit"] += 1

            # Check if this line has both question and answer (inline format)
            if line.endswith("*"):
                # Question only format: *Question?*
                current_question = line.strip("*").strip()
                current_answer = None
            else:
                # Inline format: *Question?* Answer text here.
                parts = line.split("*", 2)  # Split on first two asterisks
                if len(parts) >= 3:
                    current_question = parts[1].strip()
                    current_answer = parts[2].strip()
                else:
                    # Fallback - treat as question only
                    current_question = line.strip("*").strip()
                    current_answer = None

        elif current_question and not current_answer:
            # This is the answer to the current question (multi-line format)
            # Remove leading dash and space if present
            if line.startswith("- "):
                line = line[2:]
            current_answer = line
        elif current_question and current_answer and line.startswith("- "):
            # Additional parts of multi-line answer (like "Second Word:", "Third Word:")
            # For now, we'll treat the first dash line as the complete answer
            # This handles the "three words" question properly
            pass

    # Don't forget the last answer
    if current_question and current_answer:
        # Strip markdown formatting before counting characters
        clean_answer = _strip_markdown_formatting(current_answer)
        char_count = len(clean_answer)
        within_limit = char_count <= character_limit

        results["answers"].append(
            {
                "question": current_question,
                "answer": current_answer,  # Keep original with formatting for display
                "clean_answer": clean_answer,  # Cleaned version for reference
                "character_count": char_count,
                "within_limit": within_limit,
                "over_by": max(0, char_count - character_limit),
            }
        )

        if within_limit:
            results["answers_within_limit"] += 1
        else:
            results["answers_over_limit"] += 1

    results["total_answers"] = len(results["answers"])

    return results


def validate_constraints(
    content: str,
    word_limit: int | None = None,
    character_limit: int | None = None,
    min_words: int | None = None,
    optimal_word_range: tuple[int, int] | None = None,
) -> ConstraintValidation:
    """
    Validate text against word/character constraints with precise accuracy.

    Critical for college applications where limits are strictly enforced:
    - Common App: 650 words maximum
    - UC Essays: 350 words maximum
    - Supplemental essays: Various limits (100-500 words)

    LLMs frequently miscalculate these constraints, leading to rejected applications.

    Args:
        content: Text to validate
        word_limit: Maximum words allowed
        character_limit: Maximum characters allowed
        min_words: Minimum words required
        optimal_word_range: Tuple of (min_optimal, max_optimal) for best range

    Returns:
        ConstraintValidation with precise constraint checking

    Example:
        >>> validation = validate_constraints("Long essay text...", word_limit=650)
        >>> validation.within_word_limit
        True
        >>> validation.words_over_under
        -23  # 23 words under limit
    """
    metrics = get_text_metrics(content)
    word_count = metrics.word_count
    character_count = metrics.character_count

    # Initialize validation result
    validation = ConstraintValidation(
        word_count=word_count,
        character_count=character_count,
        word_limit=word_limit,
        character_limit=character_limit,
    )

    # Word limit validation
    if word_limit:
        validation.within_word_limit = word_count <= word_limit
        validation.words_over_under = word_count - word_limit
        validation.word_usage_percentage = (word_count / word_limit) * 100

        if word_count > word_limit:
            validation.needs_trimming = True
            validation.optimal_range = False

    # Character limit validation
    if character_limit:
        validation.within_character_limit = character_count <= character_limit
        validation.characters_over_under = character_count - character_limit
        validation.character_usage_percentage = (
            character_count / character_limit
        ) * 100

    # Minimum word validation
    if min_words and word_count < min_words:
        validation.needs_expansion = True
        validation.optimal_range = False

    # Optimal range validation
    if optimal_word_range:
        min_optimal, max_optimal = optimal_word_range
        if not (min_optimal <= word_count <= max_optimal):
            validation.optimal_range = False
            if word_count < min_optimal:
                validation.needs_expansion = True
            elif word_count > max_optimal:
                validation.needs_trimming = True

    return validation


def analyze_readability(content: str) -> dict[str, float]:
    """
    Analyze text readability with multiple objective metrics.

    Provides quantified readability scores that LLMs cannot calculate accurately.
    Essential for ensuring appropriate complexity level for target audience.

    Args:
        content: Text to analyze

    Returns:
        Dictionary with readability metrics
    """
    words = _extract_words(content)
    sentences = _extract_sentences(content)

    if not words or not sentences:
        return {
            "flesch_kincaid_grade": 0.0,
            "flesch_reading_ease": 0.0,
            "average_sentence_length": 0.0,
            "syllables_per_word": 0.0,
        }

    return {
        "flesch_kincaid_grade": _calculate_flesch_kincaid(content, words, sentences),
        "flesch_reading_ease": _calculate_flesch_reading_ease(
            content, words, sentences
        ),
        "average_sentence_length": len(words) / len(sentences),
        "syllables_per_word": _calculate_average_syllables_per_word(words),
    }


def analyze_vocabulary(content: str) -> dict[str, Any]:
    """
    Analyze vocabulary sophistication and diversity.

    Provides objective vocabulary metrics that inform writing quality assessment.

    Args:
        content: Text to analyze

    Returns:
        Dictionary with vocabulary analysis
    """
    words = _extract_words(content)

    if not words:
        return {
            "total_words": 0,
            "unique_words": 0,
            "vocabulary_diversity": 0.0,
            "average_word_length": 0.0,
            "complex_words_percentage": 0.0,
        }

    unique_words = {word.lower() for word in words}
    word_lengths = [len(word) for word in words]
    complex_words = [word for word in words if _count_syllables(word) >= 3]

    return {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "vocabulary_diversity": len(unique_words) / len(words),
        "average_word_length": sum(word_lengths) / len(word_lengths),
        "complex_words_percentage": (len(complex_words) / len(words)) * 100,
    }


# Helper functions for text analysis


def _extract_words(content: str) -> list[str]:
    """Extract words from content, handling punctuation properly."""
    # Remove punctuation and split on whitespace
    translator = str.maketrans("", "", string.punctuation)
    clean_content = content.translate(translator)
    return [word for word in clean_content.split() if word.strip()]


def _extract_sentences(content: str) -> list[str]:
    """Extract sentences from content."""
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", content)
    return [s.strip() for s in sentences if s.strip()]


def _extract_paragraphs(content: str) -> list[str]:
    """Extract paragraphs from content."""
    paragraphs = content.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def _count_syllables(word: str) -> int:
    """Count syllables in a word using vowel-based heuristic."""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    # Handle silent 'e'
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)  # Every word has at least 1 syllable


def _calculate_average_syllables_per_word(words: list[str]) -> float:
    """Calculate average syllables per word."""
    if not words:
        return 0.0

    total_syllables = sum(_count_syllables(word) for word in words)
    return total_syllables / len(words)


def _calculate_flesch_kincaid(
    content: str, words: list[str], sentences: list[str]
) -> float:
    """Calculate Flesch-Kincaid Grade Level."""
    if not words or not sentences:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = _calculate_average_syllables_per_word(words)

    return 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59


def _calculate_flesch_reading_ease(
    content: str, words: list[str], sentences: list[str]
) -> float:
    """Calculate Flesch Reading Ease score."""
    if not words or not sentences:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = _calculate_average_syllables_per_word(words)

    return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word


def _calculate_passive_voice_percentage(sentences: list[str]) -> float:
    """Calculate percentage of sentences using passive voice."""
    if not sentences:
        return 0.0

    passive_indicators = [
        r"\b(was|were|is|are|am|be|been|being)\s+\w+ed\b",
        r"\b(was|were|is|are|am|be|been|being)\s+\w+en\b",
    ]

    passive_count = 0
    for sentence in sentences:
        for pattern in passive_indicators:
            if re.search(pattern, sentence.lower()):
                passive_count += 1
                break

    return (passive_count / len(sentences)) * 100


def _count_transition_words(content: str) -> int:
    """Count transition words and phrases."""
    transition_words = [
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "consequently",
        "nevertheless",
        "additionally",
        "meanwhile",
        "subsequently",
        "thus",
        "hence",
        "accordingly",
        "likewise",
        "similarly",
        "conversely",
        "on the other hand",
        "in contrast",
        "for example",
        "for instance",
        "in conclusion",
        "to summarize",
        "in summary",
        "finally",
    ]

    content_lower = content.lower()
    count = 0

    for transition in transition_words:
        count += content_lower.count(transition)

    return count


def _calculate_complex_sentences_percentage(sentences: list[str]) -> float:
    """Calculate percentage of complex sentences (containing subordinate clauses)."""
    if not sentences:
        return 0.0

    complex_indicators = [
        r"\b(because|since|although|though|while|whereas|if|unless|until|before|after|when|whenever)\b",
        r"\b(who|whom|whose|which|that)\b",  # Relative pronouns
        r"[,;:]",  # Punctuation indicating complexity
    ]

    complex_count = 0
    for sentence in sentences:
        for pattern in complex_indicators:
            if re.search(pattern, sentence.lower()):
                complex_count += 1
                break

    return (complex_count / len(sentences)) * 100
