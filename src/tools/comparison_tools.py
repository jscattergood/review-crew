"""
Text comparison and similarity analysis tools.

These tools provide objective comparison between multiple texts that LLMs
cannot reliably measure. Essential for detecting near-identical versions
and identifying meaningful differences.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Any

from .logging_utils import log_tool_execution


@dataclass
class SimilarityResult:
    """Results from text similarity analysis."""

    # Overall similarity metrics
    similarity_ratio: float  # 0.0 to 1.0
    similarity_percentage: float  # 0 to 100

    # Character-level metrics
    character_similarity: float  # 0.0 to 1.0
    character_differences: int  # Count of different characters

    # Word-level metrics
    word_similarity: float  # 0.0 to 1.0
    word_differences: int  # Count of different words

    # Assessment flags
    are_identical: bool  # 100% match
    are_nearly_identical: bool  # >95% similar
    are_substantially_similar: bool  # >80% similar
    have_material_differences: bool  # <80% similar

    # Difference summary
    total_changes: int  # Total number of changes
    additions: int  # Lines/words added
    deletions: int  # Lines/words deleted
    modifications: int  # Lines/words changed


@dataclass
class TextDiff:
    """Detailed differences between two texts."""

    # Line-by-line differences
    added_lines: list[str]
    removed_lines: list[str]
    modified_lines: list[tuple[str, str]]  # (old, new) pairs

    # Word-level differences
    added_words: list[str]
    removed_words: list[str]
    changed_words: list[tuple[str, str]]  # (old, new) pairs

    # Summary statistics
    total_lines_added: int
    total_lines_removed: int
    total_lines_modified: int
    total_words_changed: int


@log_tool_execution("text_similarity")
def calculate_text_similarity(
    text1: str, text2: str, ignore_whitespace: bool = True
) -> SimilarityResult:
    """
    Calculate comprehensive similarity metrics between two texts.

    This tool provides precise measurements that LLMs cannot reliably calculate:
    - Exact similarity ratios (LLMs give subjective assessments like "mostly similar")
    - Character and word-level precision
    - Objective thresholds for comparison decisions

    Critical for:
    - Detecting near-identical essay versions
    - Avoiding hallucinated differences in comparisons
    - Making data-driven revision decisions

    Args:
        text1: First text to compare
        text2: Second text to compare
        ignore_whitespace: If True, normalize whitespace before comparison

    Returns:
        SimilarityResult with comprehensive similarity metrics

    Example:
        >>> result = calculate_text_similarity(essay_v1, essay_v2)
        >>> if result.are_nearly_identical:
        ...     print(f"Essays are {result.similarity_percentage:.1f}% similar")
        >>> result.similarity_percentage
        98.7
    """
    if not text1 and not text2:
        return _create_identical_result()

    if not text1 or not text2:
        return _create_completely_different_result()

    # Normalize whitespace if requested
    if ignore_whitespace:
        text1_normalized = _normalize_whitespace(text1)
        text2_normalized = _normalize_whitespace(text2)
    else:
        text1_normalized = text1
        text2_normalized = text2

    # Calculate overall sequence similarity
    matcher = difflib.SequenceMatcher(None, text1_normalized, text2_normalized)
    similarity_ratio = matcher.ratio()
    similarity_percentage = similarity_ratio * 100

    # Calculate character-level similarity
    char_similarity = _calculate_character_similarity(
        text1_normalized, text2_normalized
    )
    char_differences = _count_character_differences(text1_normalized, text2_normalized)

    # Calculate word-level similarity
    words1 = _extract_words_for_comparison(text1_normalized)
    words2 = _extract_words_for_comparison(text2_normalized)
    word_similarity = _calculate_word_similarity(words1, words2)
    word_differences = _count_word_differences(words1, words2)

    # Get change counts
    changes = _count_changes(text1_normalized, text2_normalized)

    # Determine assessment flags
    are_identical = similarity_ratio >= 0.999
    are_nearly_identical = similarity_ratio >= 0.95
    are_substantially_similar = similarity_ratio >= 0.80
    have_material_differences = similarity_ratio < 0.80

    return SimilarityResult(
        similarity_ratio=similarity_ratio,
        similarity_percentage=similarity_percentage,
        character_similarity=char_similarity,
        character_differences=char_differences,
        word_similarity=word_similarity,
        word_differences=word_differences,
        are_identical=are_identical,
        are_nearly_identical=are_nearly_identical,
        are_substantially_similar=are_substantially_similar,
        have_material_differences=have_material_differences,
        total_changes=changes["total"],
        additions=changes["additions"],
        deletions=changes["deletions"],
        modifications=changes["modifications"],
    )


@log_tool_execution("text_diff")
def get_text_diff(text1: str, text2: str) -> TextDiff:
    """
    Get detailed line-by-line and word-by-word differences between two texts.

    Useful for understanding exactly what changed between versions.

    Args:
        text1: Original text
        text2: Modified text

    Returns:
        TextDiff with detailed difference information

    Example:
        >>> diff = get_text_diff(old_essay, new_essay)
        >>> print(f"Changed {len(diff.changed_words)} words")
        >>> for old, new in diff.changed_words[:5]:
        ...     print(f"  '{old}' → '{new}'")
    """
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Get line-level differences
    differ = difflib.Differ()
    line_diff = list(differ.compare(lines1, lines2))

    added_lines = []
    removed_lines = []
    modified_lines = []

    for line in line_diff:
        if line.startswith("+ "):
            added_lines.append(line[2:])
        elif line.startswith("- "):
            removed_lines.append(line[2:])
        elif line.startswith("? "):
            # Indicator line showing where changes occurred
            continue

    # Get word-level differences
    words1 = _extract_words_for_comparison(text1)
    words2 = _extract_words_for_comparison(text2)

    word_differ = difflib.SequenceMatcher(None, words1, words2)
    added_words = []
    removed_words = []
    changed_words = []

    for tag, i1, i2, j1, j2 in word_differ.get_opcodes():
        if tag == "insert":
            added_words.extend(words2[j1:j2])
        elif tag == "delete":
            removed_words.extend(words1[i1:i2])
        elif tag == "replace":
            # Pairs of changed words
            old_words = words1[i1:i2]
            new_words = words2[j1:j2]
            # Create pairs (if lengths differ, some words are just added/removed)
            for idx in range(max(len(old_words), len(new_words))):
                old_word = old_words[idx] if idx < len(old_words) else None
                new_word = new_words[idx] if idx < len(new_words) else None
                if old_word and new_word:
                    changed_words.append((old_word, new_word))
                elif new_word:
                    added_words.append(new_word)
                elif old_word:
                    removed_words.append(old_word)

    return TextDiff(
        added_lines=added_lines,
        removed_lines=removed_lines,
        modified_lines=modified_lines,
        added_words=added_words,
        removed_words=removed_words,
        changed_words=changed_words,
        total_lines_added=len(added_lines),
        total_lines_removed=len(removed_lines),
        total_lines_modified=len(modified_lines),
        total_words_changed=len(changed_words) + len(added_words) + len(removed_words),
    )


def extract_essay_content(compiled_text: str) -> list[str]:
    """
    Extract all essay contents from a compiled document.

    Removes metadata, assignment context, and formatting to get pure essay texts.
    Useful for comparing multiple essay versions without extraneous information.

    Args:
        compiled_text: Compiled document text with metadata

    Returns:
        List of clean essay texts (typically [primary_essay, supporting_essay])
        Returns empty list if no essays found

    Example:
        >>> essays = extract_essay_content(compiled_text)
        >>> if len(essays) == 2:
        ...     similarity = calculate_text_similarity(essays[0], essays[1])
    """
    essays = []

    # Look for both Primary and Supporting Document sections
    section_headers = ["Primary Document", "Supporting Document", "Document"]

    # Split by ## markers
    sections = compiled_text.split("##")

    for section in sections:
        section_stripped = section.strip()

        # Check if this section matches any of our headers
        for header in section_headers:
            if section_stripped.startswith(header):
                # Look for "**ESSAY TO REVIEW:**" marker
                if "**ESSAY TO REVIEW:**" in section:
                    # Extract everything after this marker until the next section
                    parts = section.split("**ESSAY TO REVIEW:**", 1)
                    if len(parts) > 1:
                        essay_content = parts[1].strip()
                        # Remove any trailing section markers or content after next ##
                        if "##" in essay_content:
                            essay_content = essay_content.split("##")[0].strip()
                        # Clean up artifacts and add to list
                        essay_content = _clean_essay_artifacts(essay_content)
                        if essay_content:  # Only add non-empty essays
                            essays.append(essay_content)
                        break
                else:
                    # Fallback: get content after metadata lines
                    lines = section.split("\n")
                    content_lines = []
                    skip_metadata = True
                    for line in lines:
                        if skip_metadata and (
                            line.startswith("•")
                            or line.startswith("**")
                            or not line.strip()
                        ):
                            continue
                        skip_metadata = False
                        content_lines.append(line)
                    essay_content = "\n".join(content_lines).strip()
                    essay_content = _clean_essay_artifacts(essay_content)
                    if essay_content:  # Only add non-empty essays
                        essays.append(essay_content)
                    break

    return essays


# Helper functions


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for consistent comparison."""
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing whitespace from lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _extract_words_for_comparison(text: str) -> list[str]:
    """Extract words for comparison, preserving order."""
    # Split on whitespace and punctuation boundaries, keeping words
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def _calculate_character_similarity(text1: str, text2: str) -> float:
    """Calculate character-level similarity ratio."""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def _count_character_differences(text1: str, text2: str) -> int:
    """Count number of different characters."""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    differences = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            differences += max(i2 - i1, j2 - j1)
    return differences


def _calculate_word_similarity(words1: list[str], words2: list[str]) -> float:
    """Calculate word-level similarity ratio."""
    matcher = difflib.SequenceMatcher(None, words1, words2)
    return matcher.ratio()


def _count_word_differences(words1: list[str], words2: list[str]) -> int:
    """Count number of different words."""
    matcher = difflib.SequenceMatcher(None, words1, words2)
    differences = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            differences += max(i2 - i1, j2 - j1)
    return differences


def _count_changes(text1: str, text2: str) -> dict[str, int]:
    """Count additions, deletions, and modifications."""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    additions = 0
    deletions = 0
    modifications = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            additions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "replace":
            modifications += max(i2 - i1, j2 - j1)

    return {
        "additions": additions,
        "deletions": deletions,
        "modifications": modifications,
        "total": additions + deletions + modifications,
    }


def _create_identical_result() -> SimilarityResult:
    """Create result for identical texts (both empty)."""
    return SimilarityResult(
        similarity_ratio=1.0,
        similarity_percentage=100.0,
        character_similarity=1.0,
        character_differences=0,
        word_similarity=1.0,
        word_differences=0,
        are_identical=True,
        are_nearly_identical=True,
        are_substantially_similar=True,
        have_material_differences=False,
        total_changes=0,
        additions=0,
        deletions=0,
        modifications=0,
    )


def _create_completely_different_result() -> SimilarityResult:
    """Create result for completely different texts (one empty)."""
    return SimilarityResult(
        similarity_ratio=0.0,
        similarity_percentage=0.0,
        character_similarity=0.0,
        character_differences=0,
        word_similarity=0.0,
        word_differences=0,
        are_identical=False,
        are_nearly_identical=False,
        are_substantially_similar=False,
        have_material_differences=True,
        total_changes=0,
        additions=0,
        deletions=0,
        modifications=0,
    )


def _clean_essay_artifacts(text: str) -> str:
    """Clean up markdown and formatting artifacts from essay text."""
    # Remove empty lines at start/end
    text = text.strip()

    # Remove repeated newlines (more than 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text
