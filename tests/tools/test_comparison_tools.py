"""Tests for comparison tools."""

import pytest

from src.tools.comparison_tools import (
    calculate_text_similarity,
    extract_essay_content,
    get_text_diff,
)


class TestCalculateTextSimilarity:
    """Tests for text similarity calculation."""

    def test_identical_texts(self) -> None:
        """Test that identical texts return 100% similarity."""
        text = "This is a test essay about college admissions."
        result = calculate_text_similarity(text, text)

        assert result.are_identical
        assert result.similarity_percentage == 100.0
        assert result.word_differences == 0
        assert result.character_differences == 0

    def test_nearly_identical_texts(self) -> None:
        """Test that nearly identical texts (>95%) are detected."""
        text1 = "This is a test essay about college admissions. I have learned many things and I didn't expect to grow so much."
        text2 = "This is a test essay about college admissions. I have learned many things and I did not expect to grow so much."

        result = calculate_text_similarity(text1, text2)

        # Should be >95% similar (only "didn't" vs "did not" difference)
        assert result.are_nearly_identical
        assert result.similarity_percentage > 95.0
        assert result.word_differences <= 2  # "didn't" -> "did not" is 1-2 words

    def test_substantially_different_texts(self) -> None:
        """Test that substantially different texts are correctly identified."""
        text1 = "This is a test essay about college admissions."
        text2 = "I love playing basketball and practicing every day after school."

        result = calculate_text_similarity(text1, text2)

        assert result.have_material_differences
        assert not result.are_nearly_identical
        assert result.similarity_percentage < 50.0

    def test_empty_texts(self) -> None:
        """Test that empty texts are handled correctly."""
        result = calculate_text_similarity("", "")

        assert result.are_identical
        assert result.similarity_percentage == 100.0

    def test_one_empty_text(self) -> None:
        """Test that one empty text results in 0% similarity."""
        result = calculate_text_similarity("Some text", "")

        assert result.have_material_differences
        assert result.similarity_percentage == 0.0

    def test_whitespace_normalization(self) -> None:
        """Test that whitespace normalization works correctly."""
        text1 = "This is   a test   essay"
        text2 = "This is a test essay"

        result = calculate_text_similarity(text1, text2, ignore_whitespace=True)

        # Should be identical after whitespace normalization
        assert result.are_nearly_identical
        assert result.similarity_percentage > 95.0

    def test_case_sensitivity(self) -> None:
        """Test that comparison is case-sensitive by default."""
        text1 = "This is a Test Essay"
        text2 = "this is a test essay"

        result = calculate_text_similarity(text1, text2)

        # Should still be very similar but not identical
        assert result.are_substantially_similar
        assert result.similarity_percentage > 80.0

    def test_real_essay_comparison(self) -> None:
        """Test comparison with realistic essay variations."""
        essay1 = """
        When I was in ninth grade, a student smuggled a knife onto campus and stabbed 
        a girl nearly to death. A girl—my friend—who I'd just had lunch with. Her name 
        was Ava, like mine, and I didn't know if I was ever going to see her again.
        """

        essay2 = """
        When I was in ninth grade, a student smuggled a knife onto campus and stabbed 
        a girl nearly to death. A girl—my friend—who I'd just had lunch with. Her name 
        was Ava, like mine, and I did not know if I was ever going to see her again.
        """

        result = calculate_text_similarity(essay1, essay2)

        # Should be nearly identical (only "didn't" vs "did not" difference)
        assert result.are_nearly_identical
        assert result.similarity_percentage > 98.0
        assert result.word_differences <= 2  # "didn't" -> "did not" is 1-2 words


class TestGetTextDiff:
    """Tests for text diff functionality."""

    def test_simple_word_changes(self) -> None:
        """Test detection of simple word changes."""
        text1 = "I love college admissions"
        text2 = "I love university admissions"

        diff = get_text_diff(text1, text2)

        # Should detect word change
        assert len(diff.changed_words) > 0 or len(diff.removed_words) > 0

    def test_additions(self) -> None:
        """Test detection of added words."""
        text1 = "I love college"
        text2 = "I love college admissions very much"

        diff = get_text_diff(text1, text2)

        # Should detect additions
        assert len(diff.added_words) > 0
        assert diff.total_words_changed > 0

    def test_deletions(self) -> None:
        """Test detection of removed words."""
        text1 = "I love college admissions very much"
        text2 = "I love college"

        diff = get_text_diff(text1, text2)

        # Should detect deletions
        assert len(diff.removed_words) > 0
        assert diff.total_words_changed > 0

    def test_line_differences(self) -> None:
        """Test line-level difference detection."""
        text1 = "Line one\nLine two\nLine three"
        text2 = "Line one\nLine two modified\nLine three"

        diff = get_text_diff(text1, text2)

        # Should detect line changes
        assert diff.total_lines_added + diff.total_lines_removed > 0


class TestExtractEssayContent:
    """Tests for essay content extraction."""

    def test_extract_multiple_documents(self) -> None:
        """Test extraction of both primary and supporting documents."""
        compiled_text = """
## Primary Document

• **File:** essay.md

**ASSIGNMENT CONTEXT:**
- Essay Type: Common App
- Word Limit: 650 words

**ESSAY TO REVIEW:**

This is the primary essay content that should be extracted.
It has multiple paragraphs.

## Supporting Document

• **File:** essay2.md

**ESSAY TO REVIEW:**

This is the supporting essay content.
Multiple paragraphs here too.
"""

        essays = extract_essay_content(compiled_text)

        assert len(essays) == 2
        assert "primary essay content" in essays[0]
        assert "supporting essay content" in essays[1]
        assert "ASSIGNMENT CONTEXT" not in essays[0]
        assert "ASSIGNMENT CONTEXT" not in essays[1]

    def test_extract_single_document(self) -> None:
        """Test extraction when only one document is present."""
        compiled_text = """
## Primary Document

• **File:** essay.md

**ESSAY TO REVIEW:**

This is the only essay content.
"""

        essays = extract_essay_content(compiled_text)

        assert len(essays) == 1
        assert "only essay content" in essays[0]

    def test_extract_with_no_marker(self) -> None:
        """Test extraction when essay marker is not present."""
        compiled_text = """
## Primary Document

• **File:** essay.md

Just essay content without explicit markers.
"""

        essays = extract_essay_content(compiled_text)

        # Should return list with fallback extraction
        assert len(essays) >= 1
        assert len(essays[0]) > 0

    def test_extract_empty_content(self) -> None:
        """Test extraction with no documents."""
        compiled_text = """
Some random text without document markers.
"""

        essays = extract_essay_content(compiled_text)

        # Should return empty list when no documents found
        assert essays == []


class TestSimilarityMetrics:
    """Tests for similarity metrics accuracy."""

    def test_word_level_similarity(self) -> None:
        """Test that word-level similarity is calculated correctly."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown cat jumps over the lazy dog"

        result = calculate_text_similarity(text1, text2)

        # Should be very similar (only 1 word different out of 9)
        assert result.word_similarity > 0.85
        assert result.word_differences == 1

    def test_character_level_similarity(self) -> None:
        """Test that character-level similarity is calculated correctly."""
        text1 = "Hello world"
        text2 = "Hello world!"

        result = calculate_text_similarity(text1, text2)

        # Should be nearly identical (only 1 character added)
        assert result.character_similarity > 0.95
        assert result.character_differences == 1

    def test_change_counts(self) -> None:
        """Test that addition/deletion/modification counts are accurate."""
        text1 = "Original text here"
        text2 = "Modified text here now"

        result = calculate_text_similarity(text1, text2)

        # Should detect some changes
        assert result.total_changes > 0
        assert (
            result.additions + result.deletions + result.modifications
            == result.total_changes
        )
