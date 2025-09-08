"""Unit tests for context parser."""

import pytest

from src.tools.context_parser import (
    ContentConstraints,
    extract_constraints_from_content,
    extract_essay_content,
    format_constraints_for_prompt,
    get_analysis_types_for_content,
)


class TestExtractConstraintsFromContent:
    """Test constraint extraction from content."""

    def test_common_app_format(self):
        """Test extraction from Common App format."""
        content = """
**ASSIGNMENT CONTEXT:**
- Essay Type: Common Application Personal Statement
- Word Limit: 650 words
- Constraint: Must work for ALL schools - cannot mention specific institutions

**ESSAY TO REVIEW:**
Essay content here...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.word_limit == 650
        assert constraints.essay_type == "Common Application Personal Statement"
        assert len(constraints.special_requirements) > 0
        assert (
            "Must work for ALL schools - cannot mention specific institutions"
            in constraints.special_requirements
        )

    def test_harvard_supplemental_format(self):
        """Test extraction from Harvard supplemental format."""
        content = """
**ASSIGNMENT CONTEXT:**
Essay Type: Harvard Supplemental Essay
Word Limit: 150 words
Prompt: "What would you want your future college roommate to know about you?"

**ESSAY TO REVIEW:**
Essay content...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.word_limit == 150
        assert constraints.essay_type == "Harvard Supplemental Essay"
        assert constraints.school_name == "Harvard"
        assert constraints.prompt is not None

    def test_stanford_format(self):
        """Test extraction from Stanford format."""
        content = """
Essay Type: Stanford Supplemental Essay
Word Limit: 250 words
Prompt: "Tell us about something meaningful to you."

Essay content follows...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.word_limit == 250
        assert constraints.essay_type == "Stanford Supplemental Essay"
        assert constraints.school_name == "Stanford"

    def test_character_limit_extraction(self):
        """Test character limit extraction."""
        content = """
Character Limit: 4000 characters
Essay Type: Short Response
Content here...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.character_limit == 4000
        assert constraints.essay_type == "Short Response"

    def test_no_constraints(self):
        """Test with content that has no constraints."""
        content = "Just plain essay content without any context."

        constraints = extract_constraints_from_content(content)

        assert constraints.word_limit is None
        assert constraints.essay_type is None
        assert constraints.school_name is None
        assert len(constraints.special_requirements) == 0

    def test_multiple_word_limits(self):
        """Test that first word limit is extracted when multiple exist."""
        content = """
Word Limit: 650 words
Maximum: 500 words
Limit: 750 words
Essay content...
        """

        constraints = extract_constraints_from_content(content)

        # Should extract the first one found
        assert constraints.word_limit == 650

    def test_case_insensitive_extraction(self):
        """Test case-insensitive pattern matching."""
        content = """
word limit: 650 words
essay type: Common App Personal Statement
Essay content...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.word_limit == 650
        assert constraints.essay_type == "Common App Personal Statement"

    def test_special_requirements_detection(self):
        """Test detection of special requirements."""
        content = """
Word Limit: 650 words
Constraint: Cannot mention specific institutions
Note: Must work for all schools
Requirement: Generic content only
Essay content...
        """

        constraints = extract_constraints_from_content(content)

        assert len(constraints.special_requirements) >= 3
        assert any(
            "Cannot mention specific institutions" in req
            for req in constraints.special_requirements
        )

    def test_optimal_range_calculation(self):
        """Test optimal range calculation."""
        content = """
Essay Type: Common Application Personal Statement
Word Limit: 650 words
Essay content...
        """

        constraints = extract_constraints_from_content(content)

        assert constraints.optimal_word_range is not None
        min_words, max_words = constraints.optimal_word_range
        assert min_words < max_words
        assert max_words == 650  # Should use full limit for Common App


class TestGetAnalysisTypesForContent:
    """Test analysis type selection based on content."""

    def test_common_app_analysis_types(self):
        """Test analysis types for Common App essays."""
        constraints = ContentConstraints(
            essay_type="Common Application Personal Statement", word_limit=650
        )

        analysis_types = get_analysis_types_for_content(constraints)

        assert "metrics" in analysis_types
        assert "constraints" in analysis_types
        assert "structure" in analysis_types
        assert "strength" in analysis_types
        assert "cliches" in analysis_types
        assert "voice" in analysis_types

    def test_supplemental_analysis_types(self):
        """Test analysis types for supplemental essays."""
        constraints = ContentConstraints(
            essay_type="Harvard Supplemental Essay", word_limit=150
        )

        analysis_types = get_analysis_types_for_content(constraints)

        assert "metrics" in analysis_types
        assert "constraints" in analysis_types
        assert "readability" in analysis_types
        assert "strength" in analysis_types

    def test_technical_analysis_types(self):
        """Test analysis types for technical content."""
        constraints = ContentConstraints(essay_type="Technical Documentation")

        analysis_types = get_analysis_types_for_content(constraints)

        assert "metrics" in analysis_types
        assert "readability" in analysis_types
        assert "vocabulary" in analysis_types
        assert "structure" in analysis_types

    def test_unknown_type_analysis(self):
        """Test analysis types for unknown content type."""
        constraints = ContentConstraints()

        analysis_types = get_analysis_types_for_content(constraints)

        # Should return basic analysis types
        assert "metrics" in analysis_types
        assert "readability" in analysis_types
        assert "structure" in analysis_types


class TestFormatConstraintsForPrompt:
    """Test constraint formatting for prompts."""

    def test_format_complete_constraints(self):
        """Test formatting with complete constraint information."""
        constraints = ContentConstraints(
            word_limit=650,
            essay_type="Common Application Personal Statement",
            school_name="Generic",
            special_requirements=["Must work for all schools", "No specific mentions"],
            optimal_word_range=(600, 650),
        )

        formatted = format_constraints_for_prompt(constraints)

        assert "Word limit: 650 words" in formatted
        assert "Optimal range: 600-650 words" in formatted
        assert "Essay type: Common Application Personal Statement" in formatted
        assert "Special requirements:" in formatted
        assert "Must work for all schools" in formatted

    def test_format_minimal_constraints(self):
        """Test formatting with minimal constraint information."""
        constraints = ContentConstraints(word_limit=150)

        formatted = format_constraints_for_prompt(constraints)

        assert "Word limit: 150 words" in formatted
        assert formatted.strip()  # Should not be empty

    def test_format_empty_constraints(self):
        """Test formatting with no constraints."""
        constraints = ContentConstraints()

        formatted = format_constraints_for_prompt(constraints)

        assert formatted == ""

    def test_format_character_limit(self):
        """Test formatting with character limit."""
        constraints = ContentConstraints(
            character_limit=4000, essay_type="Short Response"
        )

        formatted = format_constraints_for_prompt(constraints)

        assert "Character limit: 4000" in formatted
        assert "Essay type: Short Response" in formatted


class TestContentConstraintsDataClass:
    """Test the ContentConstraints dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        constraints = ContentConstraints()

        assert constraints.word_limit is None
        assert constraints.character_limit is None
        assert constraints.essay_type is None
        assert constraints.school_name is None
        assert constraints.special_requirements == []
        assert constraints.optimal_word_range is None

    def test_post_init_special_requirements(self):
        """Test __post_init__ sets empty list for special_requirements."""
        constraints = ContentConstraints(word_limit=650, special_requirements=None)

        assert constraints.special_requirements == []

    def test_full_initialization(self):
        """Test initialization with all fields."""
        requirements = ["Must be generic", "No school names"]
        constraints = ContentConstraints(
            word_limit=650,
            character_limit=3000,
            essay_type="Common App",
            school_name="Generic",
            special_requirements=requirements,
            optimal_word_range=(600, 650),
            prompt="Test prompt",
            target_audience="Admissions officers",
        )

        assert constraints.word_limit == 650
        assert constraints.character_limit == 3000
        assert constraints.essay_type == "Common App"
        assert constraints.school_name == "Generic"
        assert constraints.special_requirements == requirements
        assert constraints.optimal_word_range == (600, 650)
        assert constraints.prompt == "Test prompt"
        assert constraints.target_audience == "Admissions officers"


class TestExtractEssayContent:
    """Test essay content extraction from full documents."""

    def test_extract_with_essay_to_review_marker(self):
        """Test extraction with 'ESSAY TO REVIEW:' marker."""
        content = """
**ASSIGNMENT CONTEXT:**
- Essay Type: Common Application Personal Statement
- Word Limit: 650 words

**ESSAY TO REVIEW:**

"Everyone, stay in your seats. The school is on lockdown."

The announcement wasn't over the loudspeaker; it whispered from Mrs. Scott.
"""
        
        essay_content = extract_essay_content(content)
        
        assert essay_content.startswith('"Everyone, stay in your seats.')
        assert "ASSIGNMENT CONTEXT" not in essay_content
        assert "Word Limit" not in essay_content
        assert "ESSAY TO REVIEW" not in essay_content

    def test_extract_with_essay_marker(self):
        """Test extraction with 'Essay:' marker."""
        content = """
Assignment: Write a personal statement
Word Limit: 500 words

Essay:

This is my personal statement about overcoming challenges.
"""
        
        essay_content = extract_essay_content(content)
        
        assert essay_content == "This is my personal statement about overcoming challenges."
        assert "Assignment:" not in essay_content
        assert "Word Limit:" not in essay_content

    def test_extract_fallback_method(self):
        """Test fallback extraction when no explicit markers."""
        content = """
# Assignment Details
- Type: Personal Statement  
- Limit: 650 words
- Prompt: Describe a challenge

My essay begins here with an interesting story about perseverance.
I learned many valuable lessons through this experience.
"""
        
        essay_content = extract_essay_content(content)
        
        assert essay_content.startswith("My essay begins here")
        assert "Assignment Details" not in essay_content
        assert "Type: Personal Statement" not in essay_content

    def test_extract_no_separation_needed(self):
        """Test when content is already just essay text."""
        content = "This is a simple essay without any assignment context."
        
        essay_content = extract_essay_content(content)
        
        assert essay_content == content

    def test_extract_empty_content(self):
        """Test extraction with empty or None content."""
        assert extract_essay_content("") == ""
        assert extract_essay_content("   ") == "   "  # Function returns content.strip() which preserves whitespace
        assert extract_essay_content(None) == None

    def test_extract_common_app_format(self):
        """Test extraction with real Common App format."""
        content = """# Common Application Essay

**ASSIGNMENT CONTEXT:**
- Essay Type: Common Application Personal Statement
- Word Limit: 650 words
- Constraint: Must work for ALL schools - cannot mention specific institutions

**ESSAY TO REVIEW:**

"Everyone, stay in your seats. The school is on lockdown."

The announcement wasn't over the loudspeaker; it whispered from Mrs. Scott. My biology class froze in disbelief."""
        
        essay_content = extract_essay_content(content)
        
        # Should extract just the essay content
        assert essay_content.startswith('"Everyone, stay in your seats.')
        assert "Common Application Essay" not in essay_content
        assert "ASSIGNMENT CONTEXT" not in essay_content
        assert "Word Limit: 650" not in essay_content
        
        # Should include the essay text
        assert "Mrs. Scott" in essay_content
        assert "biology class" in essay_content
