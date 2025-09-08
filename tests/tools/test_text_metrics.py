"""Unit tests for text metrics tools."""

import pytest

from src.tools.text_metrics import (
    ConstraintValidation,
    TextMetrics,
    analyze_readability,
    analyze_vocabulary,
    analyze_short_answers,
    get_text_metrics,
    validate_constraints,
    _strip_markdown_formatting,
)


class TestGetTextMetrics:
    """Test the get_text_metrics function."""

    def test_empty_content(self):
        """Test with empty content."""
        metrics = get_text_metrics("")
        assert metrics.word_count == 0
        assert metrics.character_count == 0
        assert metrics.sentence_count == 0
        assert metrics.paragraph_count == 0

    def test_simple_content(self):
        """Test with simple content."""
        content = "This is a test sentence. This is another sentence."
        metrics = get_text_metrics(content)

        assert metrics.word_count == 9  # Actual count from word extraction
        assert metrics.character_count == len(content)
        assert metrics.sentence_count == 2
        assert metrics.paragraph_count == 1
        assert metrics.average_words_per_sentence == 4.5

    def test_multi_paragraph_content(self):
        """Test with multiple paragraphs."""
        content = """This is the first paragraph. It has two sentences.

This is the second paragraph. It also has content."""

        metrics = get_text_metrics(content)
        assert metrics.paragraph_count == 2
        assert metrics.sentence_count == 4
        assert metrics.word_count == 18  # Actual count from word extraction

    def test_vocabulary_diversity(self):
        """Test vocabulary diversity calculation."""
        # Content with repeated words
        content = "test test test different word"
        metrics = get_text_metrics(content)

        # 3 unique words out of 5 total = 0.6 diversity
        assert metrics.vocabulary_diversity == 0.6

    def test_readability_scores(self):
        """Test readability score calculation."""
        content = "This is a simple sentence. This is another simple sentence."
        metrics = get_text_metrics(content)

        assert isinstance(metrics.flesch_kincaid_grade, float)
        assert isinstance(metrics.flesch_reading_ease, float)
        assert metrics.flesch_kincaid_grade > 0
        assert metrics.flesch_reading_ease > 0


class TestValidateConstraints:
    """Test the validate_constraints function."""

    def test_within_word_limit(self):
        """Test content within word limit."""
        content = "This is a short essay with exactly ten words total."
        constraints = validate_constraints(content, word_limit=15)

        assert constraints.word_count == 10
        assert constraints.within_word_limit is True
        assert constraints.words_over_under == -5
        assert constraints.needs_trimming is False

    def test_over_word_limit(self):
        """Test content over word limit."""
        content = "This is a longer essay that exceeds the word limit significantly."
        constraints = validate_constraints(content, word_limit=5)

        assert constraints.word_count == 11
        assert constraints.within_word_limit is False
        assert constraints.words_over_under == 6
        assert constraints.needs_trimming is True

    def test_optimal_range(self):
        """Test optimal range validation."""
        content = "This essay has exactly eight words in it."
        constraints = validate_constraints(
            content, word_limit=15, optimal_word_range=(7, 10)
        )

        assert constraints.word_count == 8
        assert constraints.optimal_range is True
        assert constraints.needs_expansion is False
        assert constraints.needs_trimming is False

    def test_below_optimal_range(self):
        """Test content below optimal range."""
        content = "Short essay."
        constraints = validate_constraints(
            content, word_limit=20, optimal_word_range=(10, 15)
        )

        assert constraints.word_count == 2
        assert constraints.optimal_range is False
        assert constraints.needs_expansion is True

    def test_character_limit(self):
        """Test character limit validation."""
        content = "Test content"
        constraints = validate_constraints(content, character_limit=10)

        assert constraints.character_count == len(content)
        assert constraints.within_character_limit is False
        assert constraints.characters_over_under > 0

    def test_usage_percentages(self):
        """Test usage percentage calculations."""
        content = "This has five words exactly."
        constraints = validate_constraints(content, word_limit=10)

        assert constraints.word_usage_percentage == 50.0


class TestAnalyzeReadability:
    """Test the analyze_readability function."""

    def test_basic_readability(self):
        """Test basic readability analysis."""
        content = "This is a simple sentence for testing readability scores."
        result = analyze_readability(content)

        assert "flesch_kincaid_grade" in result
        assert "flesch_reading_ease" in result
        assert "average_sentence_length" in result
        assert "syllables_per_word" in result

        assert isinstance(result["flesch_kincaid_grade"], float)
        assert result["average_sentence_length"] > 0

    def test_empty_content_readability(self):
        """Test readability with empty content."""
        result = analyze_readability("")

        assert result["flesch_kincaid_grade"] == 0.0
        assert result["flesch_reading_ease"] == 0.0
        assert result["average_sentence_length"] == 0.0


class TestAnalyzeVocabulary:
    """Test the analyze_vocabulary function."""

    def test_vocabulary_analysis(self):
        """Test vocabulary analysis."""
        content = "This is a comprehensive vocabulary analysis test with various words."
        result = analyze_vocabulary(content)

        assert "total_words" in result
        assert "unique_words" in result
        assert "vocabulary_diversity" in result
        assert "average_word_length" in result
        assert "complex_words_percentage" in result

        assert result["total_words"] == 10  # Actual count from word extraction
        assert result["unique_words"] == 10  # All words are unique
        assert result["vocabulary_diversity"] == 1.0

    def test_empty_vocabulary(self):
        """Test vocabulary analysis with empty content."""
        result = analyze_vocabulary("")

        assert result["total_words"] == 0
        assert result["unique_words"] == 0
        assert result["vocabulary_diversity"] == 0.0

    def test_repeated_words_vocabulary(self):
        """Test vocabulary with repeated words."""
        content = "test test test different word"
        result = analyze_vocabulary(content)

        assert result["total_words"] == 5
        assert result["unique_words"] == 3
        assert result["vocabulary_diversity"] == 0.6


class TestDataClasses:
    """Test the data classes."""

    def test_text_metrics_dataclass(self):
        """Test TextMetrics dataclass."""
        metrics = TextMetrics(
            word_count=100,
            character_count=500,
            character_count_no_spaces=400,
            sentence_count=5,
            paragraph_count=2,
            average_words_per_sentence=20.0,
            average_sentences_per_paragraph=2.5,
            vocabulary_diversity=0.8,
            flesch_kincaid_grade=10.0,
            flesch_reading_ease=60.0,
            passive_voice_percentage=10.0,
            transition_words_count=3,
            complex_sentences_percentage=25.0,
        )

        assert metrics.word_count == 100
        assert metrics.vocabulary_diversity == 0.8

    def test_constraint_validation_dataclass(self):
        """Test ConstraintValidation dataclass."""
        validation = ConstraintValidation(
            word_count=650,
            character_count=3000,
            word_limit=650,
            within_word_limit=True,
            words_over_under=0,
            word_usage_percentage=100.0,
        )

        assert validation.word_count == 650
        assert validation.within_word_limit is True


class TestStripMarkdownFormatting:
    """Test markdown formatting removal."""

    def test_strip_italics(self):
        """Test stripping italic formatting."""
        # Test underscore italics
        text = "_The Art of Tahdig_: Perfect Crispy Rice"
        result = _strip_markdown_formatting(text)
        assert result == "The Art of Tahdig: Perfect Crispy Rice"
        
        # Test asterisk italics
        text = "*italic text* and normal text"
        result = _strip_markdown_formatting(text)
        assert result == "italic text and normal text"

    def test_strip_bold(self):
        """Test stripping bold formatting."""
        # Test double asterisk bold
        text = "**Bold text** and normal text"
        result = _strip_markdown_formatting(text)
        assert result == "Bold text and normal text"
        
        # Test double underscore bold
        text = "__Bold text__ and normal text"
        result = _strip_markdown_formatting(text)
        assert result == "Bold text and normal text"

    def test_strip_bold_italics(self):
        """Test stripping bold italic formatting."""
        text = "***Bold italic text*** and normal text"
        result = _strip_markdown_formatting(text)
        assert result == "Bold italic text and normal text"
        
        text = "___Bold italic text___ and normal text"
        result = _strip_markdown_formatting(text)
        assert result == "Bold italic text and normal text"

    def test_strip_code(self):
        """Test stripping code formatting."""
        text = "Use `code` formatting here"
        result = _strip_markdown_formatting(text)
        assert result == "Use code formatting here"

    def test_strip_links(self):
        """Test stripping link formatting."""
        text = "[Link text](https://example.com) should become just text"
        result = _strip_markdown_formatting(text)
        assert result == "Link text should become just text"

    def test_mixed_formatting(self):
        """Test stripping multiple types of formatting."""
        text = "**Bold** and _italic_ and `code` and [link](url)"
        result = _strip_markdown_formatting(text)
        assert result == "Bold and italic and code and link"

    def test_no_formatting(self):
        """Test text without formatting remains unchanged."""
        text = "Plain text without any formatting"
        result = _strip_markdown_formatting(text)
        assert result == text

    def test_real_world_example(self):
        """Test with the actual USC example."""
        text = "_The Art of Tahdig_: Perfect Crispy Rice—where burning the bottom isn't failure; it's the whole point."
        result = _strip_markdown_formatting(text)
        expected = "The Art of Tahdig: Perfect Crispy Rice—where burning the bottom isn't failure; it's the whole point."
        assert result == expected
        assert len(result) == 100  # Should be exactly 100 characters


class TestAnalyzeShortAnswers:
    """Test short answer analysis functionality."""

    def test_single_inline_answer(self):
        """Test parsing single inline format answer."""
        content = "*What is your favorite color?* Blue is my favorite color."
        result = analyze_short_answers(content, 100)
        
        assert result['total_answers'] == 1
        assert result['answers_within_limit'] == 1
        assert result['answers_over_limit'] == 0
        
        answer = result['answers'][0]
        assert answer['question'] == "What is your favorite color?"
        assert answer['answer'] == "Blue is my favorite color."
        assert answer['character_count'] == 26
        assert answer['within_limit'] is True

    def test_single_multiline_answer(self):
        """Test parsing single multi-line format answer."""
        content = """*Describe yourself in three words.*
- First Word: Compassionate"""
        result = analyze_short_answers(content, 100)
        
        assert result['total_answers'] == 1
        answer = result['answers'][0]
        assert answer['question'] == "Describe yourself in three words."
        assert answer['answer'] == "First Word: Compassionate"
        assert answer['character_count'] == 25

    def test_multiple_mixed_format_answers(self):
        """Test parsing multiple answers in different formats."""
        content = """*What is your favorite snack?* Trader Joe's chips.

*Dream job:* Head of my own NGO.

*Describe yourself.*
- Creative person"""
        
        result = analyze_short_answers(content, 50)
        
        assert result['total_answers'] == 3
        
        # Check first answer (inline)
        assert result['answers'][0]['question'] == "What is your favorite snack?"
        assert result['answers'][0]['answer'] == "Trader Joe's chips."
        
        # Check second answer (inline)
        assert result['answers'][1]['question'] == "Dream job:"
        assert result['answers'][1]['answer'] == "Head of my own NGO."
        
        # Check third answer (multiline)
        assert result['answers'][2]['question'] == "Describe yourself."
        assert result['answers'][2]['answer'] == "Creative person"

    def test_character_limit_validation(self):
        """Test character limit validation."""
        content = """*Short answer?* Yes.
*Long answer?* This is a very long answer that definitely exceeds the character limit we set for testing purposes."""
        
        result = analyze_short_answers(content, 20)
        
        assert result['total_answers'] == 2
        assert result['answers_within_limit'] == 1
        assert result['answers_over_limit'] == 1
        
        # Short answer should be within limit
        assert result['answers'][0]['within_limit'] is True
        assert result['answers'][0]['character_count'] <= 20
        
        # Long answer should be over limit
        assert result['answers'][1]['within_limit'] is False
        assert result['answers'][1]['character_count'] > 20
        assert result['answers'][1]['over_by'] > 0

    def test_markdown_stripping_in_answers(self):
        """Test that markdown formatting is stripped from character counts."""
        content = "*Class topic?* _The Art of Tahdig_: Perfect Crispy Rice—where burning the bottom isn't failure; it's the whole point."
        
        result = analyze_short_answers(content, 100)
        
        assert result['total_answers'] == 1
        answer = result['answers'][0]
        
        # Original answer should contain markdown
        assert "_The Art of Tahdig_" in answer['answer']
        
        # Clean answer should not contain markdown
        assert "_The Art of Tahdig_" not in answer['clean_answer']
        assert "The Art of Tahdig:" in answer['clean_answer']
        
        # Character count should be based on clean version
        assert answer['character_count'] == 100  # Without markdown formatting
        assert answer['within_limit'] is True

    def test_empty_content(self):
        """Test with empty content."""
        result = analyze_short_answers("", 100)
        
        assert result['total_answers'] == 0
        assert result['answers_within_limit'] == 0
        assert result['answers_over_limit'] == 0
        assert result['answers'] == []

    def test_no_answers_found(self):
        """Test with content that has no recognizable Q&A format."""
        content = "This is just plain text without any question format."
        result = analyze_short_answers(content, 100)
        
        assert result['total_answers'] == 0
        assert result['answers'] == []
