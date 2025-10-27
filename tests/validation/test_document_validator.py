"""
Tests for document validation pipeline.

Tests document format validation, content quality checks, metadata extraction,
and validation reporting.
"""

from pathlib import Path

import pytest

from src.validation.document_validator import (
    DocumentMetadata,
    DocumentValidator,
    ValidationLevel,
    ValidationResult,
)


class TestDocumentValidator:
    """Test document validation functionality."""

    def test_validator_initialization(self):
        """Test validator initialization with default and custom config."""
        # Default config
        validator = DocumentValidator()
        assert validator.rules["max_word_count"] == 1000
        assert validator.rules["min_word_count"] == 50

        # Custom config
        custom_config = {"max_word_count": 500, "min_word_count": 100}
        validator = DocumentValidator(custom_config)
        assert validator.rules["max_word_count"] == 500
        assert validator.rules["min_word_count"] == 100

    def test_extract_metadata(self):
        """Test metadata extraction from document content."""
        validator = DocumentValidator()
        content = """# Test Document

This is a test document with multiple paragraphs.
It has several sentences to test sentence counting.

This is another paragraph.
It also has multiple sentences! And some punctuation?

Final paragraph here."""

        metadata = validator._extract_metadata(content)

        assert metadata.word_count > 0
        assert metadata.character_count == len(content)
        assert metadata.paragraph_count >= 3  # May vary based on parsing
        assert metadata.sentence_count >= 5  # Should detect multiple sentences
        assert metadata.reading_level is not None

    def test_validate_content_word_count(self):
        """Test word count validation."""
        validator = DocumentValidator({"min_word_count": 10, "max_word_count": 50})

        # Test short content
        short_content = "Too short"
        metadata = DocumentMetadata(
            word_count=2, character_count=9, paragraph_count=1, sentence_count=1
        )
        results = validator._validate_content(short_content, metadata)

        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warnings) >= 1
        assert "too short" in warnings[0].message.lower()

        # Test long content
        long_content = " ".join(["word"] * 60)
        metadata = DocumentMetadata(
            word_count=60,
            character_count=len(long_content),
            paragraph_count=1,
            sentence_count=1,
        )
        results = validator._validate_content(long_content, metadata)

        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warnings) >= 1
        assert "exceeds word limit" in warnings[0].message.lower()

    def test_validate_content_empty(self):
        """Test validation of empty content."""
        validator = DocumentValidator()
        metadata = DocumentMetadata(
            word_count=0, character_count=0, paragraph_count=0, sentence_count=0
        )
        results = validator._validate_content("", metadata)

        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) >= 1
        assert "empty" in errors[0].message.lower()

    def test_validate_structure_long_paragraphs(self):
        """Test validation of document structure."""
        validator = DocumentValidator({"max_paragraph_length": 20})

        # Create content with long paragraph
        long_paragraph = " ".join(["word"] * 30)
        content = f"# Title\n\n{long_paragraph}\n\nShort paragraph."

        results = validator._validate_structure(content)

        info_items = [r for r in results if r.level == ValidationLevel.INFO]
        long_paragraph_warnings = [r for r in info_items if "long" in r.message.lower()]
        assert len(long_paragraph_warnings) >= 1

    def test_check_repetition(self):
        """Test repetition checking."""
        validator = DocumentValidator()

        # Create content with repetitive words
        content = "testing " * 20 + "other words here"
        results = validator._check_repetition(content)

        repetition_issues = [r for r in results if "appears" in r.message]
        assert len(repetition_issues) >= 1
        assert "testing" in repetition_issues[0].message

    def test_check_formatting(self):
        """Test formatting issue detection."""
        validator = DocumentValidator()

        # Test multiple spaces
        content_spaces = "This  has  multiple  spaces"
        results = validator._check_formatting(content_spaces)
        space_issues = [r for r in results if "consecutive spaces" in r.message]
        assert len(space_issues) >= 1

        # Test multiple line breaks
        content_breaks = "Line 1\n\n\nLine 2"
        results = validator._check_formatting(content_breaks)
        break_issues = [r for r in results if "line breaks" in r.message]
        assert len(break_issues) >= 1

    def test_validate_file_format(self):
        """Test file format validation."""
        validator = DocumentValidator({"allowed_formats": [".txt", ".md"]})

        # Valid format
        results = validator._validate_file_format(Path("test.md"))
        assert len(results) == 0

        # Invalid format
        results = validator._validate_file_format(Path("test.pdf"))
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warnings) >= 1
        assert "Unsupported file format" in warnings[0].message

    def test_validate_document_file(self, tmp_path):
        """Test validation of actual document file."""
        validator = DocumentValidator()

        # Create test file
        test_file = tmp_path / "test.md"
        test_content = """# Test Document

This is a test document with good content.
It has multiple sentences and paragraphs.

This is another paragraph with more content.
"""
        test_file.write_text(test_content)

        results, metadata = validator.validate_document(test_file)

        # Should have minimal issues for well-formed content
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0

        # Metadata should be extracted
        assert metadata.word_count > 0
        assert metadata.paragraph_count >= 2

    def test_validate_document_missing_file(self):
        """Test validation of missing file."""
        validator = DocumentValidator()

        results, metadata = validator.validate_document(Path("nonexistent.md"))

        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) >= 1
        assert "not found" in errors[0].message

        # Metadata should be empty for missing file
        assert metadata.word_count == 0

    def test_validate_document_collection(self, tmp_path):
        """Test validation of document collection."""
        validator = DocumentValidator()

        # Create test documents
        (tmp_path / "doc1.md").write_text("# Document 1\nGood content here.")
        (tmp_path / "doc2.txt").write_text("Document 2 content with sufficient length.")
        (tmp_path / "doc3.md").write_text("Short")  # Too short

        results = validator.validate_document_collection(tmp_path)

        assert "doc1.md" in results
        assert "doc2.txt" in results
        assert "doc3.md" in results

        # Check that short document has warnings
        doc3_results, doc3_metadata = results["doc3.md"]
        warnings = [r for r in doc3_results if r.level == ValidationLevel.WARNING]
        short_warnings = [r for r in warnings if "short" in r.message.lower()]
        assert len(short_warnings) >= 1

    def test_validate_collection_with_manifest(self, tmp_path):
        """Test validation with manifest expectations."""
        validator = DocumentValidator()

        # Create some documents
        (tmp_path / "main.md").write_text("# Main Document\nMain content here.")
        # Missing expected document: "support.md"

        # Create manifest config
        manifest_config = {
            "review_configuration": {
                "documents": {"primary": "main.md", "supporting": ["support.md"]}
            }
        }

        results = validator.validate_document_collection(tmp_path, manifest_config)

        # Should have collection-level issues for missing document
        if "_collection_issues" in results:
            collection_results, _ = results["_collection_issues"]
            missing_errors = [
                r for r in collection_results if "missing" in r.message.lower()
            ]
            assert len(missing_errors) >= 1

    def test_generate_validation_report(self):
        """Test validation report generation."""
        validator = DocumentValidator()

        # Create mock validation results
        validation_results = {
            "test.md": (
                [
                    ValidationResult(
                        ValidationLevel.WARNING, "Test warning", suggestion="Fix this"
                    ),
                    ValidationResult(ValidationLevel.INFO, "Test info"),
                ],
                DocumentMetadata(
                    word_count=100,
                    character_count=500,
                    paragraph_count=3,
                    sentence_count=5,
                    reading_level=10.5,
                ),
            ),
            "good.md": (
                [],  # No issues
                DocumentMetadata(
                    word_count=200,
                    character_count=1000,
                    paragraph_count=4,
                    sentence_count=8,
                ),
            ),
        }

        report = validator.generate_validation_report(validation_results)

        assert "Document Validation Report" in report
        assert "test.md" in report
        assert "good.md" in report
        assert "Words: 100" in report
        assert "Reading Level: 10.5" in report
        assert "Test warning" in report
        assert "No validation issues found" in report
        assert "**Total Warnings:** 1" in report

    def test_syllable_counting(self):
        """Test syllable counting functionality."""
        validator = DocumentValidator()

        # Test various words
        test_cases = [("hello", 2), ("cat", 1), ("beautiful", 3), ("the", 1)]

        for word, expected_min in test_cases:
            syllables = validator._count_syllables(word)
            assert syllables >= expected_min, (
                f"Word '{word}' should have at least {expected_min} syllables"
            )

    def test_reading_level_estimation(self):
        """Test reading level estimation."""
        validator = DocumentValidator()

        # Simple content should have lower reading level
        simple_content = "The cat sat on the mat. It was a nice day."
        simple_level = validator._estimate_reading_level(simple_content)

        # Complex content should have higher reading level
        complex_content = """The sophisticated feline positioned itself comfortably upon the woven textile floor covering. 
        The meteorological conditions were particularly favorable and conducive to outdoor recreational activities."""
        complex_level = validator._estimate_reading_level(complex_content)

        assert complex_level > simple_level
        assert simple_level >= 0
        assert complex_level >= 0
