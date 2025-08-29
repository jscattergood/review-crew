"""
Document validation pipeline for ensuring quality and consistency.

This module provides comprehensive validation for documents before review,
including format validation, content quality checks, and metadata extraction.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of document validation."""

    level: ValidationLevel
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Extracted document metadata."""

    word_count: int
    character_count: int
    paragraph_count: int
    sentence_count: int
    reading_level: Optional[float] = None
    detected_language: str = "en"
    format_type: str = "text"


class DocumentValidator:
    """Comprehensive document validation and pre-processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration.

        Args:
            config: Validation configuration dictionary
        """
        self.config = config or {}
        self.default_rules = {
            "max_word_count": 1000,
            "min_word_count": 50,
            "max_paragraph_length": 200,
            "require_title": False,
            "check_spelling": False,
            "check_grammar": False,
            "allowed_formats": [".txt", ".md", ".docx"],
            "encoding": "utf-8",
        }
        self.rules = {**self.default_rules, **self.config}

    def validate_document(
        self, file_path: Path
    ) -> Tuple[List[ValidationResult], DocumentMetadata]:
        """Validate a document file comprehensively.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of validation results and extracted metadata
        """
        results = []

        # File existence and format validation
        format_results = self._validate_file_format(file_path)
        results.extend(format_results)

        if not file_path.exists():
            results.append(
                ValidationResult(
                    ValidationLevel.ERROR,
                    f"Document file not found: {file_path}",
                    suggestion="Ensure the file path is correct and the file exists",
                )
            )
            return results, DocumentMetadata(0, 0, 0, 0)

        # Read and validate content
        try:
            content = self._read_document(file_path)
        except Exception as e:
            results.append(
                ValidationResult(
                    ValidationLevel.ERROR,
                    f"Failed to read document: {e}",
                    suggestion="Check file encoding and permissions",
                )
            )
            return results, DocumentMetadata(0, 0, 0, 0)

        # Extract metadata
        metadata = self._extract_metadata(content)

        # Content validation
        content_results = self._validate_content(content, metadata)
        results.extend(content_results)

        # Structure validation
        structure_results = self._validate_structure(content)
        results.extend(structure_results)

        # Quality checks
        quality_results = self._validate_quality(content, metadata)
        results.extend(quality_results)

        return results, metadata

    def validate_document_collection(
        self, directory_path: Path, manifest_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[List[ValidationResult], DocumentMetadata]]:
        """Validate a collection of documents.

        Args:
            directory_path: Path to directory containing documents
            manifest_config: Optional manifest configuration

        Returns:
            Dictionary mapping file names to validation results and metadata
        """
        results = {}

        # Get expected documents from manifest if provided
        expected_docs = self._get_expected_documents(manifest_config)

        # Validate each document
        for file_path in directory_path.iterdir():
            if file_path.is_file() and self._is_document_file(file_path):
                validation_results, metadata = self.validate_document(file_path)
                results[file_path.name] = (validation_results, metadata)

        # Check for missing expected documents
        if expected_docs:
            missing_results = self._check_missing_documents(
                directory_path, expected_docs
            )
            if missing_results:
                results["_collection_issues"] = (
                    missing_results,
                    DocumentMetadata(0, 0, 0, 0),
                )
        print(f"Validation results: {results}")
        return results

    def _validate_file_format(self, file_path: Path) -> List[ValidationResult]:
        """Validate file format and extension."""
        results = []

        if not file_path.suffix.lower() in self.rules["allowed_formats"]:
            results.append(
                ValidationResult(
                    ValidationLevel.WARNING,
                    f"Unsupported file format: {file_path.suffix}",
                    suggestion=f"Supported formats: {', '.join(self.rules['allowed_formats'])}",
                )
            )

        return results

    def _read_document(self, file_path: Path) -> str:
        """Read document content with proper encoding."""
        encoding = self.rules["encoding"]

        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise

    def _extract_metadata(self, content: str) -> DocumentMetadata:
        """Extract comprehensive metadata from document content."""
        # Basic counts
        word_count = len(content.split())
        character_count = len(content)
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])
        sentence_count = len(re.findall(r"[.!?]+", content))

        # Estimate reading level (simplified Flesch-Kincaid)
        reading_level = self._estimate_reading_level(content)

        return DocumentMetadata(
            word_count=word_count,
            character_count=character_count,
            paragraph_count=paragraph_count,
            sentence_count=sentence_count,
            reading_level=reading_level,
            detected_language="en",  # Could be enhanced with language detection
            format_type="text",
        )

    def _validate_content(
        self, content: str, metadata: DocumentMetadata
    ) -> List[ValidationResult]:
        """Validate document content against rules."""
        results = []

        # Word count validation
        if metadata.word_count < self.rules["min_word_count"]:
            results.append(
                ValidationResult(
                    ValidationLevel.WARNING,
                    f"Document is too short: {metadata.word_count} words (minimum: {self.rules['min_word_count']})",
                    suggestion="Consider expanding the content to meet minimum requirements",
                )
            )
        elif metadata.word_count > self.rules["max_word_count"]:
            results.append(
                ValidationResult(
                    ValidationLevel.WARNING,
                    f"Document exceeds word limit: {metadata.word_count} words (maximum: {self.rules['max_word_count']})",
                    suggestion="Consider condensing the content to meet word limit requirements",
                )
            )

        # Empty content check
        if not content.strip():
            results.append(
                ValidationResult(
                    ValidationLevel.ERROR,
                    "Document is empty",
                    suggestion="Add content to the document",
                )
            )

        return results

    def _validate_structure(self, content: str) -> List[ValidationResult]:
        """Validate document structure and formatting."""
        results = []

        # Check for title if required
        if self.rules["require_title"]:
            lines = content.split("\n")
            if not lines or not lines[0].strip():
                results.append(
                    ValidationResult(
                        ValidationLevel.WARNING,
                        "Document appears to be missing a title",
                        suggestion="Consider adding a clear title at the beginning",
                    )
                )

        # Check paragraph structure
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            words_in_paragraph = len(paragraph.split())
            if words_in_paragraph > self.rules["max_paragraph_length"]:
                results.append(
                    ValidationResult(
                        ValidationLevel.INFO,
                        f"Paragraph {i+1} is very long ({words_in_paragraph} words)",
                        location=f"Paragraph {i+1}",
                        suggestion="Consider breaking long paragraphs into smaller ones for better readability",
                    )
                )

        return results

    def _validate_quality(
        self, content: str, metadata: DocumentMetadata
    ) -> List[ValidationResult]:
        """Validate content quality indicators."""
        results = []

        # Check for repeated words or phrases
        repeated_issues = self._check_repetition(content)
        results.extend(repeated_issues)

        # Check for common formatting issues
        formatting_issues = self._check_formatting(content)
        results.extend(formatting_issues)

        # Reading level assessment
        if metadata.reading_level and metadata.reading_level > 16:
            results.append(
                ValidationResult(
                    ValidationLevel.INFO,
                    f"Document has high reading level ({metadata.reading_level:.1f})",
                    suggestion="Consider simplifying complex sentences for better accessibility",
                )
            )

        return results

    def _check_repetition(self, content: str) -> List[ValidationResult]:
        """Check for excessive repetition of words or phrases."""
        results = []

        # Simple repetition check - could be enhanced
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Flag words that appear very frequently
        total_words = len(words)
        for word, count in word_freq.items():
            if count > max(5, total_words * 0.02):  # More than 2% of total words
                results.append(
                    ValidationResult(
                        ValidationLevel.INFO,
                        f"Word '{word}' appears {count} times - consider varying vocabulary",
                        suggestion="Use synonyms or rephrase to avoid repetition",
                    )
                )

        return results

    def _check_formatting(self, content: str) -> List[ValidationResult]:
        """Check for common formatting issues."""
        results = []

        # Multiple consecutive spaces
        if "  " in content:
            results.append(
                ValidationResult(
                    ValidationLevel.INFO,
                    "Document contains multiple consecutive spaces",
                    suggestion="Use single spaces between words",
                )
            )

        # Multiple consecutive line breaks
        if "\n\n\n" in content:
            results.append(
                ValidationResult(
                    ValidationLevel.INFO,
                    "Document contains excessive line breaks",
                    suggestion="Use single line breaks between paragraphs",
                )
            )

        return results

    def _estimate_reading_level(self, content: str) -> float:
        """Estimate reading level using simplified Flesch-Kincaid formula."""
        sentences = len(re.findall(r"[.!?]+", content))
        words = len(content.split())
        syllables = self._count_syllables(content)

        if sentences == 0 or words == 0:
            return 0.0

        # Simplified Flesch-Kincaid Grade Level
        grade_level = (
            (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        )
        return max(0.0, grade_level)

    def _count_syllables(self, content: str) -> int:
        """Estimate syllable count (simplified)."""
        words = re.findall(r"\b\w+\b", content.lower())
        syllable_count = 0

        for word in words:
            # Simple syllable counting heuristic
            vowels = "aeiouy"
            syllables = 0
            prev_was_vowel = False

            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllables += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False

            # Adjust for silent e
            if word.endswith("e") and syllables > 1:
                syllables -= 1

            # Ensure at least one syllable per word
            syllables = max(1, syllables)
            syllable_count += syllables

        return syllable_count

    def _get_expected_documents(
        self, manifest_config: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract expected document list from manifest."""
        if not manifest_config:
            return []

        review_config = manifest_config.get("review_configuration", {})
        documents = review_config.get("documents", {})

        expected = []
        if "primary" in documents:
            expected.append(documents["primary"])
        if "supporting" in documents:
            expected.extend(documents["supporting"])

        return expected

    def _check_missing_documents(
        self, directory_path: Path, expected_docs: List[str]
    ) -> List[ValidationResult]:
        """Check for missing expected documents."""
        results = []
        existing_files = {f.name for f in directory_path.iterdir() if f.is_file()}

        for expected_doc in expected_docs:
            if expected_doc not in existing_files:
                results.append(
                    ValidationResult(
                        ValidationLevel.ERROR,
                        f"Expected document missing: {expected_doc}",
                        suggestion=f"Add the missing document: {expected_doc}",
                    )
                )

        return results

    def _is_document_file(self, file_path: Path) -> bool:
        """Check if file is a document file based on extension."""
        return file_path.suffix.lower() in self.rules["allowed_formats"]

    def generate_validation_report(
        self,
        validation_results: Dict[str, Tuple[List[ValidationResult], DocumentMetadata]],
    ) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        report_lines.append("# Document Validation Report")
        report_lines.append("")

        total_errors = 0
        total_warnings = 0
        total_info = 0

        for filename, (results, metadata) in validation_results.items():
            if filename.startswith("_"):
                continue  # Skip collection-level issues for now

            report_lines.append(f"## {filename}")
            report_lines.append("")

            # Metadata summary
            report_lines.append("**Document Statistics:**")
            report_lines.append(f"- Words: {metadata.word_count}")
            report_lines.append(f"- Characters: {metadata.character_count}")
            report_lines.append(f"- Paragraphs: {metadata.paragraph_count}")
            report_lines.append(f"- Sentences: {metadata.sentence_count}")
            if metadata.reading_level:
                report_lines.append(f"- Reading Level: {metadata.reading_level:.1f}")
            report_lines.append("")

            # Validation results
            if results:
                errors = [r for r in results if r.level == ValidationLevel.ERROR]
                warnings = [r for r in results if r.level == ValidationLevel.WARNING]
                info = [r for r in results if r.level == ValidationLevel.INFO]

                total_errors += len(errors)
                total_warnings += len(warnings)
                total_info += len(info)

                if errors:
                    report_lines.append("**❌ Errors:**")
                    for error in errors:
                        report_lines.append(f"- {error.message}")
                        if error.suggestion:
                            report_lines.append(f"  *Suggestion: {error.suggestion}*")
                    report_lines.append("")

                if warnings:
                    report_lines.append("**⚠️ Warnings:**")
                    for warning in warnings:
                        report_lines.append(f"- {warning.message}")
                        if warning.suggestion:
                            report_lines.append(f"  *Suggestion: {warning.suggestion}*")
                    report_lines.append("")

                if info:
                    report_lines.append("**ℹ️ Information:**")
                    for info_item in info:
                        report_lines.append(f"- {info_item.message}")
                        if info_item.suggestion:
                            report_lines.append(
                                f"  *Suggestion: {info_item.suggestion}*"
                            )
                    report_lines.append("")
            else:
                report_lines.append("✅ **No validation issues found**")
                report_lines.append("")

        # Summary
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(f"- **Total Errors:** {total_errors}")
        report_lines.append(f"- **Total Warnings:** {total_warnings}")
        report_lines.append(f"- **Total Information:** {total_info}")

        if total_errors == 0:
            report_lines.append("")
            report_lines.append("✅ **All documents passed validation!**")

        return "\n".join(report_lines)
