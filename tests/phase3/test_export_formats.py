"""
Tests for structured export formats.

Tests JSON, HTML, and summary export functionality including
export managers and CLI integration.
"""

import pytest
import json
from pathlib import Path
from src.export.formatters import (
    JSONExporter, HTMLReportExporter, SummaryExporter, 
    ExportManager, ExportMetadata
)


class MockResult:
    """Mock ConversationResult for testing."""
    
    def __init__(self):
        self.review_results = [
            type('MockReview', (), {
                'reviewer_name': 'Test Reviewer 1',
                'feedback': 'This is test feedback from reviewer 1.',
                'timestamp': '2024-01-01T12:00:00',
                'status': 'completed'
            })(),
            type('MockReview', (), {
                'reviewer_name': 'Test Reviewer 2',
                'feedback': {'content': 'This is structured feedback from reviewer 2.'},
                'timestamp': '2024-01-01T12:05:00',
                'status': 'completed'
            })()
        ]
        
        self.analysis_results = [
            type('MockAnalysis', (), {
                'analyzer_name': 'Test Analyzer',
                'analysis': 'This is test analysis content.',
                'timestamp': '2024-01-01T12:10:00'
            })()
        ]
        
        self.context_results = [
            type('MockContext', (), {
                'contextualizer_name': 'Test Contextualizer',
                'context': 'This is test context information.',
                'timestamp': '2024-01-01T12:15:00'
            })()
        ]
        
        self.review_errors = []


class TestJSONExporter:
    """Test JSON export functionality."""
    
    def test_json_export_success(self, tmp_path):
        """Test successful JSON export."""
        exporter = JSONExporter()
        result = MockResult()
        output_path = tmp_path / "test_export.json"
        
        metadata = ExportMetadata(
            export_timestamp="2024-01-01T12:00:00",
            export_format="json",
            review_type="test",
            document_count=2,
            reviewer_count=2,
            analyzer_count=1
        )
        
        success = exporter.export(result, output_path, metadata)
        
        assert success is True
        assert output_path.exists()
        
        # Verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "review_results" in data
        assert "analysis_results" in data
        assert "context_results" in data
        assert "summary" in data
        
        assert len(data["review_results"]) == 2
        assert len(data["analysis_results"]) == 1
        assert len(data["context_results"]) == 1
        
        # Check metadata
        assert data["metadata"]["export_format"] == "json"
        assert data["metadata"]["reviewer_count"] == 2
        
        # Check summary
        assert data["summary"]["total_reviewers"] == 2
        assert data["summary"]["total_analyzers"] == 1
        assert data["summary"]["has_errors"] is False

    def test_json_export_with_errors(self, tmp_path):
        """Test JSON export with review errors."""
        exporter = JSONExporter()
        result = MockResult()
        result.review_errors = ["Test error 1", "Test error 2"]
        
        output_path = tmp_path / "test_export_errors.json"
        
        success = exporter.export(result, output_path)
        assert success is True
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["summary"]["has_errors"] is True

    def test_json_extract_feedback(self):
        """Test feedback extraction from different formats."""
        exporter = JSONExporter()
        
        # String feedback
        review1 = type('MockReview', (), {'feedback': 'Simple string feedback'})()
        feedback1 = exporter._extract_feedback(review1)
        assert feedback1 == 'Simple string feedback'
        
        # Dict feedback
        review2 = type('MockReview', (), {'feedback': {'content': 'Dict feedback'}})()
        feedback2 = exporter._extract_feedback(review2)
        assert feedback2 == 'Dict feedback'
        
        # No feedback
        review3 = type('MockReview', (), {})()
        feedback3 = exporter._extract_feedback(review3)
        assert feedback3 == ""


class TestHTMLReportExporter:
    """Test HTML report export functionality."""
    
    def test_html_export_success(self, tmp_path):
        """Test successful HTML export."""
        exporter = HTMLReportExporter()
        result = MockResult()
        output_path = tmp_path / "test_export.html"
        
        metadata = ExportMetadata(
            export_timestamp="2024-01-01T12:00:00",
            export_format="html",
            review_type="test",
            document_count=2,
            reviewer_count=2,
            analyzer_count=1
        )
        
        success = exporter.export(result, output_path, metadata)
        
        assert success is True
        assert output_path.exists()
        
        # Verify HTML content
        with open(output_path, 'r') as f:
            html_content = f.read()
        
        assert "<!DOCTYPE html>" in html_content
        assert "Review-Crew Report" in html_content
        assert "Test Reviewer 1" in html_content
        assert "Test Analyzer" in html_content
        assert "Test Contextualizer" in html_content
        
        # Check metadata section
        assert "Generated: 2024-01-01T12:00:00" in html_content
        assert "<strong>Documents:</strong> 2" in html_content
        assert "<strong>Reviewers:</strong> 2" in html_content
        
        # Check styling
        assert "<style>" in html_content
        assert "font-family:" in html_content

    def test_html_format_content(self):
        """Test HTML content formatting."""
        exporter = HTMLReportExporter()
        
        # Test markdown-like formatting
        content = "This is **bold** and *italic* text.\n\nNew paragraph here."
        formatted = exporter._format_content_html(content)
        
        assert "<p>" in formatted
        assert "</p>" in formatted
        # Check that content is properly formatted with paragraphs
        assert "<p>" in formatted and "</p>" in formatted
        
        # Test empty content
        empty_formatted = exporter._format_content_html("")
        assert "<em>No content available</em>" in empty_formatted


class TestSummaryExporter:
    """Test summary export functionality."""
    
    def test_summary_export_success(self, tmp_path):
        """Test successful summary export."""
        exporter = SummaryExporter()
        result = MockResult()
        output_path = tmp_path / "test_summary.md"
        
        metadata = ExportMetadata(
            export_timestamp="2024-01-01T12:00:00",
            export_format="summary",
            review_type="multi-document",
            document_count=3,
            reviewer_count=2,
            analyzer_count=1
        )
        
        success = exporter.export(result, output_path, metadata)
        
        assert success is True
        assert output_path.exists()
        
        # Verify summary content
        with open(output_path, 'r') as f:
            summary_content = f.read()
        
        assert "# Review Summary" in summary_content
        assert "**Generated:** 2024-01-01T12:00:00" in summary_content
        assert "**Review Type:** multi-document" in summary_content
        assert "**Documents:** 3" in summary_content
        assert "**Reviews Completed:** 2" in summary_content
        assert "**Analyses Completed:** 1" in summary_content
        assert "Test Reviewer 1" in summary_content
        assert "Test Analyzer" in summary_content
        assert "✅ **Completed successfully**" in summary_content

    def test_summary_export_with_errors(self, tmp_path):
        """Test summary export with errors."""
        exporter = SummaryExporter()
        result = MockResult()
        result.review_errors = ["Error 1", "Error 2"]
        
        output_path = tmp_path / "test_summary_errors.md"
        
        success = exporter.export(result, output_path)
        assert success is True
        
        with open(output_path, 'r') as f:
            summary_content = f.read()
        
        assert "⚠️ **Completed with errors**" in summary_content
        assert "Error 1" in summary_content
        assert "Error 2" in summary_content


class TestExportManager:
    """Test export manager functionality."""
    
    def test_export_manager_initialization(self):
        """Test export manager initialization."""
        manager = ExportManager()
        
        available_formats = manager.get_available_formats()
        assert "json" in available_formats
        assert "html" in available_formats
        assert "summary" in available_formats
        assert len(available_formats) == 3

    def test_export_result_json(self, tmp_path):
        """Test exporting result in JSON format."""
        manager = ExportManager()
        result = MockResult()
        output_path = tmp_path / "manager_test.json"
        
        success = manager.export_result(result, output_path, "json")
        
        assert success is True
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert data["metadata"]["export_format"] == "json"

    def test_export_result_html(self, tmp_path):
        """Test exporting result in HTML format."""
        manager = ExportManager()
        result = MockResult()
        output_path = tmp_path / "manager_test.html"
        
        metadata = {"review_type": "test", "document_count": 5}
        success = manager.export_result(result, output_path, "html", metadata)
        
        assert success is True
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            html_content = f.read()
        
        assert "Review-Crew Report" in html_content
        assert "<strong>Documents:</strong> 5" in html_content

    def test_export_result_summary(self, tmp_path):
        """Test exporting result in summary format."""
        manager = ExportManager()
        result = MockResult()
        output_path = tmp_path / "manager_test.md"
        
        success = manager.export_result(result, output_path, "summary")
        
        assert success is True
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            summary_content = f.read()
        
        assert "# Review Summary" in summary_content

    def test_export_result_unsupported_format(self, tmp_path):
        """Test exporting with unsupported format."""
        manager = ExportManager()
        result = MockResult()
        output_path = tmp_path / "test.pdf"
        
        success = manager.export_result(result, output_path, "pdf")
        
        assert success is False

    def test_export_result_creates_directory(self, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        manager = ExportManager()
        result = MockResult()
        
        # Create nested path that doesn't exist
        output_path = tmp_path / "nested" / "directory" / "test.json"
        
        success = manager.export_result(result, output_path, "json")
        
        assert success is True
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_create_metadata(self):
        """Test metadata creation."""
        manager = ExportManager()
        result = MockResult()
        
        metadata_dict = {"review_type": "test", "document_count": 3}
        metadata = manager._create_metadata(result, "json", metadata_dict)
        
        assert metadata.export_format == "json"
        assert metadata.review_type == "test"
        assert metadata.document_count == 3
        assert metadata.reviewer_count == 2  # From MockResult
        assert metadata.analyzer_count == 1   # From MockResult
        assert metadata.export_timestamp is not None


class TestExportMetadata:
    """Test export metadata functionality."""
    
    def test_export_metadata_creation(self):
        """Test creating export metadata."""
        metadata = ExportMetadata(
            export_timestamp="2024-01-01T12:00:00",
            export_format="json",
            review_type="multi-document",
            document_count=5,
            reviewer_count=3,
            analyzer_count=2,
            total_review_time=120.5
        )
        
        assert metadata.export_timestamp == "2024-01-01T12:00:00"
        assert metadata.export_format == "json"
        assert metadata.review_type == "multi-document"
        assert metadata.document_count == 5
        assert metadata.reviewer_count == 3
        assert metadata.analyzer_count == 2
        assert metadata.total_review_time == 120.5
        assert metadata.export_version == "1.0"

    def test_export_metadata_defaults(self):
        """Test export metadata with default values."""
        metadata = ExportMetadata(
            export_timestamp="2024-01-01T12:00:00",
            export_format="html",
            review_type="single-document",
            document_count=1,
            reviewer_count=2,
            analyzer_count=1
        )
        
        assert metadata.total_review_time is None
        assert metadata.export_version == "1.0"
