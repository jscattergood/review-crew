"""
Export formatters for different output formats.

This module provides formatters for exporting review results to various
structured formats including JSON, HTML, and summary reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class ExportMetadata:
    """Metadata for exported review results."""
    export_timestamp: str
    export_format: str
    review_type: str
    document_count: int
    reviewer_count: int
    analyzer_count: int
    total_review_time: Optional[float] = None
    export_version: str = "1.0"


class BaseExporter(ABC):
    """Base class for result exporters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize exporter with configuration.
        
        Args:
            config: Export configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def export(self, result: Any, output_path: Path, metadata: Optional[ExportMetadata] = None) -> bool:
        """Export review result to specified format.
        
        Args:
            result: ConversationResult to export
            output_path: Path where to save the exported file
            metadata: Optional export metadata
            
        Returns:
            True if export successful, False otherwise
        """
        pass


class JSONExporter(BaseExporter):
    """Export review results to structured JSON format."""
    
    def export(self, result: Any, output_path: Path, metadata: Optional[ExportMetadata] = None) -> bool:
        """Export to JSON format.
        
        Args:
            result: ConversationResult to export
            output_path: Path to JSON file
            metadata: Optional export metadata
            
        Returns:
            True if successful
        """
        try:
            export_data = self._convert_to_dict(result, metadata)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            print(f"❌ JSON export failed: {e}")
            return False
    
    def _convert_to_dict(self, result: Any, metadata: Optional[ExportMetadata]) -> Dict[str, Any]:
        """Convert ConversationResult to dictionary format."""
        export_data = {
            "metadata": asdict(metadata) if metadata else {},
            "review_results": [],
            "analysis_results": [],
            "context_results": [],
            "summary": {}
        }
        
        # Extract review results
        if hasattr(result, 'review_results') and result.review_results:
            for review in result.review_results:
                review_data = {
                    "reviewer_name": getattr(review, 'reviewer_name', 'Unknown'),
                    "feedback": self._extract_feedback(review),
                    "timestamp": getattr(review, 'timestamp', None),
                    "status": getattr(review, 'status', 'completed')
                }
                export_data["review_results"].append(review_data)
        
        # Extract analysis results
        if hasattr(result, 'analysis_results') and result.analysis_results:
            for analysis in result.analysis_results:
                analysis_data = {
                    "analyzer_name": getattr(analysis, 'analyzer_name', 'Unknown'),
                    "analysis": self._extract_analysis(analysis),
                    "timestamp": getattr(analysis, 'timestamp', None)
                }
                export_data["analysis_results"].append(analysis_data)
        
        # Extract context results
        if hasattr(result, 'context_results') and result.context_results:
            for context in result.context_results:
                context_data = {
                    "contextualizer_name": getattr(context, 'contextualizer_name', 'Unknown'),
                    "context": self._extract_context(context),
                    "timestamp": getattr(context, 'timestamp', None)
                }
                export_data["context_results"].append(context_data)
        
        # Generate summary
        export_data["summary"] = {
            "total_reviewers": len(export_data["review_results"]),
            "total_analyzers": len(export_data["analysis_results"]),
            "total_contextualizers": len(export_data["context_results"]),
            "has_errors": hasattr(result, 'review_errors') and bool(result.review_errors)
        }
        
        return export_data
    
    def _extract_feedback(self, review: Any) -> str:
        """Extract feedback text from review object."""
        if hasattr(review, 'feedback'):
            feedback = review.feedback
            if isinstance(feedback, dict):
                return feedback.get('content', str(feedback))
            return str(feedback)
        return ""
    
    def _extract_analysis(self, analysis: Any) -> str:
        """Extract analysis text from analysis object."""
        if hasattr(analysis, 'analysis'):
            return str(analysis.analysis)
        return str(analysis)
    
    def _extract_context(self, context: Any) -> str:
        """Extract context text from context object."""
        if hasattr(context, 'context'):
            return str(context.context)
        return str(context)


class HTMLReportExporter(BaseExporter):
    """Export review results to HTML report format."""
    
    def export(self, result: Any, output_path: Path, metadata: Optional[ExportMetadata] = None) -> bool:
        """Export to HTML report format.
        
        Args:
            result: ConversationResult to export
            output_path: Path to HTML file
            metadata: Optional export metadata
            
        Returns:
            True if successful
        """
        try:
            html_content = self._generate_html_report(result, metadata)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
        except Exception as e:
            print(f"❌ HTML export failed: {e}")
            return False
    
    def _generate_html_report(self, result: Any, metadata: Optional[ExportMetadata]) -> str:
        """Generate HTML report content."""
        html_parts = []
        
        # HTML header
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review-Crew Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .review-section { border-left: 4px solid #3498db; padding-left: 20px; margin: 20px 0; }
        .analysis-section { border-left: 4px solid #e74c3c; padding-left: 20px; margin: 20px 0; }
        .context-section { border-left: 4px solid #f39c12; padding-left: 20px; margin: 20px 0; }
        .summary { background: #d5dbdb; padding: 15px; border-radius: 5px; margin: 20px 0; }
        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">""")
        
        # Title and metadata
        html_parts.append("<h1>📋 Review-Crew Report</h1>")
        
        if metadata:
            html_parts.append('<div class="metadata">')
            html_parts.append(f"<strong>Generated:</strong> {metadata.export_timestamp}<br>")
            html_parts.append(f"<strong>Format:</strong> {metadata.export_format}<br>")
            html_parts.append(f"<strong>Review Type:</strong> {metadata.review_type}<br>")
            html_parts.append(f"<strong>Documents:</strong> {metadata.document_count}<br>")
            html_parts.append(f"<strong>Reviewers:</strong> {metadata.reviewer_count}<br>")
            html_parts.append(f"<strong>Analyzers:</strong> {metadata.analyzer_count}")
            html_parts.append("</div>")
        
        # Review results
        if hasattr(result, 'review_results') and result.review_results:
            html_parts.append("<h2>🔍 Review Results</h2>")
            for i, review in enumerate(result.review_results, 1):
                reviewer_name = getattr(review, 'reviewer_name', f'Reviewer {i}')
                feedback = self._extract_feedback_html(review)
                timestamp = getattr(review, 'timestamp', '')
                
                html_parts.append(f'<div class="review-section">')
                html_parts.append(f"<h3>{reviewer_name}</h3>")
                if timestamp:
                    html_parts.append(f'<div class="timestamp">Generated: {timestamp}</div>')
                html_parts.append(f"<div>{self._format_content_html(feedback)}</div>")
                html_parts.append("</div>")
        
        # Analysis results
        if hasattr(result, 'analysis_results') and result.analysis_results:
            html_parts.append("<h2>🧠 Analysis Results</h2>")
            for i, analysis in enumerate(result.analysis_results, 1):
                analyzer_name = getattr(analysis, 'analyzer_name', f'Analyzer {i}')
                analysis_content = self._extract_analysis_html(analysis)
                timestamp = getattr(analysis, 'timestamp', '')
                
                html_parts.append(f'<div class="analysis-section">')
                html_parts.append(f"<h3>{analyzer_name}</h3>")
                if timestamp:
                    html_parts.append(f'<div class="timestamp">Generated: {timestamp}</div>')
                html_parts.append(f"<div>{self._format_content_html(analysis_content)}</div>")
                html_parts.append("</div>")
        
        # Context results
        if hasattr(result, 'context_results') and result.context_results:
            html_parts.append("<h2>📚 Context Information</h2>")
            for i, context in enumerate(result.context_results, 1):
                contextualizer_name = getattr(context, 'contextualizer_name', f'Contextualizer {i}')
                context_content = self._extract_context_html(context)
                
                html_parts.append(f'<div class="context-section">')
                html_parts.append(f"<h3>{contextualizer_name}</h3>")
                html_parts.append(f"<div>{self._format_content_html(context_content)}</div>")
                html_parts.append("</div>")
        
        # Summary
        html_parts.append('<div class="summary">')
        html_parts.append("<h2>📊 Summary</h2>")
        review_count = len(result.review_results) if hasattr(result, 'review_results') and result.review_results else 0
        analysis_count = len(result.analysis_results) if hasattr(result, 'analysis_results') and result.analysis_results else 0
        html_parts.append(f"<strong>Total Reviews:</strong> {review_count}<br>")
        html_parts.append(f"<strong>Total Analyses:</strong> {analysis_count}<br>")
        if hasattr(result, 'review_errors') and result.review_errors:
            html_parts.append(f"<strong>Errors:</strong> {len(result.review_errors)}")
        html_parts.append("</div>")
        
        # HTML footer
        html_parts.append("""
    </div>
</body>
</html>""")
        
        return "\n".join(html_parts)
    
    def _extract_feedback_html(self, review: Any) -> str:
        """Extract feedback for HTML format."""
        if hasattr(review, 'feedback'):
            feedback = review.feedback
            if isinstance(feedback, dict):
                return feedback.get('content', str(feedback))
            return str(feedback)
        return ""
    
    def _extract_analysis_html(self, analysis: Any) -> str:
        """Extract analysis for HTML format."""
        if hasattr(analysis, 'analysis'):
            return str(analysis.analysis)
        return str(analysis)
    
    def _extract_context_html(self, context: Any) -> str:
        """Extract context for HTML format."""
        if hasattr(context, 'context'):
            return str(context.context)
        return str(context)
    
    def _format_content_html(self, content: str) -> str:
        """Format content for HTML display."""
        if not content:
            return "<em>No content available</em>"
        
        # Convert markdown-like formatting to HTML
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        content = f"<p>{content}</p>"
        
        # Handle basic markdown
        content = content.replace('**', '<strong>').replace('**', '</strong>')
        content = content.replace('*', '<em>').replace('*', '</em>')
        
        return content


class SummaryExporter(BaseExporter):
    """Export concise summary of review results."""
    
    def export(self, result: Any, output_path: Path, metadata: Optional[ExportMetadata] = None) -> bool:
        """Export to summary format.
        
        Args:
            result: ConversationResult to export
            output_path: Path to summary file
            metadata: Optional export metadata
            
        Returns:
            True if successful
        """
        try:
            summary_content = self._generate_summary(result, metadata)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            return True
        except Exception as e:
            print(f"❌ Summary export failed: {e}")
            return False
    
    def _generate_summary(self, result: Any, metadata: Optional[ExportMetadata]) -> str:
        """Generate summary content."""
        lines = []
        
        # Header
        lines.append("# Review Summary")
        lines.append("")
        
        if metadata:
            lines.append(f"**Generated:** {metadata.export_timestamp}")
            lines.append(f"**Review Type:** {metadata.review_type}")
            lines.append(f"**Documents:** {metadata.document_count}")
            lines.append("")
        
        # Key metrics
        review_count = len(result.review_results) if hasattr(result, 'review_results') and result.review_results else 0
        analysis_count = len(result.analysis_results) if hasattr(result, 'analysis_results') and result.analysis_results else 0
        
        lines.append("## Key Metrics")
        lines.append(f"- **Reviews Completed:** {review_count}")
        lines.append(f"- **Analyses Completed:** {analysis_count}")
        
        if hasattr(result, 'review_errors') and result.review_errors:
            lines.append(f"- **Errors:** {len(result.review_errors)}")
        
        lines.append("")
        
        # Reviewer summary
        if hasattr(result, 'review_results') and result.review_results:
            lines.append("## Reviewers")
            for review in result.review_results:
                reviewer_name = getattr(review, 'reviewer_name', 'Unknown Reviewer')
                lines.append(f"- {reviewer_name}")
            lines.append("")
        
        # Analysis summary
        if hasattr(result, 'analysis_results') and result.analysis_results:
            lines.append("## Analyzers")
            for analysis in result.analysis_results:
                analyzer_name = getattr(analysis, 'analyzer_name', 'Unknown Analyzer')
                lines.append(f"- {analyzer_name}")
            lines.append("")
        
        # Status
        lines.append("## Status")
        if hasattr(result, 'review_errors') and result.review_errors:
            lines.append("⚠️ **Completed with errors**")
            for error in result.review_errors:
                lines.append(f"- {error}")
        else:
            lines.append("✅ **Completed successfully**")
        
        return "\n".join(lines)


class ExportManager:
    """Manages different export formats and handles export requests."""
    
    def __init__(self):
        """Initialize export manager with available formatters."""
        self.exporters = {
            'json': JSONExporter(),
            'html': HTMLReportExporter(),
            'summary': SummaryExporter()
        }
    
    def export_result(
        self, 
        result: Any, 
        output_path: Path, 
        format_type: str = 'json',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Export result in specified format.
        
        Args:
            result: ConversationResult to export
            output_path: Path where to save exported file
            format_type: Export format ('json', 'html', 'summary')
            metadata: Optional metadata dictionary
            
        Returns:
            True if export successful
        """
        if format_type not in self.exporters:
            print(f"❌ Unsupported export format: {format_type}")
            return False
        
        # Create export metadata
        export_metadata = self._create_metadata(result, format_type, metadata)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export using appropriate formatter
        exporter = self.exporters[format_type]
        success = exporter.export(result, output_path, export_metadata)
        
        if success:
            print(f"✅ Exported {format_type.upper()} report to: {output_path}")
        
        return success
    
    def _create_metadata(
        self, 
        result: Any, 
        format_type: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> ExportMetadata:
        """Create export metadata from result and provided metadata."""
        review_count = len(result.review_results) if hasattr(result, 'review_results') and result.review_results else 0
        analysis_count = len(result.analysis_results) if hasattr(result, 'analysis_results') and result.analysis_results else 0
        
        return ExportMetadata(
            export_timestamp=datetime.now().isoformat(),
            export_format=format_type,
            review_type=metadata.get('review_type', 'multi-document') if metadata else 'multi-document',
            document_count=metadata.get('document_count', 0) if metadata else 0,
            reviewer_count=review_count,
            analyzer_count=analysis_count,
            total_review_time=metadata.get('total_review_time') if metadata else None
        )
    
    def get_available_formats(self) -> List[str]:
        """Get list of available export formats."""
        return list(self.exporters.keys())
