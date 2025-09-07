"""
Tests for CLI functionality.

Tests the manifest-driven workflows, directory processing, and command-line interface.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from src.cli.main import cli


class TestCLI:
    """Test CLI functionality for manifest-driven workflows and command-line interface."""

    @patch('src.cli.main.ConversationManager')
    def test_directory_with_manifest_detection(self, mock_manager_class):
        """Test that CLI detects and reports manifest presence in directories."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test document
            (temp_path / "test.md").write_text("# Test Document\nContent here.")
            
            # Create a manifest
            manifest_content = {
                "review_configuration": {
                    "name": "Test Review",
                    "reviewer_categories": ["content"]
                }
            }
            (temp_path / "manifest.yaml").write_text(yaml.dump(manifest_content))
            
            # Run CLI on directory
            result = runner.invoke(cli, [
                'review', str(temp_path), 
                '--provider', 'lm_studio'
            ])
            
            # Check that manifest detection message appears
            assert "📋 Found manifest file - will use custom reviewer selection" in result.output
            assert "📂 Processing document collection from:" in result.output

    @patch('src.cli.main.ConversationManager')
    def test_directory_without_manifest_detection(self, mock_manager_class):
        """Test that CLI detects and reports lack of manifest in directories."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test document (no manifest)
            (temp_path / "test.md").write_text("# Test Document\nContent here.")
            
            # Run CLI on directory
            result = runner.invoke(cli, [
                'review', str(temp_path), 
                '--provider', 'lm_studio'
            ])
            
            # Check that no-manifest message appears
            assert "📄 No manifest found - will use all available reviewers" in result.output
            assert "📂 Processing document collection from:" in result.output

    @patch('src.cli.main.ConversationManager')
    def test_agents_flag_warning_with_manifest(self, mock_manager_class):
        """Test warning when --agents flag is used with directory containing manifest."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test document
            (temp_path / "test.md").write_text("# Test Document\nContent here.")
            
            # Create a manifest
            manifest_content = {
                "review_configuration": {
                    "name": "Test Review",
                    "reviewers": ["Content Reviewer"]
                }
            }
            (temp_path / "manifest.yaml").write_text(yaml.dump(manifest_content))
            
            # Run CLI with --agents flag
            result = runner.invoke(cli, [
                'review', str(temp_path),
                '--agents', 'Technical Reviewer',
                '--provider', 'lm_studio'
            ])
            
            # Check that warning appears
            assert "⚠️  Warning: --agents flag will be ignored because manifest.yaml found in directory" in result.output
            assert "The manifest will determine reviewer selection" in result.output

    @patch('src.cli.main.ConversationManager')
    def test_agents_flag_no_warning_without_manifest(self, mock_manager_class):
        """Test no warning when --agents flag is used with directory without manifest."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test document (no manifest)
            (temp_path / "test.md").write_text("# Test Document\nContent here.")
            
            # Run CLI with --agents flag
            result = runner.invoke(cli, [
                'review', str(temp_path),
                '--agents', 'Technical Reviewer',
                '--provider', 'lm_studio'
            ])
            
            # Check that warning does NOT appear
            assert "⚠️  Warning: --agents flag will be ignored" not in result.output
            assert "📄 No manifest found - will use all available reviewers" in result.output

    @patch('src.cli.main.ConversationManager')
    def test_agents_flag_no_warning_with_file(self, mock_manager_class):
        """Test no warning when --agents flag is used with single file."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test file
            test_file = temp_path / "test.md"
            test_file.write_text("# Test Document\nContent here.")
            
            # Run CLI with --agents flag on file
            result = runner.invoke(cli, [
                'review', str(test_file),
                '--agents', 'Technical Reviewer',
                '--provider', 'lm_studio'
            ])
            
            # Check that warning does NOT appear
            assert "⚠️  Warning: --agents flag will be ignored" not in result.output
            assert "📁 Reading content from:" in result.output

    def test_help_text_includes_manifest_documentation(self):
        """Test that help text documents manifest functionality."""
        runner = CliRunner()
        result = runner.invoke(cli, ['review', '--help'])
        
        # Check that help includes manifest documentation
        assert "multi-document review" in result.output
        assert "manifest.yaml" in result.output
        assert "reviewer categories" in result.output

    @patch('src.cli.main.ConversationManager')
    def test_directory_processing_messages(self, mock_manager_class):
        """Test that directory processing provides clear user feedback."""
        # Mock the ConversationManager to prevent actual model calls
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple test documents
            (temp_path / "doc1.md").write_text("# Document 1\nContent here.")
            (temp_path / "doc2.txt").write_text("Document 2 content.")
            
            # Run CLI
            result = runner.invoke(cli, [
                'review', str(temp_path),
                '--provider', 'lm_studio'
            ])
            
            # Check informative messages
            assert "📂 Processing document collection from:" in result.output
            assert "📄 No manifest found - will use all available reviewers" in result.output

    def test_max_context_length_parameter_removed(self):
        """Test that --max-context-length parameter is no longer accepted."""
        runner = CliRunner()
        
        # Try to use the removed parameter
        result = runner.invoke(cli, [
            'review', 'test content',
            '--max-context-length', '8192',
            '--provider', 'lm_studio'
        ])
        
        # Should fail with "no such option" error
        assert result.exit_code != 0
        assert "No such option: --max-context-length" in result.output

    def test_help_text_no_max_context_length(self):
        """Test that help text no longer mentions --max-context-length."""
        runner = CliRunner()
        result = runner.invoke(cli, ['review', '--help'])
        
        # Should not mention the removed parameter
        assert "--max-context-length" not in result.output
        assert "max-context-length" not in result.output

    @patch('src.cli.main.ConversationManager')
    def test_model_id_override_from_cli(self, mock_manager_class):
        """Test that model-id can be overridden from CLI."""
        # Mock the ConversationManager
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_result = MagicMock()
        mock_manager.run_review.return_value = mock_result
        mock_manager.format_results.return_value = "Mock review results"
        
        runner = CliRunner()
        
        # Run CLI with model-id override
        result = runner.invoke(cli, [
            'review', 'test content',
            '--provider', 'lm_studio',
            '--model-id', 'custom-model-123'
        ])
        
        # Should succeed
        assert result.exit_code == 0
        
        # Check that ConversationManager was called with model config
        mock_manager_class.assert_called_once()
        call_kwargs = mock_manager_class.call_args[1]
        assert 'model_config' in call_kwargs
        assert call_kwargs['model_config']['model_id'] == 'custom-model-123'