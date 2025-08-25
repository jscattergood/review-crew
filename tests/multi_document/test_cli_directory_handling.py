"""Tests for CLI directory handling functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner

from src.cli.main import review


class TestCLIDirectoryHandling:
    """Test CLI multi-document directory handling."""
    
    def test_cli_detects_single_file(self, tmp_path):
        """Test CLI correctly detects and processes single file input."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        with patch('src.cli.main.ConversationManager') as mock_manager:
            mock_result = Mock()
            mock_manager.return_value.run_review.return_value = mock_result
            mock_manager.return_value.format_results.return_value = "Test output"
            
            runner = CliRunner()
            result = runner.invoke(review, [str(test_file)])
            
            # Should call run_review with file content (not path)
            mock_manager.return_value.run_review.assert_called_once()
            args = mock_manager.return_value.run_review.call_args[0]
            assert args[0] == "Test content"  # Content should be read from file
            assert result.exit_code == 0
    
    def test_cli_detects_directory_input(self, tmp_path):
        """Test CLI correctly detects directory input."""
        test_dir = tmp_path / "test_docs"
        test_dir.mkdir()
        
        (test_dir / "doc1.txt").write_text("Document 1 content")
        (test_dir / "doc2.md").write_text("Document 2 content")
        
        with patch('src.cli.main.ConversationManager') as mock_manager:
            mock_result = Mock()
            mock_manager.return_value.run_review.return_value = mock_result
            mock_manager.return_value.format_results.return_value = "Test output"
            
            runner = CliRunner()
            result = runner.invoke(review, [str(test_dir)])
            
            # Should call run_review with directory path (for manager to handle)
            mock_manager.return_value.run_review.assert_called_once()
            args = mock_manager.return_value.run_review.call_args[0]
            assert str(test_dir) in args[0]  # Should pass directory path
            assert result.exit_code == 0
    
    def test_cli_handles_nonexistent_path(self):
        """Test CLI handles non-existent paths gracefully."""
        with patch('src.cli.main.ConversationManager') as mock_manager:
            mock_result = Mock()
            mock_manager.return_value.run_review.return_value = mock_result
            mock_manager.return_value.format_results.return_value = "Test output"
            
            runner = CliRunner()
            result = runner.invoke(review, ["/nonexistent/path"])
            
            # Should treat as text content, not path
            mock_manager.return_value.run_review.assert_called_once()
            args = mock_manager.return_value.run_review.call_args[0]
            assert args[0] == "/nonexistent/path"  # Should pass as text
            assert result.exit_code == 0
    
    def test_cli_rejects_invalid_path_type(self, tmp_path):
        """Test CLI rejects paths that exist but are neither file nor directory."""
        # Create a symlink or other special file type
        special_file = tmp_path / "special"
        special_file.touch()
        
        # Mock as something that exists but is neither file nor directory
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=False), \
             patch('pathlib.Path.is_dir', return_value=False):
            
            runner = CliRunner()
            result = runner.invoke(review, [str(special_file)])
            
            assert result.exit_code == 0  # Should exit with error
            assert "neither file nor directory" in result.output
    
    def test_cli_handles_directory_read_errors(self, tmp_path):
        """Test CLI handles directory processing errors."""
        test_dir = tmp_path / "test_docs"
        test_dir.mkdir()
        
        with patch('src.cli.main.ConversationManager') as mock_manager:
            mock_manager.return_value.run_review.side_effect = Exception("Test error")
            
            runner = CliRunner()
            result = runner.invoke(review, [str(test_dir)])
            
            assert result.exit_code == 0  # CLI handles error gracefully
            assert "Error during review" in result.output
