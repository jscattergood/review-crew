"""Tests for multi-document compilation functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.agents.conversation_manager import ConversationManager


class TestDocumentCompilation:
    """Test multi-document collection and compilation."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ConversationManager for testing."""
        with patch('src.agents.conversation_manager.PersonaLoader') as mock_loader:
            # Mock the PersonaLoader to prevent it from loading actual personas
            mock_loader.return_value.load_reviewer_personas.return_value = []
            mock_loader.return_value.load_analyzer_personas.return_value = []
            mock_loader.return_value.load_contextualizer_personas.return_value = []
            
            manager = ConversationManager()
            return manager
    
    def test_collect_documents_from_directory(self, tmp_path, mock_manager):
        """Test collecting documents from a directory."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        # Create test files with different extensions
        (test_dir / "doc1.txt").write_text("Text document content")
        (test_dir / "doc2.md").write_text("# Markdown document\nContent here")
        (test_dir / "doc3.py").write_text("# Python code\nprint('hello')")
        (test_dir / "doc4.json").write_text('{"key": "value"}')
        (test_dir / "ignored.exe").write_text("Binary file")  # Should be ignored
        
        manager = mock_manager
        documents = manager._collect_documents_from_directory(test_dir)
        
        # Should collect text files but ignore binary
        assert len(documents) == 4
        
        # Check document structure
        doc_names = [doc['name'] for doc in documents]
        assert "doc1.txt" in doc_names
        assert "doc2.md" in doc_names  
        assert "doc3.py" in doc_names
        assert "doc4.json" in doc_names
        assert "ignored.exe" not in doc_names
        
        # Check content is loaded
        txt_doc = next(doc for doc in documents if doc['name'] == 'doc1.txt')
        assert txt_doc['content'] == "Text document content"
    
    def test_collect_documents_empty_directory(self, tmp_path, mock_manager):
        """Test collecting documents from an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        manager = mock_manager
        documents = manager._collect_documents_from_directory(empty_dir)
        
        assert documents == []
    
    def test_collect_documents_with_read_errors(self, tmp_path, mock_manager):
        """Test collecting documents handles read errors gracefully."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        (test_dir / "good.txt").write_text("Good content")
        (test_dir / "bad.txt").write_text("Bad content")
        
        manager = mock_manager
        
        # Mock file read to simulate error for one file
        original_open = open
        def mock_open(file, *args, **kwargs):
            if "bad.txt" in str(file):
                raise PermissionError("Cannot read file")
            return original_open(file, *args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open):
            documents = manager._collect_documents_from_directory(test_dir)
        
        # Should only collect the good file
        assert len(documents) == 1
        assert documents[0]['name'] == 'good.txt'
        assert documents[0]['content'] == 'Good content'
    
    def test_compile_documents_for_review(self, mock_manager):
        """Test compiling multiple documents into review format."""
        documents = [
            {"name": "doc1.md", "content": "# First Document\nContent of first doc"},
            {"name": "doc2.txt", "content": "Second document content"},
            {"name": "doc3.py", "content": "def hello():\n    print('world')"}
        ]
        
        manager = mock_manager
        compiled = manager._compile_documents_for_review(documents)
        
        # Check format includes separators
        assert "=== Document: doc1.md ===" in compiled
        assert "=== Document: doc2.txt ===" in compiled
        assert "=== Document: doc3.py ===" in compiled
        
        # Check content is included
        assert "# First Document" in compiled
        assert "Second document content" in compiled
        assert "def hello():" in compiled
        
        # Check proper separation
        lines = compiled.split('\n')
        doc1_sep_idx = lines.index("=== Document: doc1.md ===")
        doc2_sep_idx = lines.index("=== Document: doc2.txt ===")
        assert doc2_sep_idx > doc1_sep_idx + 2  # Should have content and empty line between
    
    def test_compile_documents_empty_list(self, mock_manager):
        """Test compiling empty document list."""
        manager = mock_manager
        compiled = manager._compile_documents_for_review([])
        
        assert compiled == ""
    
    @patch('src.agents.conversation_manager.ConversationManager._count_tokens')
    @patch('src.agents.conversation_manager.ConversationManager._truncate_to_token_limit')
    def test_multi_document_review_chunking(self, mock_truncate, mock_count_tokens, tmp_path, mock_manager):
        """Test that multi-document review applies chunking when content is too large."""
        # Set up directory with test documents
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        (test_dir / "large_doc.md").write_text("Large document content" * 100)
        
        # Mock token counting to simulate large content
        mock_count_tokens.return_value = 5000  # Exceeds default 4096 limit
        mock_truncate.return_value = "Truncated content"
        
        manager = mock_manager
        
        # Mock the single document review to prevent actual model calls
        with patch.object(manager, '_run_single_document_review') as mock_single_review:
            mock_single_review.return_value = MagicMock()
            
            result = manager._run_multi_document_review(test_dir)
            
            # Verify chunking was applied
            mock_count_tokens.assert_called()
            mock_truncate.assert_called_once()
            
            # Check that truncate was called with compiled content and correct limit
            call_args = mock_truncate.call_args[0]
            assert len(call_args) == 2  # content and limit
            assert call_args[1] == 3596  # 4096 - 500 buffer
    
    def test_multi_document_review_no_chunking_needed(self, tmp_path, mock_manager):
        """Test that multi-document review skips chunking for small content."""
        # Set up directory with small documents
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        (test_dir / "small_doc.md").write_text("Small content")
        
        manager = mock_manager
        
        # Mock token counting to simulate small content
        with patch.object(manager, '_count_tokens', return_value=100):
            with patch.object(manager, '_truncate_to_token_limit') as mock_truncate:
                with patch.object(manager, '_run_single_document_review') as mock_single_review:
                    mock_single_review.return_value = MagicMock()
                    
                    result = manager._run_multi_document_review(test_dir)
                    
                    # Verify chunking was NOT applied
                    mock_truncate.assert_not_called()
    
    def test_multi_document_review_with_manifest(self, tmp_path, mock_manager):
        """Test multi-document review with manifest configuration."""
        # Set up directory with manifest
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        (test_dir / "doc.md").write_text("Test content")
        
        manifest_content = """
        review_configuration:
          name: "Test Review"
          reviewers: ["Content Reviewer"]
        """
        (test_dir / "manifest.yaml").write_text(manifest_content)
        
        manager = mock_manager
        
        # Mock the manifest review method
        with patch.object(manager, '_run_manifest_review') as mock_manifest_review:
            mock_manifest_review.return_value = MagicMock()
            
            result = manager._run_multi_document_review(test_dir)
            
            # Verify manifest review was called
            mock_manifest_review.assert_called_once()
    
    def test_multi_document_review_no_documents_error(self, tmp_path, mock_manager):
        """Test error handling when no documents found in directory."""
        # Set up empty directory
        test_dir = tmp_path / "empty"
        test_dir.mkdir()
        
        manager = mock_manager
        
        # Should raise ValueError for no documents
        with pytest.raises(ValueError, match="No readable documents found"):
            manager._run_multi_document_review(test_dir)