"""Tests for multi-document compilation functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.agents.conversation_manager import ConversationManager


class TestDocumentCompilation:
    """Test multi-document collection and compilation."""
    
    def test_collect_documents_from_directory(self, tmp_path):
        """Test collecting documents from a directory."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        # Create test files with different extensions
        (test_dir / "doc1.txt").write_text("Text document content")
        (test_dir / "doc2.md").write_text("# Markdown document\nContent here")
        (test_dir / "doc3.py").write_text("# Python code\nprint('hello')")
        (test_dir / "doc4.json").write_text('{"key": "value"}')
        (test_dir / "ignored.exe").write_text("Binary file")  # Should be ignored
        
        manager = ConversationManager()
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
    
    def test_collect_documents_empty_directory(self, tmp_path):
        """Test collecting documents from an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        manager = ConversationManager()
        documents = manager._collect_documents_from_directory(empty_dir)
        
        assert documents == []
    
    def test_collect_documents_with_read_errors(self, tmp_path):
        """Test collecting documents handles read errors gracefully."""
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        
        (test_dir / "good.txt").write_text("Good content")
        (test_dir / "bad.txt").write_text("Bad content")
        
        manager = ConversationManager()
        
        # Mock one file to raise an error
        original_open = open
        def mock_open(*args, **kwargs):
            if "bad.txt" in str(args[0]):
                raise PermissionError("Access denied")
            return original_open(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open):
            documents = manager._collect_documents_from_directory(test_dir)
        
        # Should collect the good file and skip the bad one
        assert len(documents) == 1
        assert documents[0]['name'] == "good.txt"
        assert documents[0]['content'] == "Good content"
    
    def test_compile_documents_for_review(self):
        """Test compiling multiple documents into review format."""
        documents = [
            {"name": "doc1.txt", "content": "First document content"},
            {"name": "doc2.md", "content": "Second document content"},
            {"name": "doc3.py", "content": "Third document content"}
        ]
        
        manager = ConversationManager()
        compiled = manager._compile_documents_for_review(documents)
        
        # Check format
        assert "=== Document: doc1.txt ===" in compiled
        assert "First document content" in compiled
        assert "=== Document: doc2.md ===" in compiled
        assert "Second document content" in compiled
        assert "=== Document: doc3.py ===" in compiled
        assert "Third document content" in compiled
        
        # Check order is preserved
        lines = compiled.split('\n')
        doc1_index = next(i for i, line in enumerate(lines) if "doc1.txt" in line)
        doc2_index = next(i for i, line in enumerate(lines) if "doc2.md" in line)
        doc3_index = next(i for i, line in enumerate(lines) if "doc3.py" in line)
        
        assert doc1_index < doc2_index < doc3_index
    
    def test_compile_documents_empty_list(self):
        """Test compiling an empty document list."""
        manager = ConversationManager()
        compiled = manager._compile_documents_for_review([])
        
        assert compiled == ""
    
    def test_multi_document_review_integration(self, tmp_path):
        """Test complete multi-document review workflow."""
        # Create test directory with documents
        test_dir = tmp_path / "application"
        test_dir.mkdir()
        
        (test_dir / "essay1.md").write_text("Personal statement content")
        (test_dir / "essay2.md").write_text("Supplemental essay content")
        (test_dir / "activities.txt").write_text("Activities list content")
        
        # Mock the single document review to avoid LLM calls
        with patch('src.agents.conversation_manager.ConversationManager._run_single_document_review') as mock_review:
            mock_result = Mock()
            mock_review.return_value = mock_result
            
            manager = ConversationManager()
            result = manager._run_multi_document_review(test_dir)
            
            # Should call single document review with compiled content
            mock_review.assert_called_once()
            compiled_content = mock_review.call_args[0][0]
            
            # Check compiled content contains all documents
            assert "=== Document: essay1.md ===" in compiled_content
            assert "Personal statement content" in compiled_content
            assert "=== Document: essay2.md ===" in compiled_content  
            assert "Supplemental essay content" in compiled_content
            assert "=== Document: activities.txt ===" in compiled_content
            assert "Activities list content" in compiled_content
            
            assert result == mock_result
    
    def test_multi_document_review_with_manifest(self, tmp_path):
        """Test multi-document review with manifest file."""
        # Create test directory with documents and manifest
        test_dir = tmp_path / "application" 
        test_dir.mkdir()
        
        (test_dir / "essay.md").write_text("Essay content")
        
        manifest_content = """
review_configuration:
  name: "Test Review"
  reviewers:
    - "Content Reviewer"
"""
        (test_dir / "manifest.yaml").write_text(manifest_content)
        
        # Mock the manifest review to avoid LLM calls
        with patch('src.agents.conversation_manager.ConversationManager._run_manifest_review') as mock_manifest_review:
            mock_result = Mock()
            mock_manifest_review.return_value = mock_result
            
            manager = ConversationManager()
            result = manager._run_multi_document_review(test_dir)
            
            # Should call manifest review instead of single document review
            mock_manifest_review.assert_called_once()
            
            # Check arguments
            args = mock_manifest_review.call_args[0]
            compiled_content = args[0]
            assert "=== Document: essay.md ===" in compiled_content
            assert "Essay content" in compiled_content
            
            # Should load and pass manifest config
            manifest_config = args[2]
            assert manifest_config is not None
            assert "review_configuration" in manifest_config
            
            assert result == mock_result
    
    def test_multi_document_review_no_documents(self, tmp_path):
        """Test multi-document review with no readable documents."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        manager = ConversationManager()
        
        with pytest.raises(ValueError, match="No readable documents found"):
            manager._run_multi_document_review(empty_dir)
    
    @patch('src.agents.conversation_manager.ConversationManager._count_tokens')
    @patch('src.agents.conversation_manager.ConversationManager._truncate_to_token_limit')
    def test_multi_document_review_chunking(self, mock_truncate, mock_count_tokens, tmp_path):
        """Test multi-document review with content chunking."""
        test_dir = tmp_path / "large_docs"
        test_dir.mkdir()
        
        (test_dir / "large.txt").write_text("Very large document content " * 100)
        
        # Mock token counting to trigger chunking
        mock_count_tokens.return_value = 5000  # Exceeds default 4096 limit
        mock_truncate.return_value = "Truncated content"
        
        with patch('src.agents.conversation_manager.ConversationManager._run_single_document_review') as mock_review:
            mock_result = Mock()
            mock_review.return_value = mock_result
            
            manager = ConversationManager()
            result = manager._run_multi_document_review(test_dir)
            
            # Should count tokens and truncate
            mock_count_tokens.assert_called_once()
            # Should call truncate with the actual compiled content, not the token count
            mock_truncate.assert_called_once()
            truncate_args = mock_truncate.call_args[0]
            assert truncate_args[1] == 3596  # 4096 - 500 token limit
            assert "=== Document: large.txt ===" in truncate_args[0]  # Should be compiled content
            
            # Should call review with truncated content
            mock_review.assert_called_once()
            assert result == mock_result
