"""
Tests for DocumentProcessorNode.

This module tests the document processing functionality extracted from ConversationManager
into a dedicated Strands Graph node.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agents.document_processor_node import DocumentProcessorNode, DocumentProcessorResult
from strands.multiagent.base import Status


class TestDocumentProcessorNode:
    """Test cases for DocumentProcessorNode."""
    
    @pytest.fixture
    def processor_node(self):
        """Create a DocumentProcessorNode instance for testing."""
        return DocumentProcessorNode()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_manifest(self):
        """Sample manifest configuration for testing."""
        return {
            "review_configuration": {
                "documents": {
                    "primary": "main_document.md",
                    "supporting": ["context.md", "background.txt"],
                    "context_files": [
                        {"path": "context/additional.md", "type": "background", "weight": "high"}
                    ],
                    "relationships": [
                        {"source": "main_document.md", "target": "context.md", "type": "references"}
                    ]
                },
                "review_focus": {
                    "primary_concerns": [
                        {"concern": "accuracy", "weight": "critical", "description": "Check facts"}
                    ]
                },
                "output": {
                    "format": "detailed",
                    "include_scores": True
                }
            }
        }
    
    def test_init(self, processor_node):
        """Test DocumentProcessorNode initialization."""
        assert processor_node.name == "document_processor"
        assert isinstance(processor_node, DocumentProcessorNode)
    
    def test_init_custom_name(self):
        """Test DocumentProcessorNode initialization with custom name."""
        node = DocumentProcessorNode("custom_processor")
        assert node.name == "custom_processor"
    
    @pytest.mark.asyncio
    async def test_process_direct_content(self, processor_node):
        """Test processing direct content string."""
        content = "This is test content for processing."
        
        result = await processor_node.invoke_async(content)
        
        assert result.status == Status.COMPLETED
        assert processor_node.name in result.results
        
        node_result = result.results[processor_node.name]
        # Metadata is now stored in the agent result's state
        metadata = node_result.result.state["document_processor_result"]
        
        assert isinstance(metadata, DocumentProcessorResult)
        assert metadata.document_type == "single"
        assert metadata.compiled_content == content
        assert len(metadata.documents) == 1
        assert metadata.documents[0]["name"] == "direct_content"
        assert metadata.documents[0]["content"] == content
    
    @pytest.mark.asyncio
    async def test_process_single_file(self, processor_node, temp_dir):
        """Test processing a single file."""
        # Create test file
        test_file = temp_dir / "test_document.md"
        test_content = "# Test Document\n\nThis is a test document."
        test_file.write_text(test_content, encoding="utf-8")
        
        result = await processor_node.invoke_async(str(test_file))
        
        assert result.status == Status.COMPLETED
        metadata = result.results[processor_node.name].result.state["document_processor_result"]
        
        assert metadata.document_type == "single"
        assert metadata.compiled_content == test_content
        assert len(metadata.documents) == 1
        assert metadata.documents[0]["name"] == "test_document.md"
        assert metadata.documents[0]["content"] == test_content
        assert metadata.original_path == str(test_file)
    
    @pytest.mark.asyncio
    async def test_process_single_file_not_found(self, processor_node):
        """Test processing a non-existent file."""
        result = await processor_node.invoke_async("/nonexistent/file.txt")
        
        assert result.status == Status.FAILED
        assert "Document processing failed" in result.results[processor_node.name].result.message["content"][0]["text"]
    
    @pytest.mark.asyncio
    async def test_process_multi_document_directory(self, processor_node, temp_dir):
        """Test processing multiple documents from directory."""
        # Create test files
        (temp_dir / "doc1.md").write_text("# Document 1", encoding="utf-8")
        (temp_dir / "doc2.txt").write_text("Document 2 content", encoding="utf-8")
        (temp_dir / "doc3.py").write_text("# Python file\nprint('hello')", encoding="utf-8")
        (temp_dir / "ignored.bin").write_bytes(b"binary content")  # Should be ignored
        
        result = await processor_node.invoke_async(str(temp_dir))
        
        assert result.status == Status.COMPLETED
        metadata = result.results[processor_node.name].result.state["document_processor_result"]
        
        assert metadata.document_type == "multi"
        assert len(metadata.documents) == 3  # Binary file should be ignored
        assert metadata.original_path == str(temp_dir)
        
        # Check compiled content includes all documents
        assert "=== Document: doc1.md ===" in metadata.compiled_content
        assert "=== Document: doc2.txt ===" in metadata.compiled_content
        assert "=== Document: doc3.py ===" in metadata.compiled_content
    
    @pytest.mark.asyncio
    async def test_process_multi_document_with_manifest(self, processor_node, temp_dir, sample_manifest):
        """Test processing documents with manifest configuration."""
        # Create manifest file
        manifest_path = temp_dir / "manifest.yaml"
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(sample_manifest, f)
        
        # Create document files
        (temp_dir / "main_document.md").write_text("# Main Document", encoding="utf-8")
        (temp_dir / "context.md").write_text("Context information", encoding="utf-8")
        (temp_dir / "background.txt").write_text("Background details", encoding="utf-8")
        
        # Create context directory and file
        context_dir = temp_dir / "context"
        context_dir.mkdir()
        (context_dir / "additional.md").write_text("Additional context", encoding="utf-8")
        
        result = await processor_node.invoke_async(str(temp_dir))
        
        assert result.status == Status.COMPLETED
        metadata = result.results[processor_node.name].result.state["document_processor_result"]
        
        assert metadata.document_type == "multi"
        assert metadata.manifest_config is not None
        assert metadata.enhanced_manifest is not None
        
        # Check document types are properly set
        primary_docs = [d for d in metadata.documents if d.get("type") == "primary"]
        supporting_docs = [d for d in metadata.documents if d.get("type") == "supporting"]
        
        assert len(primary_docs) == 1
        assert len(supporting_docs) == 2
        assert primary_docs[0]["name"] == "main_document.md"
        
        # Check compiled content has proper headers
        assert "=== PRIMARY DOCUMENT: main_document.md ===" in metadata.compiled_content
        assert "=== SUPPORTING DOCUMENT: context.md ===" in metadata.compiled_content
    
    @pytest.mark.asyncio
    async def test_process_empty_directory(self, processor_node, temp_dir):
        """Test processing empty directory raises error."""
        result = await processor_node.invoke_async(str(temp_dir))
        
        assert result.status == Status.FAILED
        assert "No readable documents found" in result.results[processor_node.name].result.state["error"]
    
    def test_collect_documents_from_directory(self, processor_node, temp_dir):
        """Test collecting documents from directory."""
        # Create various file types
        (temp_dir / "doc.md").write_text("Markdown", encoding="utf-8")
        (temp_dir / "script.py").write_text("Python", encoding="utf-8")
        (temp_dir / "data.json").write_text('{"key": "value"}', encoding="utf-8")
        (temp_dir / "binary.exe").write_bytes(b"binary")  # Should be ignored
        (temp_dir / "no_extension").write_text("No extension", encoding="utf-8")  # Should be ignored
        
        documents = processor_node._collect_documents_from_directory(temp_dir)
        
        assert len(documents) == 3
        doc_names = [doc["name"] for doc in documents]
        assert "doc.md" in doc_names
        assert "script.py" in doc_names
        assert "data.json" in doc_names
        assert "binary.exe" not in doc_names
        assert "no_extension" not in doc_names
    
    def test_collect_documents_from_manifest(self, processor_node, temp_dir, sample_manifest):
        """Test collecting documents based on manifest configuration."""
        # Create document files
        (temp_dir / "main_document.md").write_text("Main content", encoding="utf-8")
        (temp_dir / "context.md").write_text("Context content", encoding="utf-8")
        (temp_dir / "background.txt").write_text("Background content", encoding="utf-8")
        
        documents = processor_node._collect_documents_from_manifest(sample_manifest, temp_dir)
        
        assert len(documents) == 3
        
        # Check primary document
        primary_docs = [d for d in documents if d.get("type") == "primary"]
        assert len(primary_docs) == 1
        assert primary_docs[0]["name"] == "main_document.md"
        assert primary_docs[0]["manifest_path"] == "main_document.md"
        
        # Check supporting documents
        supporting_docs = [d for d in documents if d.get("type") == "supporting"]
        assert len(supporting_docs) == 2
        supporting_names = [d["name"] for d in supporting_docs]
        assert "context.md" in supporting_names
        assert "background.txt" in supporting_names
    
    def test_collect_documents_from_manifest_missing_files(self, processor_node, temp_dir, sample_manifest):
        """Test collecting documents when some files are missing."""
        # Only create one of the required files
        (temp_dir / "main_document.md").write_text("Main content", encoding="utf-8")
        
        documents = processor_node._collect_documents_from_manifest(sample_manifest, temp_dir)
        
        # Should only get the existing file
        assert len(documents) == 1
        assert documents[0]["name"] == "main_document.md"
        assert documents[0]["type"] == "primary"
    
    def test_resolve_document_path(self, processor_node, temp_dir):
        """Test resolving document paths."""
        # Test relative path
        relative_path = processor_node._resolve_document_path("subdir/file.txt", temp_dir)
        assert relative_path == temp_dir / "subdir/file.txt"
        
        # Test absolute path
        absolute_path = processor_node._resolve_document_path("/absolute/path.txt", temp_dir)
        assert absolute_path == Path("/absolute/path.txt")
        
        # Test parent directory path - resolve both paths to handle symlinks
        parent_path = processor_node._resolve_document_path("../parent.txt", temp_dir)
        expected_path = temp_dir.parent / "parent.txt"
        assert parent_path.resolve() == expected_path.resolve()
    
    def test_compile_documents_for_review(self, processor_node):
        """Test compiling documents into review content."""
        documents = [
            {"name": "primary.md", "content": "Primary content", "type": "primary", "manifest_path": "primary.md"},
            {"name": "support.txt", "content": "Supporting content", "type": "supporting", "manifest_path": "support.txt"},
            {"name": "other.py", "content": "Other content"}
        ]
        
        compiled = processor_node._compile_documents_for_review(documents)
        
        # Check structure and order
        assert "=== PRIMARY DOCUMENT: primary.md ===" in compiled
        assert "=== SUPPORTING DOCUMENT: support.txt ===" in compiled
        assert "=== Document: other.py ===" in compiled
        
        # Check content is included
        assert "Primary content" in compiled
        assert "Supporting content" in compiled
        assert "Other content" in compiled
        
        # Check primary comes before supporting
        primary_pos = compiled.find("PRIMARY DOCUMENT")
        supporting_pos = compiled.find("SUPPORTING DOCUMENT")
        assert primary_pos < supporting_pos
    
    def test_load_manifest(self, processor_node, temp_dir):
        """Test loading manifest file."""
        manifest_data = {"test": "data", "nested": {"key": "value"}}
        manifest_path = temp_dir / "manifest.yaml"
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest_data, f)
        
        loaded = processor_node._load_manifest(manifest_path)
        assert loaded == manifest_data
    
    def test_load_manifest_invalid_file(self, processor_node, temp_dir):
        """Test loading invalid manifest file."""
        manifest_path = temp_dir / "invalid.yaml"
        manifest_path.write_text("invalid: yaml: content: [", encoding="utf-8")
        
        loaded = processor_node._load_manifest(manifest_path)
        assert loaded == {}  # Should return empty dict on error
    
    def test_process_advanced_manifest(self, processor_node, temp_dir, sample_manifest):
        """Test processing advanced manifest features."""
        # Create context file
        context_dir = temp_dir / "context"
        context_dir.mkdir()
        (context_dir / "additional.md").write_text("Additional context", encoding="utf-8")
        
        enhanced = processor_node._process_advanced_manifest(sample_manifest, temp_dir)
        
        review_config = enhanced["review_configuration"]
        
        # Check processed context files
        assert "processed_context" in review_config
        context_files = review_config["processed_context"]
        assert len(context_files) == 1
        assert context_files[0]["loaded"] is True
        assert context_files[0]["content"] == "Additional context"
        
        # Check processed relationships
        assert "processed_relationships" in review_config
        relationships = review_config["processed_relationships"]
        assert len(relationships) == 1
        assert relationships[0]["source"] == "main_document.md"
        assert relationships[0]["target"] == "context.md"
        
        # Check processed focus
        assert "processed_focus" in review_config
        focus = review_config["processed_focus"]
        assert len(focus["focus_instructions"]) == 1
        assert "CRITICAL" in focus["focus_instructions"][0]
        
        # Check processed output
        assert "processed_output" in review_config
        output = review_config["processed_output"]
        assert output["format"] == "detailed"
        assert output["include_scores"] is True
    
    def test_process_context_files(self, processor_node, temp_dir):
        """Test processing context files from manifest."""
        # Create context file
        context_dir = temp_dir / "context"
        context_dir.mkdir()
        context_file = context_dir / "test.md"
        context_file.write_text("Test context content", encoding="utf-8")
        
        review_config = {
            "documents": {
                "context_files": [
                    {"path": "context/test.md", "type": "background", "weight": "high"}
                ]
            }
        }
        
        context_files = processor_node._process_context_files(review_config, temp_dir)
        
        assert len(context_files) == 1
        assert context_files[0]["path"] == "context/test.md"
        assert context_files[0]["type"] == "background"
        assert context_files[0]["weight"] == "high"
        assert context_files[0]["loaded"] is True
        assert context_files[0]["content"] == "Test context content"
    
    def test_process_context_files_missing(self, processor_node, temp_dir):
        """Test processing missing context files."""
        review_config = {
            "documents": {
                "context_files": [
                    {"path": "missing/file.md", "type": "background", "weight": "high"}
                ]
            }
        }
        
        context_files = processor_node._process_context_files(review_config, temp_dir)
        
        assert len(context_files) == 0  # Missing files are not added to the list
    
    def test_process_document_relationships(self, processor_node):
        """Test processing document relationships."""
        review_config = {
            "documents": {
                "relationships": [
                    {"source": "doc1.md", "target": "doc2.md", "type": "references", "note": "Test note"},
                    {"source": "doc2.md", "target": "doc3.md"}  # Minimal relationship
                ]
            }
        }
        
        relationships = processor_node._process_document_relationships(review_config)
        
        assert len(relationships) == 2
        
        # Check first relationship
        assert relationships[0]["source"] == "doc1.md"
        assert relationships[0]["target"] == "doc2.md"
        assert relationships[0]["type"] == "references"
        assert relationships[0]["note"] == "Test note"
        
        # Check second relationship with defaults
        assert relationships[1]["source"] == "doc2.md"
        assert relationships[1]["target"] == "doc3.md"
        assert relationships[1]["type"] == "relates_to"  # Default
        assert relationships[1]["note"] == ""  # Default
        assert relationships[1]["weight"] == "medium"  # Default
    
    def test_process_review_focus(self, processor_node):
        """Test processing review focus configuration."""
        review_config = {
            "review_focus": {
                "primary_concerns": [
                    {"concern": "accuracy", "weight": "critical", "description": "Check facts"},
                    {"concern": "clarity", "weight": "high"}
                ],
                "secondary_concerns": [
                    {"concern": "style", "weight": "medium", "description": "Writing style"}
                ]
            }
        }
        
        focus = processor_node._process_review_focus(review_config)
        
        assert len(focus["primary_concerns"]) == 2
        assert len(focus["secondary_concerns"]) == 1
        assert len(focus["focus_instructions"]) == 3
        
        # Check instruction formatting
        instructions = focus["focus_instructions"]
        assert "🔴 CRITICAL: Pay special attention to accuracy - Check facts" in instructions
        assert "🟡 HIGH PRIORITY: Focus on clarity" in instructions
        assert "🔵 CONSIDER: style - Writing style" in instructions
    
    def test_process_output_configuration(self, processor_node):
        """Test processing output configuration."""
        review_config = {
            "output": {
                "format": "detailed",
                "include_sections": ["summary", "details"],
                "exclude_sections": ["raw_data"],
                "summary_length": "long",
                "include_scores": True,
                "highlight_critical_issues": True
            }
        }
        
        output = processor_node._process_output_configuration(review_config)
        
        assert output["format"] == "detailed"
        assert output["include_sections"] == ["summary", "details"]
        assert output["exclude_sections"] == ["raw_data"]
        assert output["summary_length"] == "long"
        assert output["include_scores"] is True
        assert output["highlight_critical_issues"] is True
    
    @patch('src.validation.document_validator.DocumentValidator')
    def test_validate_document_collection_success(self, mock_validator_class, processor_node, temp_dir):
        """Test successful document validation."""
        # Mock validator
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        mock_validator.validate_document_collection.return_value = {"test_results": "success"}
        
        result = processor_node._validate_document_collection(temp_dir)
        
        assert result == {"test_results": "success"}
        mock_validator_class.assert_called_once()
        mock_validator.validate_document_collection.assert_called_once_with(temp_dir, None)
    
    def test_validate_document_collection_import_error(self, processor_node, temp_dir):
        """Test document validation with import error."""
        # Mock the import to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = processor_node._validate_document_collection(temp_dir)
            assert result is None
    
    def test_report_validation_results(self, processor_node, capsys):
        """Test reporting validation results."""
        # Mock validation results
        mock_result = Mock()
        mock_result.level.value = "error"
        mock_result.message = "Test error message"
        
        mock_warning = Mock()
        mock_warning.level.value = "warning"
        mock_warning.message = "Test warning message"
        
        validation_results = {
            "test_file.md": ([mock_result, mock_warning], {}),
            "_metadata": ([], {})  # Should be ignored
        }
        
        processor_node._report_validation_results(validation_results)
        
        captured = capsys.readouterr()
        assert "❌ Document validation found 1 errors" in captured.out
        assert "⚠️  Document validation found 1 warnings" in captured.out
        assert "📄 test_file.md:" in captured.out
        assert "❌ Test error message" in captured.out
        assert "⚠️  Test warning message" in captured.out


class TestDocumentProcessorResult:
    """Test cases for DocumentProcessorResult dataclass."""
    
    def test_document_processor_result_creation(self):
        """Test creating DocumentProcessorResult."""
        documents = [{"name": "test.md", "content": "test content"}]
        
        result = DocumentProcessorResult(
            documents=documents,
            document_type="single",
            compiled_content="test content"
        )
        
        assert result.documents == documents
        assert result.document_type == "single"
        assert result.compiled_content == "test content"
        assert result.manifest_config is None
        assert result.validation_results is None
        assert result.enhanced_manifest is None
        assert result.original_path is None
    
    def test_document_processor_result_with_all_fields(self):
        """Test creating DocumentProcessorResult with all fields."""
        documents = [{"name": "test.md", "content": "test content"}]
        manifest_config = {"test": "config"}
        validation_results = {"test": "validation"}
        enhanced_manifest = {"enhanced": "manifest"}
        
        result = DocumentProcessorResult(
            documents=documents,
            document_type="multi",
            compiled_content="compiled content",
            manifest_config=manifest_config,
            validation_results=validation_results,
            enhanced_manifest=enhanced_manifest,
            original_path="/test/path"
        )
        
        assert result.documents == documents
        assert result.document_type == "multi"
        assert result.compiled_content == "compiled content"
        assert result.manifest_config == manifest_config
        assert result.validation_results == validation_results
        assert result.enhanced_manifest == enhanced_manifest
        assert result.original_path == "/test/path"
