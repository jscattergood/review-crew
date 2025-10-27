"""
Tests for DocumentProcessorNode, focusing on manifest schema validation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.conversation.document_processor import DocumentProcessorNode


class TestDocumentProcessorManifestValidation:
    """Test manifest schema validation functionality."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessorNode instance."""
        return DocumentProcessorNode()

    @pytest.fixture
    def valid_manifest(self):
        """Create a valid manifest dictionary."""
        return {
            "review_configuration": {
                "name": "Test Review",
                "description": "Test description",
                "version": "1.0",
                "reviewers": ["Test Reviewer"],
                "documents": {"primary": "test.md"},
            }
        }

    @pytest.fixture
    def invalid_manifest(self):
        """Create an invalid manifest dictionary (missing required fields)."""
        return {
            "review_configuration": {
                "name": "Test Review",
                # Missing required 'description' and 'version'
                "reviewers": ["Test Reviewer"],
            }
        }

    @pytest.fixture
    def schema(self):
        """Create a minimal JSON schema for testing."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["review_configuration"],
            "properties": {
                "review_configuration": {
                    "type": "object",
                    "required": ["name", "description", "version"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "version": {"type": "string"},
                        "reviewers": {"type": "array", "items": {"type": "string"}},
                        "documents": {
                            "type": "object",
                            "properties": {"primary": {"type": "string"}},
                        },
                    },
                }
            },
        }

    def test_validate_manifest_schema_with_valid_manifest(
        self, processor, valid_manifest, schema, capsys
    ):
        """Test schema validation with a valid manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create schema file
            schema_path = temp_path / "manifest.schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f)

            # Create manifest path (3 levels deep to match expected structure)
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            with patch("jsonschema.validate") as mock_validate:
                processor._validate_manifest_schema(valid_manifest, manifest_path)

                # Should call jsonschema.validate with manifest and schema
                mock_validate.assert_called_once()

            captured = capsys.readouterr()
            assert "✅ Manifest schema validation passed: manifest.yaml" in captured.out

    def test_validate_manifest_schema_with_invalid_manifest(
        self, processor, invalid_manifest, schema, capsys
    ):
        """Test schema validation with an invalid manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create schema file
            schema_path = temp_path / "manifest.schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f)

            # Create manifest path (3 levels deep to match expected structure)
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            # This should not raise an exception, just print warnings
            processor._validate_manifest_schema(invalid_manifest, manifest_path)

            captured = capsys.readouterr()
            assert (
                "❌ Manifest schema validation failed for manifest.yaml:"
                in captured.out
            )

    def test_validate_manifest_schema_no_schema_file(
        self, processor, valid_manifest, capsys
    ):
        """Test schema validation when schema file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            # No schema file created - should skip validation silently
            processor._validate_manifest_schema(valid_manifest, manifest_path)

            captured = capsys.readouterr()
            # Should not print anything when schema file is missing
            assert "validation" not in captured.out.lower()

    def test_validate_manifest_schema_no_jsonschema_module(
        self, processor, valid_manifest, capsys
    ):
        """Test schema validation when jsonschema module is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'jsonschema'"),
            ):
                processor._validate_manifest_schema(valid_manifest, manifest_path)

            captured = capsys.readouterr()
            # Should not print anything when jsonschema is not available
            assert "validation" not in captured.out.lower()

    def test_validate_manifest_schema_json_error(
        self, processor, valid_manifest, capsys
    ):
        """Test schema validation when schema file has invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid JSON schema file
            schema_path = temp_path / "manifest.schema.json"
            with open(schema_path, "w") as f:
                f.write("{ invalid json }")

            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            processor._validate_manifest_schema(valid_manifest, manifest_path)

            captured = capsys.readouterr()
            assert "⚠️  Schema validation error for manifest.yaml:" in captured.out

    def test_load_manifest_with_schema_validation(
        self, processor, valid_manifest, schema
    ):
        """Test that _load_manifest calls schema validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create schema file
            schema_path = temp_path / "manifest.schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f)

            # Create manifest file
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                yaml.dump(valid_manifest, f)

            with patch.object(processor, "_validate_manifest_schema") as mock_validate:
                result = processor._load_manifest(manifest_path)

                # Should call validation with loaded manifest
                mock_validate.assert_called_once_with(valid_manifest, manifest_path)
                assert result == valid_manifest

    def test_load_manifest_yaml_error(self, processor, capsys):
        """Test _load_manifest with invalid YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML file
            manifest_path = temp_path / "manifest.yaml"
            with open(manifest_path, "w") as f:
                f.write("invalid: yaml: content: [")

            result = processor._load_manifest(manifest_path)

            # Should return empty dict and print warning
            assert result == {}
            captured = capsys.readouterr()
            assert "⚠️  Warning: Failed to parse manifest" in captured.out

    def test_load_manifest_file_not_found(self, processor, capsys):
        """Test _load_manifest with non-existent file."""
        non_existent_path = Path("/non/existent/manifest.yaml")

        result = processor._load_manifest(non_existent_path)

        # Should return empty dict and print warning
        assert result == {}
        captured = capsys.readouterr()
        assert "⚠️  Warning: Failed to parse manifest" in captured.out


class TestDocumentProcessorIntegration:
    """Integration tests for document processor with schema validation."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessorNode instance."""
        return DocumentProcessorNode()

    def test_real_manifest_validation(self, processor):
        """Test validation with a real manifest file from the project."""
        # Use the essay_comparison manifest we created
        manifest_path = Path(
            "/Users/scattej/Documents/GitHub/personal/review-crew/input/essay_comparison/manifest.yaml"
        )

        if manifest_path.exists():
            # This should work without errors since we validated it earlier
            result = processor._load_manifest(manifest_path)

            # Should return a non-empty dict
            assert result
            assert "review_configuration" in result
            assert result["review_configuration"]["name"] == "Essay Comparison Review"

    def test_schema_path_resolution(self, processor):
        """Test that schema path is resolved correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directory structure: temp_dir/input/test/manifest.yaml
            manifest_path = temp_path / "input" / "test" / "manifest.yaml"

            # Expected schema path should be: temp_dir/manifest.schema.json
            expected_schema_path = temp_path / "manifest.schema.json"

            # Create the schema file
            schema = {"type": "object"}
            with open(expected_schema_path, "w") as f:
                json.dump(schema, f)

            manifest = {
                "review_configuration": {
                    "name": "test",
                    "description": "test",
                    "version": "1.0",
                }
            }

            with patch("jsonschema.validate") as mock_validate:
                processor._validate_manifest_schema(manifest, manifest_path)

                # Should have found and loaded the schema
                mock_validate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
