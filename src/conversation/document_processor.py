"""
Document Processor Node for Strands Graph Architecture.

This module provides a custom Strands node that handles all document processing
operations including loading, manifest processing, and validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands.agent.agent_result import AgentResult
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import ContentBlock, Message

from ..validation.document_validator import ValidationLevel


@dataclass
class DocumentProcessorResult:
    """Result from document processing operations."""

    documents: list[dict[str, str]]
    document_type: str  # "single" or "multi"
    compiled_content: str
    manifest_config: dict[str, Any] | None = None
    validation_results: dict[str, Any] | None = None
    enhanced_manifest: dict[str, Any] | None = None
    original_path: str | None = None


class DocumentProcessorNode(MultiAgentBase):
    """Custom Strands node for document processing operations.

    This node handles:
    - Document loading (single vs multi-document)
    - Manifest processing and advanced features
    - Document validation pipeline
    - Document compilation for review
    """

    def __init__(self, name: str = "document_processor"):
        """Initialize the document processor node.

        Args:
            name: Name of the node
        """
        super().__init__()
        self.name = name

    def __call__(
        self, task: str | list[ContentBlock], **kwargs: Any
    ) -> MultiAgentResult:
        """Process documents synchronously.

        Args:
            task: Input content (string path or content)
            **kwargs: Additional arguments

        Returns:
            MultiAgentResult with document processing results
        """
        # Run the async method synchronously
        import asyncio

        print(f"Call dump: {json.dumps(task, indent=2)}")
        return asyncio.run(self.invoke_async(task, **kwargs))

    async def invoke_async(
        self, task: str | list[ContentBlock], **kwargs: Any
    ) -> MultiAgentResult:
        """Process documents asynchronously.

        Args:
            task: Input content (string path or content)
            **kwargs: Additional arguments

        Returns:
            MultiAgentResult with document processing results
        """
        try:
            # Determine if input is a path or content
            if isinstance(task, str):
                task_str = task
            elif isinstance(task, list):
                task_str = task[0].get("text", "")

            content_path = Path(task_str)

            # Check if this looks like a file path (absolute or has path separators)
            looks_like_path = (
                task_str.startswith("/")  # Absolute path
                or task_str.startswith("\\")  # Windows absolute path
                or "/" in task_str  # Contains path separators
                or "\\" in task_str  # Windows path separators
                or task_str.startswith("./")  # Relative path
                or task_str.startswith("../")  # Parent directory path
            )

            if content_path.exists() and content_path.is_dir():
                # Multi-document processing
                result = await self._process_multi_document(content_path)
            elif content_path.exists() and content_path.is_file():
                # Single file processing
                result = await self._process_single_file(content_path)
            elif looks_like_path:
                # Looks like a path but doesn't exist - this is an error
                raise ValueError(f"Path does not exist: {task_str}")
            else:
                # Direct content processing
                result = await self._process_direct_content(task_str)

            # Create agent result with the compiled content in the message
            # This is what the downstream agents will receive as input
            agent_result = AgentResult(
                stop_reason="end_turn",
                message=Message(
                    role="assistant",
                    content=[ContentBlock(text=result.compiled_content)],
                ),
                metrics=EventLoopMetrics(),
                state={
                    "document_processor_result": result,  # Store metadata for debugging
                    "document_type": result.document_type,
                    "document_count": len(result.documents),
                    "manifest_config": result.manifest_config,
                    "enhanced_manifest": result.enhanced_manifest,
                },
            )

            # Return wrapped in MultiAgentResult
            return MultiAgentResult(
                status=Status.COMPLETED,
                results={
                    self.name: NodeResult(result=agent_result, status=Status.COMPLETED)
                },
            )

        except Exception as e:
            # Handle errors gracefully - return a clear error marker
            agent_result = AgentResult(
                stop_reason="end_turn",
                message=Message(
                    role="assistant",
                    content=[ContentBlock(text="ERROR_NO_CONTENT")],
                ),
                metrics=EventLoopMetrics(),
                state={"error": str(e)},
            )

            return MultiAgentResult(
                status=Status.FAILED,
                results={
                    self.name: NodeResult(result=agent_result, status=Status.FAILED)
                },
            )

    async def _process_multi_document(
        self, directory_path: Path
    ) -> DocumentProcessorResult:
        """Process multiple documents from a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            DocumentProcessorResult with processed documents
        """
        print(f"📂 Processing document collection from: {directory_path}")

        # Check for manifest file
        manifest_path = directory_path / "manifest.yaml"
        documents = []
        manifest_config = None
        enhanced_manifest = None

        if manifest_path.exists():
            print("📋 Found manifest file, using manifest-driven document loading")
            manifest_config = self._load_manifest(manifest_path)

            # Load documents according to manifest
            documents = self._collect_documents_from_manifest(
                manifest_config, directory_path
            )

            if not documents:
                print(
                    "⚠️  No documents specified in manifest, falling back to directory scan"
                )
                documents = self._collect_documents_from_directory(directory_path)

            # Process advanced manifest features
            enhanced_manifest = self._process_advanced_manifest(
                manifest_config, directory_path
            )
        else:
            print("📁 No manifest found, scanning directory for documents")
            documents = self._collect_documents_from_directory(directory_path)

        if not documents:
            raise ValueError(f"No readable documents found in {directory_path}")

        print(f"📄 Found {len(documents)} documents to process")

        # Compile documents for review
        compiled_content = self._compile_documents_for_review(documents)

        # Run validation if available
        enhanced_manifest_for_validation = None
        manifest_path = directory_path / "manifest.yaml"
        if manifest_path.exists():
            if "enhanced_manifest" in locals():
                # Use enhanced manifest which includes original config plus processed data
                enhanced_manifest_for_validation = enhanced_manifest
            else:
                # Fallback: create enhanced manifest with just the original config
                manifest_config = self._load_manifest(manifest_path)
                enhanced_manifest_for_validation = {
                    "original_manifest": manifest_config
                }

        validation_results = self._validate_loaded_documents(
            documents, enhanced_manifest_for_validation
        )
        if validation_results:
            self._report_validation_results(validation_results)

        return DocumentProcessorResult(
            documents=documents,
            document_type="multi",
            compiled_content=compiled_content,
            manifest_config=manifest_config,
            validation_results=validation_results,
            enhanced_manifest=enhanced_manifest,
            original_path=str(directory_path),
        )

    async def _process_single_file(self, file_path: Path) -> DocumentProcessorResult:
        """Process a single file.

        Args:
            file_path: Path to the file

        Returns:
            DocumentProcessorResult with processed file
        """
        print(f"📄 Processing single file: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            documents = [{"name": file_path.name, "content": content}]

            return DocumentProcessorResult(
                documents=documents,
                document_type="single",
                compiled_content=content,
                original_path=str(file_path),
            )

        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}") from e

    async def _process_direct_content(self, content: str) -> DocumentProcessorResult:
        """Process direct content string.

        Args:
            content: Content string to process

        Returns:
            DocumentProcessorResult with processed content
        """
        print(f"📝 Processing direct content ({len(content)} characters)")

        documents = [{"name": "direct_content", "content": content}]

        return DocumentProcessorResult(
            documents=documents, document_type="single", compiled_content=content
        )

    def _collect_documents_from_directory(
        self, directory_path: Path
    ) -> list[dict[str, str]]:
        """Collect all readable documents from a directory.

        Args:
            directory_path: Path to directory

        Returns:
            List of document dictionaries with 'name' and 'content' keys
        """
        documents = []

        # Common text file extensions to process
        text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".yaml",
            ".yml",
            ".json",
            ".xml",
        }

        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                    documents.append({"name": file_path.name, "content": content})
                    print(f"  ✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Skipped {file_path.name}: {e}")
                    continue

        return documents

    def _collect_documents_from_manifest(
        self, manifest_config: dict[str, Any], directory_path: Path
    ) -> list[dict[str, str]]:
        """Collect documents specified in manifest configuration.

        Args:
            manifest_config: Parsed manifest configuration
            directory_path: Base directory path for resolving relative paths

        Returns:
            List of document dictionaries with 'name', 'content', and 'type' keys
        """
        documents: list[dict[str, str]] = []

        review_config = manifest_config.get("review_configuration", {})
        document_config = review_config.get("documents", {})

        if not document_config:
            print("  ℹ️  No documents section found in manifest")
            return documents

        # Load primary document
        primary_doc = document_config.get("primary")
        if primary_doc:
            primary_path = self._resolve_document_path(primary_doc, directory_path)
            if primary_path and primary_path.exists():
                try:
                    with open(primary_path, encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        {
                            "name": primary_path.name,
                            "content": content,
                            "type": "primary",
                            "manifest_path": primary_doc,
                        }
                    )
                    print(f"  ✓ Loaded primary document: {primary_doc}")
                except Exception as e:
                    print(f"  ⚠️  Failed to load primary document {primary_doc}: {e}")
            else:
                print(f"  ⚠️  Primary document not found: {primary_doc}")

        # Load supporting documents
        supporting_docs = document_config.get("supporting", [])
        for supporting_doc in supporting_docs:
            supporting_path = self._resolve_document_path(
                supporting_doc, directory_path
            )
            if supporting_path and supporting_path.exists():
                try:
                    with open(supporting_path, encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        {
                            "name": supporting_path.name,
                            "content": content,
                            "type": "supporting",
                            "manifest_path": supporting_doc,
                        }
                    )
                    print(f"  ✓ Loaded supporting document: {supporting_doc}")
                except Exception as e:
                    print(
                        f"  ⚠️  Failed to load supporting document {supporting_doc}: {e}"
                    )
            else:
                print(f"  ⚠️  Supporting document not found: {supporting_doc}")

        if documents:
            primary_count = len([d for d in documents if d["type"] == "primary"])
            supporting_count = len([d for d in documents if d["type"] == "supporting"])
            print(
                f"  📋 Manifest specified {len(documents)} documents ({primary_count} primary, {supporting_count} supporting)"
            )

        return documents

    def _resolve_document_path(
        self, doc_path: str, base_directory: Path
    ) -> Path | None:
        """Resolve document path relative to base directory.

        Args:
            doc_path: Document path from manifest (can be relative)
            base_directory: Base directory for resolving relative paths

        Returns:
            Resolved Path object or None if invalid
        """
        try:
            # Handle relative paths starting with "../"
            if doc_path.startswith("../"):
                resolved_path = (base_directory / doc_path).resolve()
            elif doc_path.startswith("/"):
                resolved_path = Path(doc_path)
            else:
                resolved_path = base_directory / doc_path

            return resolved_path
        except Exception as e:
            print(f"  ⚠️  Failed to resolve document path {doc_path}: {e}")
            return None

    def _compile_documents_for_review(self, documents: list[dict[str, str]]) -> str:
        """Compile multiple documents into a single content string for review.

        Args:
            documents: List of document dictionaries

        Returns:
            Compiled content string
        """
        compiled_parts = []

        # Separate primary and supporting documents for better organization
        primary_docs = [doc for doc in documents if doc.get("type") == "primary"]
        supporting_docs = [doc for doc in documents if doc.get("type") == "supporting"]
        other_docs = [
            doc for doc in documents if doc.get("type") not in ["primary", "supporting"]
        ]

        # Compile primary documents first
        for doc in primary_docs:
            compiled_parts.append("## Primary Document")
            compiled_parts.append("")
            compiled_parts.append(f"• **File:** {doc['name']}")
            if doc.get("manifest_path"):
                compiled_parts.append(f"• **Source:** {doc['manifest_path']}")
            compiled_parts.append("")
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        # Then supporting documents
        for doc in supporting_docs:
            compiled_parts.append("## Supporting Document")
            compiled_parts.append("")
            compiled_parts.append(f"• **File:** {doc['name']}")
            if doc.get("manifest_path"):
                compiled_parts.append(f"• **Source:** {doc['manifest_path']}")
            compiled_parts.append("")
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        # Finally other documents (from directory scan)
        for doc in other_docs:
            compiled_parts.append("## Document")
            compiled_parts.append("")
            compiled_parts.append(f"• **File:** {doc['name']}")
            compiled_parts.append("")
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        return "\n".join(compiled_parts)

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load and parse manifest file with schema validation.

        Args:
            manifest_path: Path to manifest.yaml file

        Returns:
            Parsed manifest configuration dictionary
        """
        import yaml

        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest: dict[str, Any] = yaml.safe_load(f)

            # Validate against schema if available
            self._validate_manifest_schema(manifest, manifest_path)

            return manifest
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse manifest {manifest_path}: {e}")
            return {}

    def _validate_manifest_schema(
        self, manifest: dict[str, Any], manifest_path: Path
    ) -> None:
        """Validate manifest against JSON schema if available.

        Args:
            manifest: Parsed manifest dictionary
            manifest_path: Path to manifest file for error reporting
        """
        try:
            import json

            import jsonschema

            # Look for schema file in project root
            schema_path = manifest_path.parent.parent.parent / "manifest.schema.json"
            if not schema_path.exists():
                # Schema not found, skip validation
                return

            with open(schema_path) as f:
                schema = json.load(f)

            jsonschema.validate(manifest, schema)
            print(f"✅ Manifest schema validation passed: {manifest_path.name}")

        except ImportError:
            # jsonschema not available, skip validation
            return
        except jsonschema.ValidationError as e:
            print(f"❌ Manifest schema validation failed for {manifest_path.name}:")
            print(f"   {e.message}")
            if e.absolute_path:
                print(f"   Path: {'.'.join(str(p) for p in e.absolute_path)}")
            # Don't raise - allow processing to continue with warning
        except Exception as e:
            print(f"⚠️  Schema validation error for {manifest_path.name}: {e}")
            # Don't raise - allow processing to continue

    def _process_advanced_manifest(
        self, manifest_config: dict[str, Any], directory_path: Path
    ) -> dict[str, Any]:
        """Process advanced manifest features.

        Args:
            manifest_config: Parsed manifest configuration
            directory_path: Path to the document directory

        Returns:
            Enhanced manifest with processed advanced features
        """
        if not manifest_config:
            return manifest_config

        review_config = manifest_config.get("review_configuration", {})

        # Process context files
        context_files = self._process_context_files(review_config, directory_path)
        if context_files:
            review_config["processed_context"] = context_files

        # Process document relationships
        relationships = self._process_document_relationships(review_config)
        if relationships:
            review_config["processed_relationships"] = relationships

        # Process review focus priorities
        focus_config = self._process_review_focus(review_config)
        if focus_config:
            review_config["processed_focus"] = focus_config

        # Process output configuration
        output_config = self._process_output_configuration(review_config)
        if output_config:
            review_config["processed_output"] = output_config

        return manifest_config

    def _process_context_files(
        self, review_config: dict[str, Any], directory_path: Path
    ) -> list[dict[str, Any]]:
        """Process context files from manifest.

        Args:
            review_config: Review configuration section
            directory_path: Base directory path

        Returns:
            List of processed context file configurations
        """
        context_files = []
        documents = review_config.get("documents", {})
        context_file_configs = documents.get("context_files", [])

        for context_config in context_file_configs:
            context_path = directory_path / context_config["path"]
            if context_path.exists():
                try:
                    with open(context_path, encoding="utf-8") as f:
                        content = f.read()

                    processed_context = {
                        "path": context_config["path"],
                        "type": context_config.get("type", "general"),
                        "weight": context_config.get("weight", "medium"),
                        "content": content,
                        "loaded": True,
                    }
                    context_files.append(processed_context)
                    print(f"  ✓ Loaded context file: {context_config['path']}")
                except Exception as e:
                    print(
                        f"  ⚠️  Failed to load context file {context_config['path']}: {e}"
                    )
                    context_files.append(
                        {
                            "path": context_config["path"],
                            "type": context_config.get("type", "general"),
                            "weight": context_config.get("weight", "medium"),
                            "loaded": False,
                            "error": str(e),
                        }
                    )
            else:
                print(f"  ⚠️  Context file not found: {context_config['path']}")

        return context_files

    def _process_document_relationships(
        self, review_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process document relationships from manifest.

        Args:
            review_config: Review configuration section

        Returns:
            List of processed document relationships
        """
        documents = review_config.get("documents", {})
        relationships = documents.get("relationships", [])

        processed_relationships = []
        for rel in relationships:
            processed_rel = {
                "source": rel.get("source"),
                "target": rel.get("target"),
                "type": rel.get("type", "relates_to"),
                "note": rel.get("note", ""),
                "weight": rel.get("weight", "medium"),
            }
            processed_relationships.append(processed_rel)

        if processed_relationships:
            print(
                f"  📊 Processed {len(processed_relationships)} document relationships"
            )

        return processed_relationships

    def _process_review_focus(self, review_config: dict[str, Any]) -> dict[str, Any]:
        """Process review focus configuration from manifest.

        Args:
            review_config: Review configuration section

        Returns:
            Processed review focus configuration
        """
        review_focus = review_config.get("review_focus", {})

        if not review_focus:
            return {}

        processed_focus = {
            "primary_concerns": review_focus.get("primary_concerns", []),
            "secondary_concerns": review_focus.get("secondary_concerns", []),
            "focus_instructions": [],
        }

        # Generate focus instructions for reviewers
        all_concerns = (
            processed_focus["primary_concerns"] + processed_focus["secondary_concerns"]
        )

        for concern in all_concerns:
            weight = concern.get("weight", "medium")
            concern_text = concern.get("concern", "")
            description = concern.get("description", "")

            if weight == "critical":
                instruction = f"🔴 CRITICAL: Pay special attention to {concern_text}"
            elif weight == "high":
                instruction = f"🟡 HIGH PRIORITY: Focus on {concern_text}"
            else:
                instruction = f"🔵 CONSIDER: {concern_text}"

            if description:
                instruction += f" - {description}"

            processed_focus["focus_instructions"].append(instruction)

        if processed_focus["focus_instructions"]:
            print(
                f"  🎯 Configured {len(processed_focus['focus_instructions'])} review focus points"
            )

        return processed_focus

    def _process_output_configuration(
        self, review_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Process output configuration from manifest.

        Args:
            review_config: Review configuration section

        Returns:
            Processed output configuration
        """
        output_config = review_config.get("output", {})

        if not output_config:
            return {}

        processed_output = {
            "format": output_config.get("format", "standard"),
            "include_sections": output_config.get("include_sections", []),
            "exclude_sections": output_config.get("exclude_sections", []),
            "summary_length": output_config.get("summary_length", "standard"),
            "include_scores": output_config.get("include_scores", False),
            "highlight_critical_issues": output_config.get(
                "highlight_critical_issues", False
            ),
        }

        print(f"  📋 Output format: {processed_output['format']}")
        if processed_output["include_sections"]:
            print(
                f"     Including sections: {', '.join(processed_output['include_sections'])}"
            )

        return processed_output

    def _validate_loaded_documents(
        self,
        loaded_documents: list[dict[str, str]],
        enhanced_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Validate loaded documents against manifest expectations.

        Args:
            loaded_documents: List of successfully loaded documents
            enhanced_manifest: Enhanced manifest containing original config and processed data

        Returns:
            Validation results dictionary or None if validation disabled
        """
        try:
            # Extract validation configuration from enhanced manifest
            validation_config = {}
            if enhanced_manifest:
                # Get original manifest config
                original_manifest = enhanced_manifest.get(
                    "original_manifest", enhanced_manifest
                )
                processing_config = original_manifest.get(
                    "review_configuration", {}
                ).get("processing", {})
                if "max_content_length" in processing_config:
                    # Convert to word count (rough estimate: 5 chars per word)
                    validation_config["max_word_count"] = (
                        processing_config["max_content_length"] // 5
                    )

            # For now, focus on manifest compliance validation
            # Individual document content validation would require file paths, not just content
            validation_results = {}

            # Check manifest expectations vs loaded documents
            if enhanced_manifest:
                collection_issues = self._check_manifest_compliance(
                    loaded_documents, enhanced_manifest
                )
                if collection_issues:
                    validation_results["_collection_issues"] = collection_issues

            return validation_results

        except ImportError:
            print("⚠️  Document validation not available - skipping validation step")
            return None
        except Exception as e:
            print(f"⚠️  Document validation failed: {e}")
            return None

    def _check_manifest_compliance(
        self, loaded_documents: list[dict[str, str]], enhanced_manifest: dict[str, Any]
    ) -> tuple[list, Any] | None:
        """Check if loaded documents comply with manifest expectations.

        Args:
            loaded_documents: List of successfully loaded documents
            enhanced_manifest: Enhanced manifest with original config and processed data

        Returns:
            Tuple of (validation_results, metadata) or None
        """
        from ..validation.document_validator import (
            DocumentMetadata,
            ValidationLevel,
            ValidationResult,
        )

        issues = []
        loaded_manifest_paths = {
            doc.get("manifest_path")
            for doc in loaded_documents
            if doc.get("manifest_path")
        }

        # Get original manifest config
        original_manifest = enhanced_manifest.get(
            "original_manifest", enhanced_manifest
        )
        review_config = original_manifest.get("review_configuration", {})
        doc_config = review_config.get("documents", {})
        # Check primary document
        primary_doc = doc_config.get("primary")
        if primary_doc and primary_doc not in loaded_manifest_paths:
            issues.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Expected primary document missing: {primary_doc}",
                    suggestion=f"Add the missing primary document: {primary_doc}",
                )
            )

        # Check supporting documents
        supporting_docs = doc_config.get("supporting", [])
        for supporting_doc in supporting_docs:
            if supporting_doc not in loaded_manifest_paths:
                issues.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Expected supporting document missing: {supporting_doc}",
                        suggestion=f"Add the missing supporting document: {supporting_doc}",
                    )
                )

        # Check context files using enhanced manifest if available
        if enhanced_manifest:
            # Check processed context files from enhanced manifest
            processed_context = enhanced_manifest.get("review_configuration", {}).get(
                "processed_context", []
            )
            expected_context_files = doc_config.get("context_files", [])
            for expected_context in expected_context_files:
                expected_path = (
                    expected_context.get("path")
                    if isinstance(expected_context, dict)
                    else expected_context
                )

                # Check if this context file was successfully loaded
                context_loaded = any(
                    ctx.get("path") == expected_path and ctx.get("loaded", False)
                    for ctx in processed_context
                )

                if not context_loaded:
                    issues.append(
                        ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Expected context file missing: {expected_path}",
                            suggestion=f"Add the missing context file: {expected_path}",
                        )
                    )
        else:
            # Fallback: Check context files using original method
            context_files = doc_config.get("context_files", [])
            for context_file in context_files:
                context_path = (
                    context_file.get("path")
                    if isinstance(context_file, dict)
                    else context_file
                )
                if context_path and context_path not in loaded_manifest_paths:
                    issues.append(
                        ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Expected context file missing: {context_path}",
                            suggestion=f"Add the missing context file: {context_path}",
                        )
                    )

        if issues:
            metadata = DocumentMetadata(
                word_count=0,
                character_count=0,
                paragraph_count=0,
                sentence_count=0,
                reading_level=None,
                detected_language="en",
                format_type="text",
            )
            return (issues, metadata)

        return None

    def _report_validation_results(self, validation_results: dict[str, Any]) -> None:
        """Report validation results to user.

        Args:
            validation_results: Results from document validation
        """
        if not validation_results:
            return

        # Counters and categorized issues
        total_errors = total_warnings = total_info = 0
        file_issues = []  # Store (filename, errors, warnings) for later reporting

        # Single pass through all validation results
        for filename, (results, _metadata) in validation_results.items():
            if not results:
                continue

            # Categorize results by level
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            warnings = [r for r in results if r.level == ValidationLevel.WARNING]
            info = [r for r in results if r.level == ValidationLevel.INFO]

            # Update counts
            total_errors += len(errors)
            total_warnings += len(warnings)
            total_info += len(info)

            # Handle collection-level issues immediately
            if filename.startswith("_") and filename == "_collection_issues":
                if errors or warnings or info:
                    print("\n🔍 Document Collection Issues:")
                    for error in errors:
                        print(f"  ❌ {error.message}")
                    for warning in warnings:
                        print(f"  ⚠️  {warning.message}")
                    for item in info:
                        print(f"  ℹ️  {item.message}")
            elif not filename.startswith("_") and (errors or warnings):
                # Store file-level issues for later reporting
                file_issues.append((filename, errors, warnings))

        # Report summary
        if total_errors == 0 and total_warnings == 0 and total_info == 0:
            print("✅ All documents passed validation")
        else:
            summary_parts = []
            if total_errors > 0:
                summary_parts.append(f"{total_errors} errors")
            if total_warnings > 0:
                summary_parts.append(f"{total_warnings} warnings")
            if total_info > 0:
                summary_parts.append(f"{total_info} info items")
            print(f"📋 Document validation found: {', '.join(summary_parts)}")

        # Report file-specific issues
        for filename, errors, warnings in file_issues:
            print(f"📄 {filename}:")
            for error in errors:
                print(f"  ❌ {error.message}")
            for warning in warnings:
                print(f"  ⚠️  {warning.message}")

        print()  # Add spacing after validation report
