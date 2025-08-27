"""
Conversation Manager for orchestrating multi-agent reviews.

This module manages conversations between multiple review agents,
collecting and organizing their feedback.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .review_agent import ReviewAgent
from .analysis_agent import AnalysisAgent, AnalysisResult
from .context_agent import ContextAgent, ContextResult
from ..config.persona_loader import PersonaLoader, PersonaConfig

import tiktoken


@dataclass
class ReviewResult:
    """Result from a single agent review."""

    agent_name: str
    agent_role: str
    feedback: str
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class ConversationResult:
    """Complete conversation result with all agent reviews."""

    content: str
    reviews: List[ReviewResult]
    timestamp: datetime
    summary: Optional[str] = None
    analysis_results: List[AnalysisResult] = None
    context_results: List[ContextResult] = None
    original_content: Optional[str] = None
    analysis_errors: List[str] = None  # Track analysis failures separately

    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.context_results is None:
            self.context_results = []
        if self.analysis_errors is None:
            self.analysis_errors = []


class ConversationManager:
    """Manages multi-agent review conversations."""

    def __init__(
        self,
        persona_loader: Optional[PersonaLoader] = None,
        model_provider: str = "bedrock",
        model_config: Optional[Dict[str, Any]] = None,
        enable_analysis: bool = True,
    ):
        """Initialize the conversation manager.

        Args:
            persona_loader: Optional PersonaLoader instance
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
            enable_analysis: Whether to enable analysis of reviews using analyzer personas
        """
        self.persona_loader = persona_loader or PersonaLoader()
        self.model_provider = model_provider
        self.model_config = model_config or {}
        self.enable_analysis = enable_analysis
        self.agents: List[ReviewAgent] = []
        self.context_agents: List[ContextAgent] = []
        self.analysis_agents: List[AnalysisAgent] = []
        self._load_agents()
        self._load_contextualizers()

        if self.enable_analysis:
            self._load_analyzers()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Use cl100k_base encoding (used by GPT-4 and similar models)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _truncate_to_token_limit(self, text: str, target_tokens: int) -> str:
        """Truncate text to the target number of tokens.

        Args:
            text: Text to truncate
            target_tokens: Target number of tokens

        Returns:
            Truncated text
        """
        if target_tokens <= 0:
            return ""

        if tiktoken is None:
            raise ImportError(
                "tiktoken is required but not installed. Run: pip install tiktoken"
            )

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        if len(tokens) <= target_tokens:
            return text

        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:target_tokens]
        return encoding.decode(truncated_tokens)

    def _load_agents(self) -> None:
        """Load all reviewer personas as review agents."""
        try:
            personas = self.persona_loader.load_reviewer_personas()
            self.agents = [
                ReviewAgent(
                    persona,
                    model_provider=self.model_provider,
                    model_config_override=self.model_config,
                )
                for persona in personas
            ]
            print(
                f"✅ Loaded {len(self.agents)} review agents with {self.model_provider} provider"
            )
        except Exception as e:
            print(f"❌ Error loading agents: {e}")
            self.agents = []

    def _load_contextualizers(self) -> None:
        """Load all contextualizer personas as context agents."""
        try:
            contextualizer_personas = self.persona_loader.load_contextualizer_personas()

            if contextualizer_personas:
                print(
                    f"✅ Loaded {len(contextualizer_personas)} contextualizer personas"
                )

                # Create context agents for all contextualizer personas
                self.context_agents = []
                for persona in contextualizer_personas:
                    context_agent = ContextAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.context_agents.append(context_agent)
                    print(f"✅ Created context agent: {persona.name}")
            else:
                print("ℹ️  No contextualizer personas found")
                self.context_agents = []

        except Exception as e:
            print(f"❌ Error loading contextualizers: {e}")
            self.context_agents = []

    def _load_specific_contextualizers(self, contextualizer_personas: List) -> None:
        """Load specific contextualizer personas as context agents.

        Args:
            contextualizer_personas: List of PersonaConfig objects for contextualizers
        """
        try:
            if contextualizer_personas:
                print(
                    f"✅ Loading {len(contextualizer_personas)} selected contextualizer personas"
                )

                # Create context agents for selected contextualizer personas
                self.context_agents = []
                for persona in contextualizer_personas:
                    context_agent = ContextAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.context_agents.append(context_agent)
                    print(f"✅ Created context agent: {persona.name}")
            else:
                print("ℹ️  No contextualizer personas specified")
                self.context_agents = []

        except Exception as e:
            print(f"❌ Error loading specific contextualizers: {e}")
            self.context_agents = []

    def _load_specific_analyzers(self, analyzer_personas: List) -> None:
        """Load specific analyzer personas as analysis agents.

        Args:
            analyzer_personas: List of PersonaConfig objects for analyzers
        """
        try:
            if analyzer_personas:
                print(f"✅ Loading {len(analyzer_personas)} selected analyzer personas")

                # Create analysis agents for selected analyzer personas
                self.analysis_agents = []
                for persona in analyzer_personas:
                    analysis_agent = AnalysisAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.analysis_agents.append(analysis_agent)
                    print(f"✅ Created analysis agent: {persona.name}")
            else:
                print("ℹ️  No analyzer personas specified")
                self.analysis_agents = []

        except Exception as e:
            print(f"❌ Error loading specific analyzers: {e}")
            self.analysis_agents = []

    def _load_analyzers(self) -> None:
        """Load all analyzer personas as analysis agents."""
        try:
            analyzer_personas = self.persona_loader.load_analyzer_personas()

            if analyzer_personas:
                print(f"✅ Loaded {len(analyzer_personas)} analyzer personas")

                # Create analysis agents for all analyzer personas
                self.analysis_agents = []
                for persona in analyzer_personas:
                    analysis_agent = AnalysisAgent(
                        persona=persona,
                        model_provider=self.model_provider,
                        model_config=self.model_config,
                    )
                    self.analysis_agents.append(analysis_agent)
                    print(f"✅ Created analysis agent: {persona.name}")
            else:
                print("ℹ️  No analyzer personas found")
                self.analysis_agents = []

        except Exception as e:
            print(f"❌ Error loading analyzers: {e}")
            self.analysis_agents = []

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get information about available agents.

        Returns:
            List of agent information dictionaries
        """
        return [agent.get_info() for agent in self.agents]

    def get_available_contextualizers(self) -> List[Dict[str, Any]]:
        """Get information about available contextualizer agents.

        Returns:
            List of contextualizer agent information dictionaries
        """
        return [agent.get_info() for agent in self.context_agents]

    def get_available_analyzers(self) -> List[Dict[str, Any]]:
        """Get information about available analyzer agents.

        Returns:
            List of analyzer agent information dictionaries
        """
        return [agent.get_info() for agent in self.analysis_agents]

    def run_review(
        self,
        content: str,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run a synchronous review with selected agents.

        Args:
            content: Content to review or directory path for multi-document review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        # Check if content is a directory path for multi-document review
        content_path = Path(content)
        if content_path.exists() and content_path.is_dir():
            return self._run_multi_document_review(
                content_path, context_data, selected_agents
            )

        # Single document review
        return self._run_single_document_review(content, context_data, selected_agents)

    def _run_single_document_review(
        self,
        content: str,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run a synchronous review on single document content.

        Args:
            content: Content to review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        if not self.agents:
            raise ValueError(
                "No review agents available. Check your persona configurations."
            )

        # Process context with all contextualizers if provided and available
        context_results = []
        if context_data and self.context_agents:
            print(
                f"🔍 Processing context information with {len(self.context_agents)} contextualizers..."
            )
            for context_agent in self.context_agents:
                try:
                    context_result = context_agent.process_context(context_data)
                    if context_result:
                        context_results.append(context_result)
                        print(
                            f"  ✅ Context processed by {context_agent.persona.name}: {context_result.context_summary}"
                        )
                    else:
                        print(f"  ℹ️  {context_agent.persona.name} returned no context")
                except Exception as e:
                    print(
                        f"  ⚠️  Context processing failed for {context_agent.persona.name}: {e}"
                    )
        elif context_data:
            print("  ℹ️  Context data provided but no contextualizer personas available")

        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)

        print(f"🎭 Starting review with {len(agents_to_use)} agents...")

        reviews = []
        for agent in agents_to_use:
            print(f"  📝 {agent.persona.name} is reviewing...")

            try:
                # Prepare content for review (include context if available)
                content_to_review = self._prepare_content_for_review(
                    content, context_results
                )

                feedback = agent.review(content_to_review)
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback=feedback,
                    timestamp=datetime.now(),
                )
                reviews.append(review)
                print(f"  ✅ {agent.persona.name} completed review")

            except Exception as e:
                error_str = str(e)
                # Check if this is a context length error during review generation
                if "context length" in error_str.lower() and "4096" in error_str:
                    print(
                        f"  ❌ {agent.persona.name} failed due to context length limit (input too large for model)"
                    )
                    print(
                        f"      The model ran out of space before completing the review"
                    )
                    print(f"      Try: --max-context-length 8192 or use a larger model")
                    # Mark as failed with helpful error message
                    error_review = ReviewResult(
                        agent_name=agent.persona.name,
                        agent_role=agent.persona.role,
                        feedback="",
                        timestamp=datetime.now(),
                        error=f"Context length exceeded: {error_str}. The input content + context was too large for the 4096 token model limit. Any partial output shown in console was incomplete.",
                    )
                    reviews.append(error_review)
                else:
                    error_review = ReviewResult(
                        agent_name=agent.persona.name,
                        agent_role=agent.persona.role,
                        feedback="",
                        timestamp=datetime.now(),
                        error=error_str,
                    )
                    reviews.append(error_review)
                    print(f"  ❌ {agent.persona.name} failed: {e}")

        result = ConversationResult(
            content=content,
            reviews=reviews,
            context_results=context_results,
            original_content=None,  # No longer needed since we're not extracting
            timestamp=datetime.now(),
        )

        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agents:
            successful_reviews = [r for r in reviews if not r.error]
            if (
                len(successful_reviews) > 0
            ):  # Always run analysis if we have any successful reviews
                print(
                    f"🧠 Performing analysis with {len(self.analysis_agents)} analyzers..."
                )
                analysis_results = []

                # Convert ReviewResult objects to dictionaries for analysis
                review_dicts = [
                    {
                        "agent_name": r.agent_name,
                        "agent_role": r.agent_role,
                        "feedback": r.feedback,
                        "error": r.error,
                    }
                    for r in successful_reviews
                ]

                # Get context length from model config or use default
                max_context_length = self.model_config.get("max_context_length", None)

                analysis_errors = []
                for analysis_agent in self.analysis_agents:
                    try:
                        print(
                            f"  📊 Running analysis with {analysis_agent.persona.name}..."
                        )
                        analysis_result = analysis_agent.analyze(
                            review_dicts, max_context_length
                        )
                        analysis_results.append(analysis_result)
                        print(
                            f"  ✅ Analysis complete for {analysis_agent.persona.name}"
                        )
                    except Exception as e:
                        error_msg = f"{analysis_agent.persona.name}: {str(e)}"
                        analysis_errors.append(error_msg)
                        print(
                            f"  ⚠️  Analysis failed for {analysis_agent.persona.name}: {e}"
                        )

                result.analysis_results = analysis_results
                result.analysis_errors = analysis_errors

                if analysis_results:
                    print(
                        f"✅ Analysis complete! Ran {len(analysis_results)} successful analyzers"
                    )
                if analysis_errors:
                    print(
                        f"⚠️  {len(analysis_errors)} analyzer(s) failed but reviews are still available"
                    )
                    print("💡 Check the output for detailed analysis error information")

        print(
            f"🎉 Review complete! Collected {len([r for r in reviews if not r.error])} successful reviews"
        )
        return result

    def _run_multi_document_review(
        self,
        directory_path: Path,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run review on multiple documents in a directory.

        Args:
            directory_path: Path to directory containing documents
            context_data: Optional context information
            selected_agents: Optional list of agent names to use

        Returns:
            ConversationResult with multi-document review
        """
        print(f"📂 Processing document collection from: {directory_path}")

        # Check for manifest file first to determine document loading strategy
        manifest_path = directory_path / "manifest.yaml"
        documents = []
        compiled_content = ""

        if manifest_path.exists():
            print(f"📋 Found manifest file, using manifest-driven document loading")
            manifest_config = self._load_manifest(manifest_path)

            # Load documents according to manifest specification
            documents = self._collect_documents_from_manifest(
                manifest_config, directory_path
            )

            if not documents:
                print(
                    "⚠️  No documents specified in manifest, falling back to directory scan"
                )
                documents = self._collect_documents_from_directory(directory_path)
        else:
            print(f"📁 No manifest found, scanning directory for documents")
            # No manifest - collect all documents from directory
            documents = self._collect_documents_from_directory(directory_path)

        if not documents:
            raise ValueError(f"No readable documents found in {directory_path}")

        print(f"📄 Found {len(documents)} documents to review")

        # Compile documents into single content for review
        compiled_content = self._compile_documents_for_review(documents)

        # Check if we need chunking due to size
        max_context_length = self.model_config.get("max_context_length", 4096)
        content_tokens = self._count_tokens(compiled_content)

        if content_tokens > max_context_length:
            print(
                f"📊 Content is {content_tokens} tokens, chunking for {max_context_length} token limit"
            )
            # For now, truncate - later we can implement smart chunking
            compiled_content = self._truncate_to_token_limit(
                compiled_content, max_context_length - 500
            )

        # Document validation pipeline
        validation_results = self._validate_document_collection(directory_path)
        if validation_results:
            self._report_validation_results(validation_results)

        # If manifest exists, process advanced features and run manifest-driven review
        if manifest_path.exists():
            manifest_config = self._load_manifest(manifest_path)

            # Process advanced manifest features
            enhanced_manifest = self._process_advanced_manifest(
                manifest_config, directory_path
            )

            return self._run_manifest_review(
                compiled_content, context_data, enhanced_manifest
            )

        # No manifest - run standard review on compiled content
        return self._run_single_document_review(
            compiled_content, context_data, selected_agents
        )

    def _collect_documents_from_directory(
        self, directory_path: Path
    ) -> List[Dict[str, str]]:
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
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append({"name": file_path.name, "content": content})
                    print(f"  ✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"  ⚠️  Skipped {file_path.name}: {e}")
                    continue

        return documents

    def _collect_documents_from_manifest(
        self, manifest_config: Dict[str, Any], directory_path: Path
    ) -> List[Dict[str, str]]:
        """Collect documents specified in manifest configuration.

        Args:
            manifest_config: Parsed manifest configuration
            directory_path: Base directory path for resolving relative paths

        Returns:
            List of document dictionaries with 'name', 'content', and 'type' keys
        """
        documents = []

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
                    with open(primary_path, "r", encoding="utf-8") as f:
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
                    with open(supporting_path, "r", encoding="utf-8") as f:
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
            print(
                f"  📋 Manifest specified {len(documents)} documents ({len([d for d in documents if d['type'] == 'primary'])} primary, {len([d for d in documents if d['type'] == 'supporting'])} supporting)"
            )

        return documents

    def _resolve_document_path(
        self, doc_path: str, base_directory: Path
    ) -> Optional[Path]:
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
                # Go up from base directory and resolve
                resolved_path = (base_directory / doc_path).resolve()
            elif doc_path.startswith("/"):
                # Absolute path
                resolved_path = Path(doc_path)
            else:
                # Relative to base directory
                resolved_path = base_directory / doc_path

            return resolved_path
        except Exception as e:
            print(f"  ⚠️  Failed to resolve document path {doc_path}: {e}")
            return None

    def _compile_documents_for_review(self, documents: List[Dict[str, str]]) -> str:
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
            doc_header = f"=== PRIMARY DOCUMENT: {doc['name']} ==="
            if doc.get("manifest_path"):
                doc_header += f" (from manifest: {doc['manifest_path']})"
            compiled_parts.append(doc_header)
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        # Then supporting documents
        for doc in supporting_docs:
            doc_header = f"=== SUPPORTING DOCUMENT: {doc['name']} ==="
            if doc.get("manifest_path"):
                doc_header += f" (from manifest: {doc['manifest_path']})"
            compiled_parts.append(doc_header)
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        # Finally other documents (from directory scan)
        for doc in other_docs:
            compiled_parts.append(f"=== Document: {doc['name']} ===")
            compiled_parts.append(doc["content"])
            compiled_parts.append("")  # Empty line between documents

        return "\n".join(compiled_parts)

    def _load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """Load and parse manifest file.

        Args:
            manifest_path: Path to manifest.yaml file

        Returns:
            Parsed manifest configuration dictionary
        """
        import yaml

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            return manifest
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse manifest {manifest_path}: {e}")
            return {}

    def _process_advanced_manifest(
        self, manifest_config: Dict[str, Any], directory_path: Path
    ) -> Dict[str, Any]:
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
        self, review_config: Dict[str, Any], directory_path: Path
    ) -> List[Dict[str, Any]]:
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
                    with open(context_path, "r", encoding="utf-8") as f:
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
        self, review_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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

    def _process_review_focus(self, review_config: Dict[str, Any]) -> Dict[str, Any]:
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
        self, review_config: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    def _build_enhanced_context(
        self, context_data: Optional[str], review_config: Dict[str, Any]
    ) -> Optional[str]:
        """Build enhanced context incorporating manifest context files.

        Args:
            context_data: Original context data
            review_config: Review configuration section

        Returns:
            Enhanced context string or None
        """
        context_parts = []

        # Add original context if provided
        if context_data:
            context_parts.append("## Original Context")
            context_parts.append(context_data)
            context_parts.append("")

        # Add processed context files from manifest
        processed_context = review_config.get("processed_context", [])
        for context_file in processed_context:
            if context_file.get("loaded"):
                context_type = context_file.get("type", "general")
                weight = context_file.get("weight", "medium")

                weight_indicator = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(
                    weight, "🔵"
                )

                context_parts.append(
                    f"## Context: {context_file['path']} {weight_indicator}"
                )
                context_parts.append(f"Type: {context_type.title()}")
                context_parts.append("")
                context_parts.append(context_file["content"])
                context_parts.append("")

        # Add document relationships as context
        relationships = review_config.get("processed_relationships", [])
        if relationships:
            context_parts.append("## Document Relationships")
            for rel in relationships:
                rel_text = f"- {rel['source']} {rel['type']} {rel['target']}"
                if rel.get("note"):
                    rel_text += f": {rel['note']}"
                context_parts.append(rel_text)
            context_parts.append("")

        # Add focus instructions as context
        focus_config = review_config.get("processed_focus", {})
        focus_instructions = focus_config.get("focus_instructions", [])
        if focus_instructions:
            context_parts.append("## Review Focus Instructions")
            for instruction in focus_instructions:
                context_parts.append(f"- {instruction}")
            context_parts.append("")

        return "\n".join(context_parts) if context_parts else context_data

    def _apply_focus_to_reviewers(
        self, reviewers: List, review_config: Dict[str, Any]
    ) -> List:
        """Apply focus instructions to reviewer personas.

        Args:
            reviewers: List of reviewer PersonaConfig objects
            review_config: Review configuration section

        Returns:
            List of reviewer personas with enhanced prompts
        """
        focus_config = review_config.get("processed_focus", {})
        focus_instructions = focus_config.get("focus_instructions", [])

        if not focus_instructions:
            return reviewers

        enhanced_reviewers = []
        for reviewer in reviewers:
            # Create a copy of the reviewer with enhanced prompt
            enhanced_reviewer = type(reviewer)(
                name=reviewer.name,
                role=reviewer.role,
                goal=reviewer.goal,
                backstory=reviewer.backstory,
                prompt_template=reviewer.prompt_template,
                model_config=reviewer.model_config,
            )

            # Enhance the prompt template with focus instructions
            focus_section = "\n\n## SPECIAL FOCUS AREAS FOR THIS REVIEW\n"
            focus_section += "\n".join(focus_instructions)
            focus_section += "\n\nPlease pay particular attention to these focus areas in your review.\n"

            enhanced_reviewer.prompt_template = (
                enhanced_reviewer.prompt_template + focus_section
            )
            enhanced_reviewers.append(enhanced_reviewer)

        return enhanced_reviewers

    def _apply_output_configuration(
        self, result: "ConversationResult", review_config: Dict[str, Any]
    ) -> "ConversationResult":
        """Apply output configuration to modify result formatting.

        Args:
            result: Original conversation result
            review_config: Review configuration section

        Returns:
            Modified conversation result with applied output configuration
        """
        output_config = review_config.get("processed_output", {})

        if not output_config:
            return result

        # For now, just add metadata about the output configuration
        # In a full implementation, this would modify the actual formatting
        if hasattr(result, "metadata"):
            if not result.metadata:
                result.metadata = {}
            result.metadata["output_config"] = output_config
        else:
            # Add metadata attribute if it doesn't exist
            result.metadata = {"output_config": output_config}

        return result

    def _validate_document_collection(
        self, directory_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Validate document collection using the validation pipeline.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Validation results dictionary or None if validation disabled
        """
        try:
            from ..validation.document_validator import DocumentValidator

            # Load validation configuration from manifest if available
            manifest_path = directory_path / "manifest.yaml"
            validation_config = {}

            if manifest_path.exists():
                manifest = self._load_manifest(manifest_path)
                processing_config = manifest.get("review_configuration", {}).get(
                    "processing", {}
                )

                # Extract validation-relevant settings
                if "max_content_length" in processing_config:
                    # Convert to word count (rough estimate: 5 chars per word)
                    validation_config["max_word_count"] = (
                        processing_config["max_content_length"] // 5
                    )

            validator = DocumentValidator(validation_config)

            # Get manifest config for expected documents
            manifest_config = None
            if manifest_path.exists():
                manifest_config = self._load_manifest(manifest_path)

            validation_results = validator.validate_document_collection(
                directory_path, manifest_config
            )
            return validation_results

        except ImportError:
            print("⚠️  Document validation not available - skipping validation step")
            return None
        except Exception as e:
            print(f"⚠️  Document validation failed: {e}")
            return None

    def _report_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Report validation results to user.

        Args:
            validation_results: Results from document validation
        """
        if not validation_results:
            return

        # Count issues by severity
        total_errors = 0
        total_warnings = 0
        total_info = 0

        for filename, (results, metadata) in validation_results.items():
            if filename.startswith("_"):
                continue

            for result in results:
                if result.level.value == "error":
                    total_errors += 1
                elif result.level.value == "warning":
                    total_warnings += 1
                elif result.level.value == "info":
                    total_info += 1

        # Report summary
        if total_errors > 0:
            print(f"❌ Document validation found {total_errors} errors")
        if total_warnings > 0:
            print(f"⚠️  Document validation found {total_warnings} warnings")
        if total_info > 0:
            print(f"ℹ️  Document validation found {total_info} information items")

        if total_errors == 0 and total_warnings == 0 and total_info == 0:
            print("✅ All documents passed validation")

        # Report specific issues for errors and warnings
        for filename, (results, metadata) in validation_results.items():
            if filename.startswith("_"):
                continue

            errors = [r for r in results if r.level.value == "error"]
            warnings = [r for r in results if r.level.value == "warning"]

            if errors or warnings:
                print(f"📄 {filename}:")
                for error in errors:
                    print(f"  ❌ {error.message}")
                for warning in warnings:
                    print(f"  ⚠️  {warning.message}")

        print()  # Add spacing after validation report

    def _run_manifest_review(
        self,
        content: str,
        context_data: Optional[str] = None,
        manifest_config: Dict[str, Any] = None,
    ) -> ConversationResult:
        """Run review based on manifest configuration.

        Args:
            content: Content to review
            context_data: Optional context information
            manifest_config: Manifest configuration dictionary

        Returns:
            ConversationResult with manifest-driven review
        """
        if not manifest_config:
            # Fallback to standard review
            return self._run_single_document_review(content, context_data, None)

        review_config = manifest_config.get("review_configuration", {})

        # Process enhanced context from manifest
        enhanced_context = self._build_enhanced_context(context_data, review_config)

        # Load reviewers based on manifest
        selected_reviewers = []
        if review_config:
            try:
                selected_reviewers = self.persona_loader.load_reviewers_from_manifest(
                    review_config
                )
                print(f"🎯 Manifest specified {len(selected_reviewers)} reviewers")
            except Exception as e:
                print(f"⚠️  Error loading reviewers from manifest: {e}")
                print("📋 Falling back to all available reviewers")

        # Apply review focus instructions to reviewer personas
        if review_config.get("processed_focus"):
            selected_reviewers = self._apply_focus_to_reviewers(
                selected_reviewers, review_config
            )

        # Load contextualizers based on manifest
        selected_contextualizers = []
        original_context_agents = self.context_agents  # Store original context agents
        if review_config:
            try:
                selected_contextualizers = (
                    self.persona_loader.load_contextualizers_from_manifest(
                        review_config
                    )
                )
                if selected_contextualizers:
                    print(
                        f"🎯 Manifest specified {len(selected_contextualizers)} contextualizers"
                    )
                    # Replace context agents temporarily
                    self._load_specific_contextualizers(selected_contextualizers)
                else:
                    print(
                        "📋 No contextualizers specified in manifest, using all available"
                    )
            except Exception as e:
                print(f"⚠️  Error loading contextualizers from manifest: {e}")
                print("📋 Falling back to all available contextualizers")

        # Load analyzers based on manifest
        selected_analyzers = []
        original_analysis_agents = (
            self.analysis_agents
        )  # Store original analysis agents
        if review_config and self.enable_analysis:
            try:
                selected_analyzers = self.persona_loader.load_analyzers_from_manifest(
                    review_config
                )
                if selected_analyzers:
                    print(f"🎯 Manifest specified {len(selected_analyzers)} analyzers")
                    # Replace analysis agents temporarily
                    self._load_specific_analyzers(selected_analyzers)
                else:
                    print("📋 No analyzers specified in manifest, using all available")
            except Exception as e:
                print(f"⚠️  Error loading analyzers from manifest: {e}")
                print("📋 Falling back to all available analyzers")

        # Create temporary agents for this review
        if selected_reviewers:
            from .review_agent import ReviewAgent

            temp_agents = [
                ReviewAgent(persona, self.model_provider, self.model_config)
                for persona in selected_reviewers
            ]

            # Store original agents and replace temporarily
            original_agents = self.agents
            self.agents = temp_agents

            try:
                # Run review with selected agents and enhanced context
                result = self._run_single_document_review(
                    content, enhanced_context, None
                )

                # Apply output configuration if specified
                if review_config.get("processed_output"):
                    result = self._apply_output_configuration(result, review_config)

                return result
            finally:
                # Restore original agents
                self.agents = original_agents

                # Restore original context agents
                self.context_agents = original_context_agents

                # Restore original analysis agents
                if "original_analysis_agents" in locals():
                    self.analysis_agents = original_analysis_agents
        else:
            # No specific reviewers found, use standard review with enhanced context
            try:
                result = self._run_single_document_review(
                    content, enhanced_context, None
                )

                # Apply output configuration if specified
                if review_config.get("processed_output"):
                    result = self._apply_output_configuration(result, review_config)

                return result
            finally:
                # Restore original context agents even if no specific reviewers
                if "original_context_agents" in locals():
                    self.context_agents = original_context_agents

                # Restore original analysis agents even if no specific reviewers
                if "original_analysis_agents" in locals():
                    self.analysis_agents = original_analysis_agents

    async def run_review_async(
        self,
        content: str,
        context_data: Optional[str] = None,
        selected_agents: Optional[List[str]] = None,
    ) -> ConversationResult:
        """Run an asynchronous review with selected agents.

        Args:
            content: Content to review or directory path for multi-document review
            context_data: Optional context information to be processed by contextualizer
            selected_agents: Optional list of agent names to use (uses all if None)

        Returns:
            ConversationResult with all reviews
        """
        # Check if content is a directory path for multi-document review
        content_path = Path(content)
        if content_path.exists() and content_path.is_dir():
            # For async multi-document review, we'll use the sync method for now
            # since the multi-document logic is complex and doesn't need to be async
            return self._run_multi_document_review(
                content_path, context_data, selected_agents
            )

        if not self.agents:
            raise ValueError(
                "No review agents available. Check your persona configurations."
            )

        # Process context with all contextualizers if provided and available
        context_results = []
        if context_data and self.context_agents:
            print(
                f"🔍 Processing context information with {len(self.context_agents)} contextualizers..."
            )

            # Run all contextualizers concurrently
            context_tasks = []
            for context_agent in self.context_agents:
                task = self._process_context_with_agent_async(
                    context_agent, context_data
                )
                context_tasks.append(task)

            context_task_results = await asyncio.gather(
                *context_tasks, return_exceptions=True
            )

            # Process results
            for i, result in enumerate(context_task_results):
                context_agent = self.context_agents[i]
                if isinstance(result, Exception):
                    print(
                        f"  ⚠️  Context processing failed for {context_agent.persona.name}: {result}"
                    )
                elif result:
                    context_results.append(result)
                    print(
                        f"  ✅ Context processed by {context_agent.persona.name}: {result.context_summary}"
                    )
                else:
                    print(f"  ℹ️  {context_agent.persona.name} returned no context")
        elif context_data:
            print("  ℹ️  Context data provided but no contextualizer personas available")

        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)

        print(f"🎭 Starting async review with {len(agents_to_use)} agents...")

        # Prepare content for review
        content_to_review = self._prepare_content_for_review(content, context_results)

        # Run all reviews concurrently
        tasks = []
        for agent in agents_to_use:
            task = self._review_with_agent_async(agent, content_to_review)
            tasks.append(task)

        reviews = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_reviews = []
        for i, result in enumerate(reviews):
            agent = agents_to_use[i]

            if isinstance(result, Exception):
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=str(result),
                )
                print(f"  ❌ {agent.persona.name} failed: {result}")
            else:
                review = result
                print(f"  ✅ {agent.persona.name} completed review")

            processed_reviews.append(review)

        result = ConversationResult(
            content=content,
            reviews=processed_reviews,
            context_results=context_results,
            timestamp=datetime.now(),
        )

        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agents:
            successful_reviews = [r for r in processed_reviews if not r.error]
            if (
                len(successful_reviews) > 0
            ):  # Always run analysis if we have any successful reviews
                print(
                    f"🧠 Performing async analysis with {len(self.analysis_agents)} analyzers..."
                )

                # Convert ReviewResult objects to dictionaries for analysis
                review_dicts = [
                    {
                        "agent_name": r.agent_name,
                        "agent_role": r.agent_role,
                        "feedback": r.feedback,
                        "error": r.error,
                    }
                    for r in successful_reviews
                ]

                # Get context length from model config or use default
                max_context_length = self.model_config.get("max_context_length", None)

                # Run all analyzers concurrently
                analysis_tasks = []
                for analysis_agent in self.analysis_agents:
                    task = self._analyze_with_agent_async(
                        analysis_agent, review_dicts, max_context_length
                    )
                    analysis_tasks.append(task)

                analysis_task_results = await asyncio.gather(
                    *analysis_tasks, return_exceptions=True
                )

                # Process results
                analysis_results = []
                analysis_errors = []
                for i, analysis_result in enumerate(analysis_task_results):
                    analysis_agent = self.analysis_agents[i]
                    if isinstance(analysis_result, Exception):
                        error_msg = (
                            f"{analysis_agent.persona.name}: {str(analysis_result)}"
                        )
                        analysis_errors.append(error_msg)
                        print(
                            f"  ⚠️  Analysis failed for {analysis_agent.persona.name}: {analysis_result}"
                        )
                    else:
                        analysis_results.append(analysis_result)
                        print(
                            f"  ✅ Analysis complete for {analysis_agent.persona.name}"
                        )

                result.analysis_results = analysis_results
                result.analysis_errors = analysis_errors

                if analysis_results:
                    print(
                        f"✅ Async analysis complete! Ran {len(analysis_results)} successful analyzers"
                    )
                if analysis_errors:
                    print(
                        f"⚠️  {len(analysis_errors)} analyzer(s) failed but reviews are still available"
                    )
                    print("💡 Check the output for detailed analysis error information")

        print(
            f"🎉 Async review complete! Collected {len([r for r in processed_reviews if not r.error])} successful reviews"
        )
        return result

    async def _review_with_agent_async(
        self, agent: ReviewAgent, content: str
    ) -> ReviewResult:
        """Review content with a single agent asynchronously."""
        try:
            feedback = await agent.review_async(content)
            return ReviewResult(
                agent_name=agent.persona.name,
                agent_role=agent.persona.role,
                feedback=feedback,
                timestamp=datetime.now(),
            )
        except Exception as e:
            error_str = str(e)
            # Check if this is a context length error during review generation
            if "context length" in error_str.lower() and "4096" in error_str:
                print(
                    f"  ❌ {agent.persona.name} failed due to context length limit (input too large for model)"
                )
                print(f"      The model ran out of space before completing the review")
                print(f"      Try: --max-context-length 8192 or use a larger model")
                # Mark as failed with helpful error message
                return ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=f"Context length exceeded: {error_str}. The input content + context was too large for the 4096 token model limit. Any partial output shown in console was incomplete.",
                )
            else:
                return ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=error_str,
                )

    async def _process_context_with_agent_async(
        self, context_agent: ContextAgent, context_data: str
    ) -> Optional[ContextResult]:
        """Process context with a single context agent asynchronously."""
        try:
            return await context_agent.process_context_async(context_data)
        except Exception as e:
            print(
                f"  ⚠️  Context processing failed for {context_agent.persona.name}: {e}"
            )
            return None

    async def _analyze_with_agent_async(
        self,
        analysis_agent: AnalysisAgent,
        review_dicts: List[Dict[str, Any]],
        max_context_length: Optional[int],
    ) -> AnalysisResult:
        """Analyze reviews with a single analysis agent asynchronously."""
        return await analysis_agent.analyze_async(review_dicts, max_context_length)

    def _filter_agents(self, selected_agents: Optional[List[str]]) -> List[ReviewAgent]:
        """Filter agents based on selection.

        Args:
            selected_agents: List of agent names to include, or None for all

        Returns:
            Filtered list of ReviewAgent objects
        """
        if selected_agents is None:
            return self.agents

        # Filter by name (case-insensitive)
        selected_lower = [name.lower() for name in selected_agents]
        filtered = [
            agent
            for agent in self.agents
            if agent.persona.name.lower() in selected_lower
        ]

        if not filtered:
            print(f"⚠️  No agents found matching: {selected_agents}")
            print(f"Available agents: {[agent.persona.name for agent in self.agents]}")
            return self.agents

        return filtered

    def _prepare_content_for_review(
        self, content: str, context_results: List[ContextResult]
    ) -> str:
        """Prepare content for review by combining it with extracted context.

        Args:
            content: The cleaned content to review
            context_results: List of context extraction results

        Returns:
            Formatted content for review agents
        """
        if not context_results:
            return content

        # Format all context results for reviewers
        context_sections = []
        for i, context_result in enumerate(context_results, 1):
            # Find the corresponding context agent to format the context
            context_agent = None
            for agent in self.context_agents:
                if agent.persona and context_result.context_summary:
                    # Match by checking if the context summary matches
                    context_agent = agent
                    break

            if context_agent:
                context_section = context_agent.format_context_for_review(
                    context_result
                )
                context_sections.append(f"### Context {i}\n{context_section}")
            else:
                # Fallback formatting if no agent found
                context_sections.append(
                    f"### Context {i}\n## CONTEXT\n{context_result.formatted_context}\n\n---\n"
                )

        # Combine all context with content
        all_context = "\n".join(context_sections)
        combined_content = f"""{all_context}

## CONTENT TO REVIEW
{content}"""

        # Check if the combined content is too long and truncate if necessary
        return self._truncate_if_needed(combined_content, content, all_context)

    def _truncate_if_needed(
        self, combined_content: str, original_content: str, context_content: str
    ) -> str:
        """Truncate context if the combined content is too long for the model.

        Args:
            combined_content: The full combined content (context + original content)
            original_content: The original content to review (must be preserved)
            context_content: The context content that can be truncated

        Returns:
            Truncated content if necessary, with warning logged
        """
        # Get max context length from model config, default to 4096 if not set
        max_context_length = self.model_config.get("max_context_length", 4096)

        # Reserve tokens for model response and prompt overhead
        response_buffer = 1000  # tokens for response
        prompt_overhead = 800  # tokens for system prompt and formatting

        # Available tokens for input content
        available_tokens = max_context_length - response_buffer - prompt_overhead

        # Count actual tokens in the combined content
        current_tokens = self._count_tokens(combined_content)

        print(
            f"📏 Token limit check: {current_tokens} tokens vs {available_tokens} available"
        )

        # Check if truncation is needed
        if current_tokens <= available_tokens:
            return combined_content

        # Calculate how much context we need to truncate
        original_content_with_header = f"## CONTENT TO REVIEW\n{original_content}"
        original_tokens = self._count_tokens(original_content_with_header)
        context_budget_tokens = available_tokens - original_tokens

        if context_budget_tokens <= 0:
            # Original content itself is too long, warn but proceed
            print(
                f"⚠️  Warning: Original content ({original_tokens} tokens) exceeds available context budget"
            )
            print("   Proceeding without context to avoid model errors")
            return original_content

        # Truncate context content to fit token budget
        context_tokens = self._count_tokens(context_content)
        if context_tokens > context_budget_tokens:
            # Binary search to find the right truncation point
            truncated_context = self._truncate_to_token_limit(
                context_content, context_budget_tokens - 50
            )  # -50 for truncation message
            truncation_msg = "\n\n[... Context truncated due to token limits ...]"
            truncated_context += truncation_msg

            final_tokens = self._count_tokens(truncated_context)
            print(
                f"⚠️  Warning: Context truncated from {context_tokens} to {final_tokens} tokens"
            )
            print(
                f"   Available context budget: {context_budget_tokens} tokens, Model limit: {max_context_length} tokens"
            )

            return f"""{truncated_context}

## CONTENT TO REVIEW
{original_content}"""

        return combined_content

    def format_results(
        self,
        result: ConversationResult,
        include_content: bool = True,
        include_context: bool = False,
    ) -> str:
        """Format conversation results for display as clean markdown.

        Args:
            result: ConversationResult to format
            include_content: Whether to include the original content
            include_context: Whether to include the context results from contextualizers

        Returns:
            Formatted markdown string
        """
        output = []

        # Header
        output.append("# Review-Crew Analysis Results")
        output.append("")
        output.append(
            f"**Analysis completed:** {result.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}"
        )
        output.append("")

        if include_content:
            output.append("## Content Reviewed")
            output.append("")
            # Format content with proper markdown
            content_lines = result.content.split("\n")
            for line in content_lines:
                if line.strip():
                    output.append(line)
                else:
                    output.append("")
            output.append("")

        # Summary
        successful_reviews = [r for r in result.reviews if not r.error]
        failed_reviews = [r for r in result.reviews if r.error]

        output.append("## Summary")
        output.append("")
        output.append(f"- **Total Reviews:** {len(result.reviews)}")
        output.append(f"- **Successful:** {len(successful_reviews)} ✅")
        if failed_reviews:
            output.append(f"- **Failed:** {len(failed_reviews)} ❌")
        if result.context_results:
            output.append(
                f"- **Context Results:** {len(result.context_results)} contextualizers 🔍"
            )
        if result.analysis_results:
            output.append(
                f"- **Analysis Results:** {len(result.analysis_results)} analyzers 🧠"
            )
        if hasattr(result, "analysis_errors") and result.analysis_errors:
            output.append(
                f"- **Analysis Errors:** {len(result.analysis_errors)} analyzers failed ⚠️"
            )
        output.append("")

        # Context Results Section
        if include_context and result.context_results:
            output.append("## Context Information")
            output.append("")
            output.append(
                "The following context was processed by contextualizers and provided to reviewers:"
            )
            output.append("")

            for i, context_result in enumerate(result.context_results, 1):
                # Try to find the corresponding context agent to get its name
                context_agent_name = f"Contextualizer {i}"
                if i <= len(self.context_agents) and self.context_agents[i - 1].persona:
                    context_agent_name = self.context_agents[i - 1].persona.name

                output.append(f"### {i}. {context_agent_name}")
                output.append("")
                output.append(f"**Summary:** {context_result.context_summary}")
                output.append("")
                output.append("**Formatted Context:**")
                output.append("")
                output.append("```")
                output.append(context_result.formatted_context)
                output.append("```")
                output.append("")
                if i < len(
                    result.context_results
                ):  # Don't add separator after last context
                    output.append("---")
                    output.append("")

        # Individual Reviews
        if successful_reviews:
            output.append("## Individual Reviews")
            output.append("")

            for i, review in enumerate(successful_reviews, 1):
                output.append(f"### {i}. {review.agent_name}")
                output.append(f"**Role:** {review.agent_role}")
                output.append("")

                # Extract clean text from feedback (handle both string and dict formats)
                clean_feedback = self._extract_clean_feedback(review.feedback)

                output.append(clean_feedback)
                output.append("")
                output.append("---")
                output.append("")

        # Analysis Section
        if result.analysis_results:
            output.append("## Analysis & Synthesis")
            output.append("")

            for i, analysis in enumerate(result.analysis_results, 1):
                # Try to get the analyzer name from the analysis or use a generic name
                analyzer_name = f"Analyzer {i}"
                # If we can identify the analyzer, use its name
                if hasattr(analysis, "analyzer_name"):
                    analyzer_name = analysis.analyzer_name
                elif (
                    i <= len(self.analysis_agents)
                    and self.analysis_agents[i - 1].persona
                ):
                    analyzer_name = self.analysis_agents[i - 1].persona.name

                output.append(f"### {analyzer_name}")
                output.append("")
                output.append(analysis.synthesis)
                output.append("")
                if i < len(
                    result.analysis_results
                ):  # Don't add separator after last analysis
                    output.append("---")
                    output.append("")

        # Failed Reviews
        if failed_reviews:
            output.append("## Failed Reviews")
            output.append("")
            for review in failed_reviews:
                output.append(f"- **{review.agent_name}:** {review.error}")
            output.append("")

        # Analysis Errors
        if hasattr(result, "analysis_errors") and result.analysis_errors:
            output.append("## Analysis Errors")
            output.append("")
            output.append(
                "The following analyzers failed but reviews were completed successfully:"
            )
            output.append("")
            for error in result.analysis_errors:
                output.append(f"- **{error}**")
            output.append("")
            output.append(
                "*Note: Reviews are still available above even though analysis failed.*"
            )
            output.append("")

        return "\n".join(output)

    def _extract_clean_feedback(self, feedback) -> str:
        """Extract clean text from feedback, handling various formats."""
        import json
        import ast

        if isinstance(feedback, str):
            # Try to parse as JSON/dict if it looks like one
            if feedback.strip().startswith("{") and feedback.strip().endswith("}"):
                try:
                    # Try JSON first
                    parsed = json.loads(feedback)
                    return self._extract_clean_feedback(parsed)
                except json.JSONDecodeError:
                    try:
                        # Try literal_eval for Python dict format
                        parsed = ast.literal_eval(feedback)
                        return self._extract_clean_feedback(parsed)
                    except (ValueError, SyntaxError):
                        # If parsing fails, return as-is
                        return feedback
            else:
                # If it's already a plain string, return as-is
                return feedback
        elif isinstance(feedback, dict):
            # Handle dictionary format (like from API responses)
            if "role" in feedback and "content" in feedback:
                content = feedback["content"]
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "text" in content[0]:
                        return content[0]["text"]
                    else:
                        return str(content[0])
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)
            elif "text" in feedback:
                return feedback["text"]
            else:
                return str(feedback)
        else:
            # Fallback to string representation
            return str(feedback)

    # Removed college-specific supplemental context methods - analysis personas now handle context generation automatically
