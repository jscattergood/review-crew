"""
Persona configuration loader for Review-Crew.

This module handles loading and validating persona configurations from YAML files.
Supports loading from a configurable personas directory.

Environment Variables:
    REVIEW_CREW_PERSONAS_DIR: Directory for persona configurations
                              (defaults to examples/personas for testing)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


@dataclass
class PersonaConfig:
    """Configuration for a review persona."""

    name: str
    role: str
    goal: str
    backstory: str
    prompt_template: str
    model_config: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Persona name is required")
        if not self.role:
            raise ValueError("Persona role is required")
        if not self.goal:
            raise ValueError("Persona goal is required")
        if not self.prompt_template:
            raise ValueError("Persona prompt_template is required")
        if "{content}" not in self.prompt_template:
            raise ValueError("Prompt template must contain {content} placeholder")


class PersonaLoader:
    """Loads and manages persona configurations."""

    def __init__(
        self, project_root: Path | None = None, personas_dir: Path | None = None
    ):
        """Initialize the persona loader.

        Args:
            project_root: Path to project root. If None, auto-detects from current file.
            personas_dir: Custom personas directory. Overrides environment variable and defaults.
        """
        if project_root is None:
            # Auto-detect project root (assumes this file is in src/config/)
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)

        # Set personas directory with priority: parameter > env var > default (examples)
        if personas_dir:
            self.personas_dir = Path(personas_dir)
        elif os.getenv("REVIEW_CREW_PERSONAS_DIR"):
            self.personas_dir = Path(os.getenv("REVIEW_CREW_PERSONAS_DIR"))
        else:
            # Default to examples for testing - production should use .env file
            self.personas_dir = self.project_root / "examples" / "personas"

    def get_config_info(self) -> dict[str, Any]:
        """Get current configuration information.

        Returns:
            Dictionary with configuration details
        """
        return {
            "project_root": str(self.project_root),
            "personas_dir": str(self.personas_dir),
            "personas_dir_exists": self.personas_dir.exists(),
            "env_personas_dir": os.getenv("REVIEW_CREW_PERSONAS_DIR"),
            "is_using_env_var": bool(os.getenv("REVIEW_CREW_PERSONAS_DIR")),
            "is_default_examples": not bool(os.getenv("REVIEW_CREW_PERSONAS_DIR")),
        }

    def load_persona(self, filepath: str | Path) -> PersonaConfig:
        """Load a single persona configuration from a YAML file.

        Args:
            filepath: Path to the persona YAML file

        Returns:
            PersonaConfig object

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValueError: If the configuration is invalid
        """
        # Convert to Path object if it's a string
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Persona file not found: {filepath}")

        try:
            with open(filepath, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {filepath}: {e}") from e

        # Provide defaults for optional fields
        data.setdefault("model_config", {})
        data["model_config"].setdefault("temperature", 0.3)
        data["model_config"].setdefault("max_tokens", 1500)

        try:
            return PersonaConfig(**data)
        except TypeError as e:
            raise ValueError(f"Invalid persona configuration in {filepath}: {e}") from e

    def load_reviewer_personas(self) -> list[PersonaConfig]:
        """Load reviewer persona configurations from the reviewers directory.

        Returns:
            List of reviewer PersonaConfig objects
        """
        if not self.personas_dir.exists():
            raise ValueError(
                f"Personas directory not found: {self.personas_dir}\n"
                f"Set REVIEW_CREW_PERSONAS_DIR environment variable or create the directory."
            )

        # Load from reviewers subfolder
        reviewers_dir = self.personas_dir / "reviewers"
        if not reviewers_dir.exists():
            raise ValueError(
                f"Reviewers directory not found: {reviewers_dir}\n"
                f"Expected folder structure: {self.personas_dir}/reviewers/ and {self.personas_dir}/analyzers/"
            )

        personas = self._load_personas_from_dir(reviewers_dir)

        if personas:
            env_info = (
                " (from env var)"
                if os.getenv("REVIEW_CREW_PERSONAS_DIR")
                else " (default)"
            )
            print(
                f"✅ Loaded {len(personas)} reviewer personas from {self.personas_dir}{env_info}"
            )

        if not personas:
            raise ValueError(
                f"No persona configurations found in {self.personas_dir}\n"
                f"Add .yaml files to this directory or check REVIEW_CREW_PERSONAS_DIR"
            )

        return personas

    def load_analyzer_personas(self) -> list[PersonaConfig]:
        """Load analyzer persona configurations.

        Returns:
            List of analyzer PersonaConfig objects
        """
        analyzers_dir = self.personas_dir / "analyzers"
        if analyzers_dir.exists():
            return self._load_personas_from_dir(analyzers_dir)
        else:
            return []

    def load_contextualizer_personas(self) -> list[PersonaConfig]:
        """Load contextualizer persona configurations.

        Returns:
            List of contextualizer PersonaConfig objects
        """
        contextualizers_dir = self.personas_dir / "contextualizers"
        if contextualizers_dir.exists():
            return self._load_personas_from_dir(contextualizers_dir)
        else:
            return []

    def load_all_persona_types(self) -> dict[str, list[PersonaConfig]]:
        """Load all persona types (reviewers, analyzers, and contextualizers) separately.

        Returns:
            Dictionary with 'reviewers', 'analyzers', and 'contextualizers' keys containing their respective personas
        """
        return {
            "reviewers": self.load_reviewer_personas(),
            "analyzers": self.load_analyzer_personas(),
            "contextualizers": self.load_contextualizer_personas(),
        }

    def _load_personas_from_dir(self, directory: Path) -> list[PersonaConfig]:
        """Load all persona configurations from a directory and its subdirectories.

        Args:
            directory: Directory containing persona YAML files

        Returns:
            List of PersonaConfig objects
        """
        personas = []

        # Load from main directory (flat structure - backwards compatibility)
        personas.extend(self._load_personas_from_single_dir(directory))

        # Load from subdirectories (organized structure)
        for subdirectory in directory.iterdir():
            if subdirectory.is_dir():
                subdir_personas = self._load_personas_from_single_dir(subdirectory)
                personas.extend(subdir_personas)
                if subdir_personas:
                    print(
                        f"  ✓ Loaded {len(subdir_personas)} personas from '{subdirectory.name}' category"
                    )

        return personas

    def _load_personas_from_single_dir(self, directory: Path) -> list[PersonaConfig]:
        """Load persona configurations from a single directory (no subdirectories).

        Args:
            directory: Directory containing persona YAML files

        Returns:
            List of PersonaConfig objects
        """
        personas = []

        for filepath in directory.glob("*.yaml"):
            try:
                persona = self.load_persona(filepath)
                personas.append(persona)
            except Exception as e:
                print(f"Warning: Failed to load persona from {filepath}: {e}")
                continue

        # Also check for .yml files
        for filepath in directory.glob("*.yml"):
            try:
                persona = self.load_persona(filepath)
                personas.append(persona)
            except Exception as e:
                print(f"Warning: Failed to load persona from {filepath}: {e}")
                continue

        return personas

    def get_persona_by_name(self, name: str) -> PersonaConfig | None:
        """Get a specific persona by name.

        Args:
            name: Name of the persona to find

        Returns:
            PersonaConfig if found, None otherwise
        """
        personas = self.load_reviewer_personas()
        for persona in personas:
            if persona.name.lower() == name.lower():
                return persona
        return None

    def list_available_personas(self) -> list[str]:
        """Get a list of all available persona names.

        Returns:
            List of persona names
        """
        personas = self.load_reviewer_personas()
        return [persona.name for persona in personas]

    def load_reviewer_personas_by_category(
        self, categories: list[str]
    ) -> list[PersonaConfig]:
        """Load reviewer personas from specific categories (sub-folders).

        Args:
            categories: List of category folder names (e.g., ['academic', 'content'])

        Returns:
            List of PersonaConfig objects from specified categories
        """
        reviewers_dir = self.personas_dir / "reviewers"
        personas = []

        for category in categories:
            category_dir = reviewers_dir / category
            if category_dir.exists():
                category_personas = self._load_personas_from_single_dir(category_dir)
                personas.extend(category_personas)
                print(
                    f"✅ Loaded {len(category_personas)} reviewers from '{category}' category"
                )
            else:
                print(f"⚠️  Category '{category}' not found in {reviewers_dir}")

        return personas

    def load_reviewer_personas_by_names(self, names: list[str]) -> list[PersonaConfig]:
        """Load specific reviewer personas by name.

        Args:
            names: List of exact persona names to load

        Returns:
            List of PersonaConfig objects matching the names
        """
        all_personas = self.load_reviewer_personas()
        selected_personas = []

        for name in names:
            persona = next((p for p in all_personas if p.name == name), None)
            if persona:
                selected_personas.append(persona)
            else:
                print(f"⚠️  Reviewer '{name}' not found")

        return selected_personas

    def load_reviewers_from_manifest(
        self, manifest_config: dict[str, Any]
    ) -> list[PersonaConfig]:
        """Load reviewers based on manifest configuration.

        Args:
            manifest_config: Manifest configuration dictionary

        Returns:
            List of PersonaConfig objects based on manifest specification
        """
        personas = []

        # Load by categories
        if "reviewer_categories" in manifest_config:
            category_personas = self.load_reviewer_personas_by_category(
                manifest_config["reviewer_categories"]
            )
            personas.extend(category_personas)

        # Load by specific names
        if "reviewers" in manifest_config:
            name_personas = self.load_reviewer_personas_by_names(
                manifest_config["reviewers"]
            )
            personas.extend(name_personas)

        # Remove duplicates (in case same persona specified in both ways)
        unique_personas = []
        seen_names = set()
        for persona in personas:
            if persona.name not in seen_names:
                unique_personas.append(persona)
                seen_names.add(persona.name)

        return unique_personas

    def load_contextualizer_personas_by_names(
        self, names: list[str]
    ) -> list[PersonaConfig]:
        """Load specific contextualizer personas by name.

        Args:
            names: List of exact persona names to load

        Returns:
            List of PersonaConfig objects matching the specified names
        """
        all_contextualizers = self.load_contextualizer_personas()
        selected_contextualizers = []

        for name in names:
            found = False
            for persona in all_contextualizers:
                if persona.name == name:
                    selected_contextualizers.append(persona)
                    found = True
                    break
            if not found:
                print(f"⚠️  Contextualizer '{name}' not found")

        return selected_contextualizers

    def load_contextualizer_personas_by_category(
        self, categories: list[str]
    ) -> list[PersonaConfig]:
        """Load contextualizer personas from specific categories (sub-folders).

        Args:
            categories: List of category folder names (e.g., ['business', 'academic'])

        Returns:
            List of PersonaConfig objects from specified categories
        """
        contextualizers_dir = self.personas_dir / "contextualizers"
        personas = []

        for category in categories:
            category_dir = contextualizers_dir / category
            if category_dir.exists():
                category_personas = self._load_personas_from_single_dir(category_dir)
                personas.extend(category_personas)
                print(
                    f"✅ Loaded {len(category_personas)} contextualizers from '{category}' category"
                )
            else:
                print(f"⚠️  Category '{category}' not found in {contextualizers_dir}")

        return personas

    def load_contextualizers_from_manifest(
        self, manifest_config: dict[str, Any]
    ) -> list[PersonaConfig]:
        """Load contextualizers based on manifest configuration.

        Args:
            manifest_config: Manifest configuration dictionary

        Returns:
            List of PersonaConfig objects based on manifest specification
        """
        personas = []

        # Load by categories
        if "contextualizer_categories" in manifest_config:
            category_personas = self.load_contextualizer_personas_by_category(
                manifest_config["contextualizer_categories"]
            )
            personas.extend(category_personas)

        # Load by specific names
        if "contextualizers" in manifest_config:
            name_personas = self.load_contextualizer_personas_by_names(
                manifest_config["contextualizers"]
            )
            personas.extend(name_personas)

        # Remove duplicates (in case same persona specified in both ways)
        unique_personas = []
        seen_names = set()
        for persona in personas:
            if persona.name not in seen_names:
                unique_personas.append(persona)
                seen_names.add(persona.name)

        return unique_personas

    def load_analyzer_personas_by_names(self, names: list[str]) -> list[PersonaConfig]:
        """Load analyzer personas by specific names.

        Args:
            names: List of analyzer persona names to load

        Returns:
            List of PersonaConfig objects for analyzers
        """
        all_analyzer_personas = self.load_analyzer_personas()
        selected_personas = []

        for name in names:
            for persona in all_analyzer_personas:
                if persona.name == name:
                    selected_personas.append(persona)
                    break
            else:
                print(f"⚠️  Analyzer persona '{name}' not found")

        return selected_personas

    def load_analyzer_personas_by_category(
        self, categories: list[str]
    ) -> list[PersonaConfig]:
        """Load analyzer personas by categories.

        Args:
            categories: List of category names to load

        Returns:
            List of PersonaConfig objects for analyzers in specified categories
        """
        personas = []

        for category in categories:
            category_dir = self.personas_dir / "analyzers" / category
            if category_dir.exists():
                personas.extend(self._load_personas_from_directory(category_dir))
            else:
                print(f"⚠️  Analyzer category '{category}' not found at {category_dir}")

        return personas

    def load_analyzers_from_manifest(
        self, manifest_config: dict[str, Any]
    ) -> list[PersonaConfig]:
        """Load analyzers based on manifest configuration.

        Args:
            manifest_config: Manifest configuration dictionary

        Returns:
            List of PersonaConfig objects based on manifest specification
        """
        personas = []

        # Load by categories
        if "analyzer_categories" in manifest_config:
            category_personas = self.load_analyzer_personas_by_category(
                manifest_config["analyzer_categories"]
            )
            personas.extend(category_personas)

        # Load by specific names
        if "analyzers" in manifest_config:
            name_personas = self.load_analyzer_personas_by_names(
                manifest_config["analyzers"]
            )
            personas.extend(name_personas)

        # Remove duplicates (in case same persona specified in both ways)
        unique_personas = []
        seen_names = set()
        for persona in personas:
            if persona.name not in seen_names:
                unique_personas.append(persona)
                seen_names.add(persona.name)

        return unique_personas

    def setup_custom_config(self) -> None:
        """Set up the custom config directory by copying examples.

        This is a convenience method for first-time setup.
        """
        if self.config_dir.exists():
            print(f"Custom config directory already exists: {self.config_dir}")
            return

        if not self.examples_dir.exists():
            raise FileNotFoundError(
                f"Examples directory not found: {self.examples_dir}"
            )

        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Copy example files
        import shutil

        for example_file in self.examples_dir.glob("*.yaml"):
            target_file = self.config_dir / example_file.name
            shutil.copy2(example_file, target_file)
            print(f"Copied {example_file.name} to config/personas/")

        print(f"Custom personas setup complete in {self.config_dir}")
        print("Edit these files to customize your review personas.")


if __name__ == "__main__":
    # Example usage
    loader = PersonaLoader()

    try:
        # Load reviewers
        reviewers = loader.load_reviewer_personas()
        print(f"Loaded {len(reviewers)} reviewer personas:")
        for persona in reviewers:
            print(f"  - {persona.name} ({persona.role})")

        # Load analyzers
        analyzers = loader.load_analyzer_personas()
        print(f"\nLoaded {len(analyzers)} analyzer personas:")
        for persona in analyzers:
            print(f"  - {persona.name} ({persona.role})")
    except Exception as e:
        print(f"Error loading personas: {e}")
