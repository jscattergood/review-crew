"""
Persona configuration loader for Review-Crew.

This module handles loading and validating persona configurations from YAML files.
Supports loading from a configurable personas directory.

Environment Variables:
    REVIEW_CREW_PERSONAS_DIR: Directory for persona configurations
                              (defaults to examples/personas for testing)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    model_config: Dict[str, Any]

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

    def __init__(self, project_root: Optional[Path] = None, personas_dir: Optional[Path] = None):
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
        elif os.getenv('REVIEW_CREW_PERSONAS_DIR'):
            self.personas_dir = Path(os.getenv('REVIEW_CREW_PERSONAS_DIR'))
        else:
            # Default to examples for testing - production should use .env file
            self.personas_dir = self.project_root / "examples" / "personas"
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'project_root': str(self.project_root),
            'personas_dir': str(self.personas_dir),
            'personas_dir_exists': self.personas_dir.exists(),
            'env_personas_dir': os.getenv('REVIEW_CREW_PERSONAS_DIR'),
            'is_using_env_var': bool(os.getenv('REVIEW_CREW_PERSONAS_DIR')),
            'is_default_examples': not bool(os.getenv('REVIEW_CREW_PERSONAS_DIR')),
        }

    def load_persona(self, filepath: Path) -> PersonaConfig:
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
        if not filepath.exists():
            raise FileNotFoundError(f"Persona file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {filepath}: {e}")

        # Provide defaults for optional fields
        data.setdefault("model_config", {})
        data["model_config"].setdefault("temperature", 0.3)
        data["model_config"].setdefault("max_tokens", 1500)

        try:
            return PersonaConfig(**data)
        except TypeError as e:
            raise ValueError(f"Invalid persona configuration in {filepath}: {e}")

    def load_all_personas(self) -> List[PersonaConfig]:
        """Load all persona configurations from the configured directory.

        Returns:
            List of PersonaConfig objects
        """
        if not self.personas_dir.exists():
            raise ValueError(
                f"Personas directory not found: {self.personas_dir}\n"
                f"Set REVIEW_CREW_PERSONAS_DIR environment variable or create the directory."
            )

        personas = self._load_personas_from_dir(self.personas_dir)
        
        if personas:
            env_info = " (from env var)" if os.getenv('REVIEW_CREW_PERSONAS_DIR') else " (default)"
            print(f"✅ Loaded {len(personas)} personas from {self.personas_dir}{env_info}")
        
        if not personas:
            raise ValueError(
                f"No persona configurations found in {self.personas_dir}\n"
                f"Add .yaml files to this directory or check REVIEW_CREW_PERSONAS_DIR"
            )

        return personas

    def _load_personas_from_dir(self, directory: Path) -> List[PersonaConfig]:
        """Load all persona configurations from a directory.

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

    def get_persona_by_name(self, name: str) -> Optional[PersonaConfig]:
        """Get a specific persona by name.

        Args:
            name: Name of the persona to find

        Returns:
            PersonaConfig if found, None otherwise
        """
        personas = self.load_all_personas()
        for persona in personas:
            if persona.name.lower() == name.lower():
                return persona
        return None

    def list_available_personas(self) -> List[str]:
        """Get a list of all available persona names.

        Returns:
            List of persona names
        """
        personas = self.load_all_personas()
        return [persona.name for persona in personas]

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
        personas = loader.load_all_personas()
        print(f"Loaded {len(personas)} personas:")
        for persona in personas:
            print(f"  - {persona.name} ({persona.role})")
    except Exception as e:
        print(f"Error loading personas: {e}")
