#!/usr/bin/env python3
"""
Setup script for Review-Crew personas.

This script helps users set up their custom persona configurations
by copying examples and providing guidance.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.persona_loader import PersonaLoader


def main():
    """Main setup function."""
    print("🎭 Review-Crew Persona Setup")
    print("=" * 40)

    loader = PersonaLoader()

    # Check if custom config already exists
    if loader.config_dir.exists() and list(loader.config_dir.glob("*.yaml")):
        print(f"✅ Custom personas already exist in {loader.config_dir}")

        # List existing personas
        try:
            personas = loader.load_all_personas(use_examples=False, use_custom=True)
            print(f"\nFound {len(personas)} custom personas:")
            for persona in personas:
                print(f"  - {persona.name} ({persona.role})")
        except Exception as e:
            print(f"⚠️  Error loading existing personas: {e}")

        response = input("\nDo you want to reset and copy examples again? (y/N): ")
        if response.lower() != "y":
            print("Setup cancelled.")
            return

    # Set up custom config
    try:
        print(f"\n📁 Setting up custom personas in {loader.config_dir}")
        loader.setup_custom_config()

        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print(f"1. Edit files in {loader.config_dir} to customize your reviewers")
        print("2. Add or remove persona files as needed")
        print("3. Run 'python -m src.config.persona_loader' to test your configuration")
        print("\n💡 Tip: The config/personas/ directory is gitignored for privacy")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
