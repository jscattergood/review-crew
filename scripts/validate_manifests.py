#!/usr/bin/env python3
"""
Manifest validation script using the JSON Schema.
Validates all manifest.yaml files in the input/ directory against manifest.schema.json
"""

import json
import sys
from pathlib import Path

try:
    import jsonschema
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv add pyyaml jsonschema")
    sys.exit(1)


def validate_manifest(manifest_path: Path, schema_path: Path) -> bool:
    """Validate a manifest file against the schema."""
    try:
        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)

        # Load manifest
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # Validate
        jsonschema.validate(manifest, schema)
        print(f"✅ {manifest_path.name} is valid!")
        return True

    except jsonschema.ValidationError as e:
        print(f"❌ {manifest_path.name} validation error:")
        print(f"   {e.message}")
        if e.absolute_path:
            print(f"   Path: {'.'.join(str(p) for p in e.absolute_path)}")
        return False

    except Exception as e:
        print(f"❌ Error validating {manifest_path.name}: {e}")
        return False


def main():
    """Validate all manifest files."""
    # Schema path relative to project root
    project_root = Path(__file__).parent.parent
    schema_path = project_root / "manifest.schema.json"

    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)

    # Find all manifest files
    input_dir = project_root / "input"
    manifest_files = list(input_dir.glob("*/manifest.yaml"))

    if not manifest_files:
        print("No manifest files found in input/ directories")
        sys.exit(0)

    print(
        f"Validating {len(manifest_files)} manifest files against {schema_path.name}...\n"
    )

    all_valid = True
    for manifest_file in sorted(manifest_files):
        valid = validate_manifest(manifest_file, schema_path)
        all_valid = all_valid and valid

    print(f"\nValidation {'completed successfully' if all_valid else 'failed'}!")
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
