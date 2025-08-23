#!/usr/bin/env python3
"""
Simple test script for demonstrating persona loading.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config.persona_loader import PersonaLoader
    
    print("🚀 Running configuration example...")
    print("Testing persona loading and validation...")
    print()
    
    loader = PersonaLoader()
    print(f"Project root: {loader.project_root}")
    print(f"Examples dir: {loader.examples_dir}")
    print(f"Config dir: {loader.config_dir}")
    print()
    
    try:
        personas = loader.load_all_personas()
        print(f"✅ Loaded {len(personas)} personas:")
        for p in personas:
            print(f"  - {p.name}")
            print(f"    Role: {p.role}")
            print(f"    Goal: {p.goal[:50]}...")
            print(f"    Temperature: {p.model_config.get('temperature', 'default')}")
            print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try running: make setup")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure dependencies are installed: make install-pip")
    sys.exit(1)
