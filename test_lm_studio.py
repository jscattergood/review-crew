#!/usr/bin/env python3
"""
Test script for LM Studio integration.

This script demonstrates how to configure Review-Crew to work with LM Studio.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_lm_studio_setup():
    """Test LM Studio configuration."""
    print("🎭 Testing LM Studio Integration")
    print("=" * 40)
    
    try:
        from src.config.persona_loader import PersonaLoader
        from src.agents.review_agent import ReviewAgent
        from src.agents.conversation_manager import ConversationManager
        
        print("✅ All imports successful")
        
        # Test with LM Studio provider
        print("\n🔧 Testing LM Studio provider configuration...")
        
        # Load a persona
        loader = PersonaLoader()
        personas = loader.load_all_personas()
        
        if personas:
            test_persona = personas[0]  # Use first persona
            print(f"✅ Using persona: {test_persona.name}")
            
            # Create agent with LM Studio provider
            agent = ReviewAgent(
                test_persona, 
                model_provider='lm_studio',
                model_config_override={
                    'base_url': 'http://localhost:1234/v1',
                    'model_id': 'local-model'
                }
            )
            print(f"✅ Created LM Studio ReviewAgent")
            
            # Test conversation manager with LM Studio
            manager = ConversationManager(
                model_provider='lm_studio',
                model_config={
                    'base_url': 'http://localhost:1234/v1',
                    'model_id': 'local-model'
                }
            )
            print(f"✅ Created ConversationManager with LM Studio provider")
            
            available_agents = manager.get_available_agents()
            print(f"✅ Available agents: {len(available_agents)}")
            
            for agent_info in available_agents:
                print(f"  - {agent_info['name']} ({agent_info['role']})")
        
        print("\n🎉 LM Studio configuration test complete!")
        print("\n📝 To use LM Studio:")
        print("1. Start LM Studio and load a model")
        print("2. Enable the local server (default: http://localhost:1234)")
        print("3. Run reviews with: --provider lm_studio")
        print("   Example: python -m src.cli.main review 'test content' --provider lm_studio")
        
        print("\n💡 CLI Examples:")
        print("   # Use LM Studio with default settings")
        print("   python -m src.cli.main review 'Hello world' --provider lm_studio")
        print("")
        print("   # Use LM Studio with custom URL")
        print("   python -m src.cli.main review 'Hello world' --provider lm_studio --model-url http://localhost:1234/v1")
        print("")
        print("   # Use specific model ID")
        print("   python -m src.cli.main review 'Hello world' --provider lm_studio --model-id 'my-model'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lm_studio_setup()
    
    if not success:
        print("\n❌ LM Studio setup test failed.")
        sys.exit(1)
