#!/usr/bin/env python3
"""
Test script for demonstrating conversation functionality.

This script shows how to run conversations without needing LLM access.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_conversation_setup():
    """Test that conversation components can be imported and initialized."""
    print("🧪 Testing conversation setup...")
    
    try:
        from src.config.persona_loader import PersonaLoader
        from src.agents.review_agent import ReviewAgent
        from src.agents.conversation_manager import ConversationManager
        
        print("✅ All imports successful")
        
        # Test persona loading
        loader = PersonaLoader()
        personas = loader.load_all_personas()
        print(f"✅ Loaded {len(personas)} personas")
        
        # Test agent creation (without actually calling LLM)
        if personas:
            test_persona = personas[0]
            print(f"✅ Test persona: {test_persona.name}")
            
            # Create agent (this won't call LLM yet)
            agent = ReviewAgent(test_persona)
            print(f"✅ Created ReviewAgent for {agent.persona.name}")
            
            # Test agent info
            info = agent.get_info()
            print(f"✅ Agent info: {info['name']} - {info['role']}")
        
        # Test conversation manager
        manager = ConversationManager()
        available_agents = manager.get_available_agents()
        print(f"✅ ConversationManager loaded {len(available_agents)} agents")
        
        for agent_info in available_agents:
            print(f"  - {agent_info['name']} ({agent_info['role']})")
        
        print("\n🎉 All conversation components working!")
        print("\n📝 To run actual reviews, you'll need to:")
        print("1. Configure your LLM provider (LM Studio, AWS Bedrock, etc.)")
        print("2. Use the CLI: python -m src.cli.main review 'your content here'")
        print("3. Or use make commands: make review ARGS='\"your content\"'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1. CLI Commands:")
    print("   python -m src.cli.main agents                    # List agents")
    print("   python -m src.cli.main review 'Hello world'     # Review text")
    print("   python -m src.cli.main review file.txt          # Review file")
    print("   python -m src.cli.main single 'text' -a 'Technical Reviewer'  # Single agent")
    
    print("\n2. Makefile Commands:")
    print("   make agents                                      # List agents")
    print("   make review ARGS='\"Hello world\"'               # Review text")
    print("   make cli-status                                  # Show status")
    
    print("\n3. Advanced Options:")
    print("   --async-mode          # Run reviews concurrently")
    print("   --agents agent1,agent2  # Use specific agents")
    print("   --output results.txt  # Save to file")
    print("   --no-content         # Hide original content")

if __name__ == "__main__":
    print("🎭 Review-Crew Conversation Test")
    print("=" * 40)
    
    success = test_conversation_setup()
    
    if success:
        show_usage_examples()
    else:
        print("\n❌ Setup failed. Check your configuration with 'make test'")
        sys.exit(1)
