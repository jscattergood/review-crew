#!/usr/bin/env python3
"""
Test script to run realistic reviews with the Review-Crew system.

This script demonstrates the multi-agent review system with realistic content
that will trigger different types of feedback from each agent.

Usage:
    python test_review.py                    # Interactive mode
    python test_review.py --file 1          # Test with Python code
    python test_review.py --file 2          # Test with HTML page  
    python test_review.py --file 3          # Test with API docs
    python test_review.py --provider lm_studio --file 1  # Non-interactive
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_test_reviews(provider_choice=None, file_choice=None):
    """Run test reviews with realistic content."""
    print("🎭 Review-Crew System Test")
    print("=" * 50)
    
    try:
        from src.agents.conversation_manager import ConversationManager
        
        # Test files
        test_files = [
            ("Python Code", "test_inputs/user_registration.py"),
            ("HTML Page", "test_inputs/product_page.html"), 
            ("API Documentation", "test_inputs/api_documentation.md")
        ]
        
        print("📁 Available test files:")
        for i, (name, path) in enumerate(test_files, 1):
            print(f"  {i}. {name} ({path})")
        
        # Use provided choices or ask interactively
        if provider_choice is None:
            print("\n🔧 Choose test configuration:")
            print("1. AWS Bedrock (default)")
            print("2. LM Studio (local)")
            print("3. Test mode (no actual LLM calls)")
            
            choice = input("\nSelect provider (1-3, default=3): ").strip()
        else:
            choice = str(provider_choice)
            print(f"\n🔧 Using provider choice: {choice}")
        
        if choice == "1":
            provider = "bedrock"
            print("⚠️  Note: Requires AWS credentials and will incur costs")
        elif choice == "2":
            provider = "lm_studio"
            print("⚠️  Note: Requires LM Studio running on localhost:1234")
        else:
            provider = "test"
            print("✅ Using test mode (no LLM calls)")
        
        # Select test file
        if file_choice is None:
            print("\n📁 Available test files:")
            for i, (name, path) in enumerate(test_files, 1):
                print(f"  {i}. {name} ({path})")
            file_input = input("\nSelect test file (1-3, default=1): ").strip()
        else:
            file_input = str(file_choice)
            print(f"\n📁 Using test file choice: {file_input}")
        
        if file_input == "2":
            test_name, test_file = test_files[1]
        elif file_input == "3":
            test_name, test_file = test_files[2]
        else:
            test_name, test_file = test_files[0]
        
        print(f"\n🎯 Testing with: {test_name}")
        
        # Read test content
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"📄 Content length: {len(content)} characters")
        print(f"📄 Content preview: {content[:100]}...")
        
        if provider == "test":
            # Test mode - just show what would happen
            print("\n🧪 TEST MODE - Simulating reviews...")
            
            manager = ConversationManager()
            agents = manager.get_available_agents()
            
            print(f"\n✅ Would review with {len(agents)} agents:")
            for agent in agents:
                print(f"  👤 {agent['name']} ({agent['role']})")
                print(f"     Goal: {agent['goal'][:60]}...")
                print(f"     Temperature: {agent['temperature']}")
                print()
            
            print("💡 To run actual reviews:")
            print(f"   python -m src.cli.main review '{test_file}' --provider lm_studio")
            print(f"   make review-lm ARGS='\"$(cat {test_file})\"'")
            
        else:
            # Real review
            print(f"\n🚀 Running real review with {provider}...")
            
            if provider == "lm_studio":
                manager = ConversationManager(model_provider="lm_studio")
            else:
                manager = ConversationManager(model_provider="bedrock")
            
            # Run the review
            result = manager.run_review(content)
            
            # Display results
            formatted_output = manager.format_results(result, include_content=False)
            print("\n" + "="*60)
            print("REVIEW RESULTS")
            print("="*60)
            print(formatted_output)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_expected_feedback():
    """Show what kind of feedback to expect from each agent."""
    print("\n📋 Expected Feedback Types:")
    print("=" * 40)
    
    feedback_types = [
        ("👨‍💻 Technical Reviewer", [
            "Security vulnerabilities (MD5 hashing, no input sanitization)",
            "Code quality issues (global variables, error handling)",
            "Architecture problems (in-memory storage, debug mode)",
            "Best practices violations (hardcoded secrets, no logging)"
        ]),
        ("🎨 UX Reviewer", [
            "Accessibility issues (no alt text, poor color contrast)",
            "User experience problems (confusing forms, aggressive tactics)",
            "Mobile responsiveness concerns",
            "Trust and credibility issues (scammy design patterns)"
        ]),
        ("📝 Content Reviewer", [
            "Misleading claims and false advertising",
            "Poor documentation clarity and completeness",
            "Inappropriate tone and messaging",
            "Missing important information and disclaimers"
        ]),
        ("🔒 Security Reviewer", [
            "Critical security flaws (password exposure, no authentication)",
            "Data privacy violations (collecting SSN, no encryption)",
            "Compliance issues (GDPR, PCI DSS violations)",
            "Attack vectors (XSS, injection vulnerabilities)"
        ])
    ]
    
    for agent_name, issues in feedback_types:
        print(f"\n{agent_name}:")
        for issue in issues:
            print(f"  • {issue}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Test Review-Crew with realistic content")
    parser.add_argument("--provider", choices=["1", "2", "3"], help="Provider choice: 1=Bedrock, 2=LM Studio, 3=Test mode")
    parser.add_argument("--file", choices=["1", "2", "3"], help="File choice: 1=Python, 2=HTML, 3=Markdown")
    
    args = parser.parse_args()
    
    print("🎭 Review-Crew Realistic Test Suite")
    print("=" * 50)
    
    success = run_test_reviews(args.provider, args.file)
    
    if success:
        show_expected_feedback()
        print("\n🎉 Test complete!")
        print("\n💡 Pro tip: Try running the same content with different providers")
        print("   to compare response quality and speed!")
    else:
        print("\n❌ Test failed. Check your setup with 'make test'")
        sys.exit(1)

if __name__ == "__main__":
    main()
