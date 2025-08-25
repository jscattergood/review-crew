"""
Main CLI interface for Review-Crew.

This module provides the command-line interface for running multi-agent reviews.
"""

import click
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from ..agents.conversation_manager import ConversationManager
from ..config.persona_loader import PersonaLoader


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Review-Crew: Multi-agent content review system."""
    pass


@cli.command()
@click.argument('content', type=str, required=False)
@click.option('--agents', '-a', multiple=True, help='Specific agents to use (default: all)')
@click.option('--async-mode/--sync-mode', default=False, help='Run reviews asynchronously')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
@click.option('--no-content', is_flag=True, help='Hide original content in output')
@click.option('--provider', '-p', default='bedrock', type=click.Choice(['bedrock', 'lm_studio', 'ollama']), help='Model provider to use')
@click.option('--model-url', help='Custom model URL (for LM Studio or Ollama)')
@click.option('--model-id', help='Custom model ID')
@click.option('--no-analysis', is_flag=True, help='Disable analysis of reviews (skip analyzer personas)')
@click.option('--context', type=click.Path(exists=True), help='Path to context file to be processed by contextualizer')
@click.option('--contextualizer', help='Name of contextualizer persona to use for context processing')
@click.option('--max-context-length', type=int, help='Maximum context length for analysis (default: 4096, enables chunking if exceeded)')
def review(content: Optional[str], agents: tuple, async_mode: bool, output: Optional[str], no_content: bool, provider: str, model_url: Optional[str], model_id: Optional[str], no_analysis: bool, context: Optional[str], contextualizer: Optional[str], max_context_length: Optional[int]):
    """Review content with multiple AI agents.
    
    CONTENT can be either text content, a file path, or piped from stdin.
    If no CONTENT is provided, reads from stdin.
    """
    # Handle stdin input if no content provided
    from_stdin = False
    if content is None:
        if not sys.stdin.isatty():
            # Reading from stdin (piped input)
            content = sys.stdin.read()
            if not content.strip():
                click.echo("❌ No content received from stdin", err=True)
                return
            click.echo("📥 Reading content from stdin...")
            from_stdin = True
        else:
            # Interactive mode, no stdin
            click.echo("❌ No content provided. Use: python -m src.cli.main review 'content' or pipe content via stdin", err=True)
            return
    
    # Check if content is a file path (skip if from stdin)
    if not from_stdin:
        content_path = Path(content)
        if content_path.exists() and content_path.is_file():
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    content_text = f.read()
                click.echo(f"📁 Reading content from: {content_path}")
            except Exception as e:
                click.echo(f"❌ Error reading file {content_path}: {e}", err=True)
                return
        else:
            content_text = content
    else:
        # Content is from stdin, use it directly
        content_text = content
    
    if not content_text.strip():
        click.echo("❌ No content provided for review", err=True)
        return
    
    # Build model configuration
    model_config = {}
    if model_url:
        model_config['base_url'] = model_url
    if model_id:
        model_config['model_id'] = model_id
    # Set max_context_length with default value of 4096
    model_config['max_context_length'] = max_context_length or 4096
    
    # Initialize conversation manager
    try:
        manager = ConversationManager(
            model_provider=provider,
            model_config=model_config,
            enable_analysis=not no_analysis,
            contextualizer_persona=contextualizer
        )
    except Exception as e:
        click.echo(f"❌ Error initializing conversation manager: {e}", err=True)
        return
    
    # Convert agents tuple to list
    selected_agents = list(agents) if agents else None
    
    # Read context file if provided
    context_data = None
    if context:
        try:
            with open(context, 'r', encoding='utf-8') as f:
                context_data = f.read()
            click.echo(f"📄 Loaded context from: {context}")
        except Exception as e:
            click.echo(f"⚠️  Failed to read context file: {e}", err=True)
    
    # Run the review
    try:
        if async_mode:
            click.echo("🚀 Running async review...")
            result = asyncio.run(manager.run_review_async(content_text, context_data, selected_agents))
        else:
            click.echo("🚀 Running sync review...")
            result = manager.run_review(content_text, context_data, selected_agents)
        
        # Format and display results
        formatted_output = manager.format_results(result, include_content=not no_content)
        click.echo(formatted_output)
        
        # Analysis output now includes any context generation defined in analyzer personas
        
        # Save to file if requested
        if output:
            try:
                output_content = formatted_output
                
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                click.echo(f"💾 Results saved to: {output}")
            except Exception as e:
                click.echo(f"❌ Error saving to {output}: {e}", err=True)
        
    except Exception as e:
        click.echo(f"❌ Error during review: {e}", err=True)


@cli.command()
def agents():
    """List available review agents."""
    try:
        manager = ConversationManager()
        available_agents = manager.get_available_agents()
        
        if not available_agents:
            click.echo("❌ No review agents available. Run 'make setup' to configure personas.")
            return
        
        click.echo("🎭 Available Review Agents:")
        click.echo("=" * 40)
        
        for agent_info in available_agents:
            click.echo(f"👤 {agent_info['name']}")
            click.echo(f"   Role: {agent_info['role']}")
            click.echo(f"   Goal: {agent_info['goal']}")
            click.echo(f"   Temperature: {agent_info['temperature']}")
            click.echo(f"   Max Tokens: {agent_info['max_tokens']}")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error listing agents: {e}", err=True)


@cli.command()
@click.argument('content', type=str)
@click.option('--agent', '-a', required=True, help='Specific agent to use')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
def single(content: str, agent: str, output: Optional[str]):
    """Review content with a single agent.
    
    CONTENT can be either text content or a file path.
    """
    # Check if content is a file path
    content_path = Path(content)
    if content_path.exists() and content_path.is_file():
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_text = f.read()
            click.echo(f"📁 Reading content from: {content_path}")
        except Exception as e:
            click.echo(f"❌ Error reading file {content_path}: {e}", err=True)
            return
    else:
        content_text = content
    
    if not content_text.strip():
        click.echo("❌ No content provided for review", err=True)
        return
    
    # Initialize conversation manager
    try:
        manager = ConversationManager()
    except Exception as e:
        click.echo(f"❌ Error initializing conversation manager: {e}", err=True)
        return
    
    # Run single agent review
    try:
        click.echo(f"🚀 Running review with {agent}...")
        result = manager.run_review(content_text, [agent])
        
        # Format and display results
        formatted_output = manager.format_results(result, include_content=True)
        click.echo(formatted_output)
        
        # Save to file if requested
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                click.echo(f"💾 Results saved to: {output}")
            except Exception as e:
                click.echo(f"❌ Error saving to {output}: {e}", err=True)
        
    except Exception as e:
        click.echo(f"❌ Error during review: {e}", err=True)


# Removed college-specific context command - analysis personas now handle context generation automatically


@cli.command()
def status():
    """Show Review-Crew status and configuration."""
    try:
        # Check persona loader
        loader = PersonaLoader()
        config_info = loader.get_config_info()
        reviewers = loader.load_reviewer_personas()
        analyzers = loader.load_analyzer_personas()
        
        click.echo("📊 Review-Crew Status")
        click.echo("=" * 30)
        click.echo(f"✅ Reviewer personas loaded: {len(reviewers)}")
        click.echo(f"✅ Analyzer personas loaded: {len(analyzers)}")
        
        # Check conversation manager
        manager = ConversationManager()
        agents = manager.get_available_agents()
        click.echo(f"✅ Agents available: {len(agents)}")
        
        click.echo("\n🎭 Configured Reviewer Personas:")
        for persona in reviewers:
            click.echo(f"  - {persona.name} ({persona.role})")
        
        click.echo("\n🔍 Configured Analyzer Personas:")
        for persona in analyzers:
            click.echo(f"  - {persona.name} ({persona.role})")
        
        click.echo(f"\n📁 Personas Directory:")
        status_icon = "✅" if config_info['personas_dir_exists'] else "❌"
        source_info = ""
        if config_info['is_using_env_var']:
            source_info = " (from env var)"
        elif config_info['is_default_examples']:
            source_info = " (default - examples)"
        
        click.echo(f"  {config_info['personas_dir']} {status_icon}{source_info}")
        
        # Show environment variable if set
        if config_info['env_personas_dir']:
            click.echo(f"\n🌍 Environment Variable:")
            click.echo(f"  REVIEW_CREW_PERSONAS_DIR: {config_info['env_personas_dir']}")
        else:
            click.echo(f"\n💡 Tip: Create .env file or set REVIEW_CREW_PERSONAS_DIR to use custom personas")
        
    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)


if __name__ == '__main__':
    cli()
