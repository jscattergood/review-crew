"""
Main CLI interface for Review-Crew.

This module provides the command-line interface for running multi-agent reviews.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click

from ..config.persona_loader import PersonaLoader
from ..conversation.manager import ConversationManager
from ..logging.manager import LoggingManager


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Review-Crew: Multi-agent content review system."""
    pass


@cli.command()
@click.argument("content", type=str, required=False)
@click.option(
    "--agents", "-a", multiple=True, help="Specific agents to use (default: all)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save results to file (default: auto-generated timestamped directory in output/)",
)
@click.option("--no-content", is_flag=True, help="Hide original content in output")
@click.option(
    "--provider",
    "-p",
    default="bedrock",
    type=click.Choice(["bedrock", "lm_studio", "ollama"]),
    help="Model provider to use",
)
@click.option("--model-url", help="Custom model URL (for LM Studio or Ollama)")
@click.option("--model-id", help="Custom model ID")
@click.option(
    "--no-analysis",
    is_flag=True,
    help="Disable analysis of reviews (skip analyzer personas)",
)
@click.option(
    "--context",
    type=click.Path(exists=True),
    help="Path to context file to be processed by contextualizer (auto-selects first available contextualizer)",
)
@click.option(
    "--include-context",
    is_flag=True,
    help="Include context results from contextualizers in the output",
)
def review(
    content: str | None,
    agents: tuple,
    output: str | None,
    no_content: bool,
    provider: str,
    model_url: str | None,
    model_id: str | None,
    no_analysis: bool,
    context: str | None,
    include_context: bool,
) -> None:
    """Review content with multiple AI agents.

    CONTENT can be:
    - Text content directly
    - A file path (single document review)
    - A directory path (multi-document review)
    - Omitted to read from stdin

    For multi-document reviews:
    - Place a manifest.yaml in the directory to specify reviewer selection
    - Supports reviewer categories (academic, technical, content, business)
    - Manifest overrides --agents flag when present

    Examples:
      review "content text"              # Direct text review
      review essay.txt                   # Single file review
      review application-folder/         # Multi-document review
      echo "content" | review            # Stdin review
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
            click.echo(
                "❌ No content provided. Use: python -m src.cli.main review 'content' or pipe content via stdin",
                err=True,
            )
            return

    # Check if content is a file path or directory (skip if from stdin)
    if not from_stdin:
        content_path: Path = Path(content)
        if content_path.exists():
            if content_path.is_file():
                # Single file - existing behavior
                try:
                    with open(content_path, encoding="utf-8") as f:
                        content_text = f.read()
                    click.echo(f"📁 Reading content from: {content_path}")
                except Exception as e:
                    click.echo(f"❌ Error reading file {content_path}: {e}", err=True)
                    return
            elif content_path.is_dir():
                # Directory - new multi-document behavior
                click.echo(f"📂 Processing document collection from: {content_path}")

                # Check for manifest and provide info
                manifest_path = content_path / "manifest.yaml"
                if manifest_path.exists():
                    click.echo(
                        "📋 Found manifest file - will use custom reviewer selection"
                    )
                else:
                    click.echo(
                        "📄 No manifest found - will use all available reviewers"
                    )

                content_text = str(
                    content_path
                )  # Pass directory path for manager to handle
            else:
                click.echo(
                    f"❌ Path exists but is neither file nor directory: {content_path}",
                    err=True,
                )
                return
        else:
            # Not a path, treat as text content
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
        model_config["base_url"] = model_url
    if model_id:
        model_config["model_id"] = model_id

    # Note: Context length is now handled per-agent based on their model configuration

    # Initialize logging session
    logging_manager = LoggingManager.get_instance()
    content_info = (
        content_text[:100] + "..." if len(content_text) > 100 else content_text
    )
    session_id = logging_manager.start_session(
        content_info=content_info,
        selected_agents=list(agents) if agents else None,
        model_provider=provider,
        model_config=model_config,
    )
    click.echo(f"📝 Started logging session: {session_id}")

    # Initialize conversation manager
    try:
        manager = ConversationManager(
            model_provider=provider,
            model_config=model_config,
            enable_analysis=not no_analysis,
        )
    except Exception as e:
        click.echo(f"❌ Error initializing conversation manager: {e}", err=True)
        logging_manager.end_session()
        return

    # Convert agents tuple to list
    selected_agents = list(agents) if agents else None

    # Warn if using --agents with a directory that has a manifest
    if selected_agents and not from_stdin:
        input_path: Path | None = Path(content_text) if content_text else None
        if input_path and input_path.exists() and input_path.is_dir():
            manifest_path = input_path / "manifest.yaml"
            if manifest_path.exists():
                click.echo(
                    "⚠️  Warning: --agents flag will be ignored because manifest.yaml found in directory"
                )
                click.echo("   The manifest will determine reviewer selection")

    # Read context file if provided
    context_data = None
    if context:
        try:
            with open(context, encoding="utf-8") as f:
                context_data = f.read()
            click.echo(f"📄 Loaded context from: {context}")
        except Exception as e:
            click.echo(f"⚠️  Failed to read context file: {e}", err=True)

    # Run the review
    try:
        click.echo("🚀 Running review...")
        result = asyncio.run(
            manager.run_review(content_text, context_data, selected_agents)
        )

        # Format and display results
        formatted_output = manager.format_results(
            result, include_content=not no_content, include_context=include_context
        )
        click.echo(formatted_output)

        # Analysis output now includes any context generation defined in analyzer personas

        # Save to file - either specified output or auto-generated timestamped directory
        if output:
            # User specified output file
            try:
                output_content = formatted_output

                with open(output, "w", encoding="utf-8") as f:
                    f.write(output_content)
                click.echo(f"💾 Results saved to: {output}")
            except Exception as e:
                click.echo(f"❌ Error saving to {output}: {e}", err=True)
        else:
            # Auto-generate timestamped output directory and file
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = Path("output") / f"review_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / "results.md"
                output_content = formatted_output

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_content)
                click.echo(f"💾 Results automatically saved to: {output_file}")
            except Exception as e:
                click.echo(f"❌ Error saving to auto-generated output: {e}", err=True)

    except Exception as e:
        click.echo(f"❌ Error during review: {e}", err=True)
    finally:
        # End logging session
        logging_manager.end_session()
        session_dir = logging_manager.get_session_dir()
        if session_dir:
            click.echo(f"📁 Logs saved to: {session_dir}")


@cli.command()
def agents() -> None:
    """List available review agents."""
    try:
        manager = ConversationManager()
        available_agents = manager.get_available_agents()

        if not available_agents:
            click.echo(
                "❌ No review agents available. Run 'make setup' to configure personas."
            )
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
@click.argument("content", type=str)
@click.option("--agent", "-a", required=True, help="Specific agent to use")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save results to file (default: auto-generated timestamped directory in output/)",
)
def single(content: str, agent: str, output: str | None) -> None:
    """Review content with a single agent.

    CONTENT can be either text content or a file path.
    """
    # Check if content is a file path
    content_path = Path(content)
    if content_path.exists() and content_path.is_file():
        try:
            with open(content_path, encoding="utf-8") as f:
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

    # Initialize logging session
    logging_manager = LoggingManager.get_instance()
    content_info = (
        content_text[:100] + "..." if len(content_text) > 100 else content_text
    )
    session_id = logging_manager.start_session(
        content_info=content_info,
        selected_agents=[agent],
        model_provider="bedrock",  # Default for single command
        model_config={},
    )
    click.echo(f"📝 Started logging session: {session_id}")

    # Initialize conversation manager
    try:
        manager = ConversationManager()
    except Exception as e:
        click.echo(f"❌ Error initializing conversation manager: {e}", err=True)
        logging_manager.end_session()
        return

    # Run single agent review
    try:
        click.echo(f"🚀 Running review with {agent}...")
        result = asyncio.run(manager.run_review(content_text, None, [agent]))

        # Format and display results
        formatted_output = manager.format_results(
            result, include_content=True, include_context=False
        )
        click.echo(formatted_output)

        # Save to file - either specified output or auto-generated timestamped directory
        if output:
            # User specified output file
            try:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(formatted_output)
                click.echo(f"💾 Results saved to: {output}")
            except Exception as e:
                click.echo(f"❌ Error saving to {output}: {e}", err=True)
        else:
            # Auto-generate timestamped output directory and file
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = Path("output") / f"review_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / "results.md"

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(formatted_output)
                click.echo(f"💾 Results automatically saved to: {output_file}")
            except Exception as e:
                click.echo(f"❌ Error saving to auto-generated output: {e}", err=True)

    except Exception as e:
        click.echo(f"❌ Error during review: {e}", err=True)
    finally:
        # End logging session
        logging_manager.end_session()
        session_dir = logging_manager.get_session_dir()
        if session_dir:
            click.echo(f"📁 Logs saved to: {session_dir}")


@cli.command()
def status() -> None:
    """Show Review-Crew status and configuration."""
    try:
        # Check persona loader
        loader = PersonaLoader()
        config_info = loader.get_config_info()
        reviewers = loader.load_reviewer_personas()
        analyzers = loader.load_analyzer_personas()
        contextualizers = loader.load_contextualizer_personas()

        click.echo("📊 Review-Crew Status")
        click.echo("=" * 30)
        click.echo(f"✅ Reviewer personas loaded: {len(reviewers)}")
        click.echo(f"✅ Analyzer personas loaded: {len(analyzers)}")
        click.echo(f"✅ Contextualizer personas loaded: {len(contextualizers)}")

        # Check conversation manager
        manager = ConversationManager()
        agents = manager.get_available_agents()
        available_contextualizers = manager.get_available_contextualizers()
        click.echo(f"✅ Agents available: {len(agents)}")
        click.echo(f"✅ Contextualizers available: {len(available_contextualizers)}")

        # Get available analyzers
        available_analyzers = manager.get_available_analyzers()
        click.echo(f"✅ Analyzers available: {len(available_analyzers)}")

        click.echo("\n🎭 Configured Reviewer Personas:")
        for persona in reviewers:
            click.echo(f"  - {persona.name} ({persona.role})")

        click.echo("\n🔍 Configured Analyzer Personas:")
        for persona in analyzers:
            click.echo(f"  - {persona.name} ({persona.role})")

        click.echo("\n🔄 Configured Contextualizer Personas:")
        for persona in contextualizers:
            click.echo(f"  - {persona.name} ({persona.role})")

        click.echo("\n📁 Personas Directory:")
        status_icon = "✅" if config_info["personas_dir_exists"] else "❌"
        source_info = ""
        if config_info["is_using_env_var"]:
            source_info = " (from env var)"
        elif config_info["is_default_examples"]:
            source_info = " (default - examples)"

        click.echo(f"  {config_info['personas_dir']} {status_icon}{source_info}")

        # Show environment variable if set
        if config_info["env_personas_dir"]:
            click.echo("\n🌍 Environment Variable:")
            click.echo(f"  REVIEW_CREW_PERSONAS_DIR: {config_info['env_personas_dir']}")
        else:
            click.echo(
                "\n💡 Tip: Create .env file or set REVIEW_CREW_PERSONAS_DIR to use custom personas"
            )

    except Exception as e:
        click.echo(f"❌ Error checking status: {e}", err=True)


if __name__ == "__main__":
    cli()
