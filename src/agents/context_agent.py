"""
Context Agent for processing and formatting contextual information.

This agent is responsible for taking separate context information and
formatting it according to a contextualizer persona for use by review agents.
The agent is completely abstract and domain-agnostic.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .review_agent import ReviewAgent
from ..config.persona_loader import PersonaConfig, PersonaLoader


@dataclass
class ContextResult:
    """Result from context processing and formatting."""
    formatted_context: str
    context_summary: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ContextAgent:
    """Agent that processes and formats contextual information using a contextualizer persona."""
    
    def __init__(self, persona_name: str, model_provider: str = 'bedrock', model_config: Optional[Dict[str, Any]] = None):
        """Initialize the context agent.
        
        Args:
            persona_name: Name of the contextualizer persona to load (without .yaml extension)
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
        """
        self.persona_name = persona_name
        self.model_provider = model_provider
        self.model_config = model_config or {}
        
        # Load the contextualizer persona
        self.persona = self._load_contextualizer_persona(persona_name)
        
        # Create the underlying review agent if persona is available
        if self.persona:
            self.agent = ReviewAgent(
                self.persona,
                model_provider=model_provider,
                model_config_override=model_config
            )
        else:
            self.agent = None
    
    def _load_contextualizer_persona(self, persona_name: str) -> Optional[PersonaConfig]:
        """Load a contextualizer persona configuration."""
        try:
            loader = PersonaLoader()
            
            # Check in the contextualizers directory
            contextualizers_dir = loader.personas_dir / "contextualizers"
            persona_path = contextualizers_dir / f'{persona_name}.yaml'
            
            if persona_path.exists():
                return loader.load_persona(persona_path)
            else:
                print(f"⚠️  No contextualizer persona found: {persona_name}")
                print(f"   Expected location: {persona_path}")
                return None
        except Exception as e:
            print(f"⚠️  Could not load contextualizer persona '{persona_name}': {e}")
            return None
    
    def process_context(self, context_data: str) -> Optional[ContextResult]:
        """Process and format contextual information using the contextualizer persona.
        
        Args:
            context_data: The raw context information to be processed and formatted
            
        Returns:
            ContextResult with formatted context, or None if no contextualizer available
        """
        if not self.agent:
            print("ℹ️  No contextualizer persona available - no additional context will be provided")
            return None
            
        # Use the contextualizer agent to process the context
        result = self.agent.review(context_data)
        
        # Parse the structured response
        return self._parse_context_response(result)
    
    async def process_context_async(self, context_data: str) -> Optional[ContextResult]:
        """Asynchronously process and format contextual information.
        
        Args:
            context_data: The raw context information to be processed and formatted
            
        Returns:
            ContextResult with formatted context, or None if no contextualizer available
        """
        if not self.agent:
            print("ℹ️  No contextualizer persona available - no additional context will be provided")
            return None
            
        # Use the contextualizer agent to process the context
        result = await self.agent.review_async(context_data)
        
        # Parse the structured response
        return self._parse_context_response(result)
    
    def _parse_context_response(self, response: str) -> ContextResult:
        """Parse the structured context processing response."""
        # Try to extract a context summary if the persona provides one
        context_summary = self._extract_section(response, "CONTEXT SUMMARY")
        
        # If no structured summary found, create a simple one
        if not context_summary:
            context_summary = "Context processed by contextualizer persona"
        
        # The full response is the formatted context
        formatted_context = response
        
        return ContextResult(
            formatted_context=formatted_context,
            context_summary=context_summary
        )
    
    def _extract_section(self, text: str, section_header: str) -> Optional[str]:
        """Extract a specific section from the structured response."""
        import re
        
        # Look for the section header (with ## prefix)
        pattern = rf"## {section_header}\s*\n(.*?)(?=\n\s*## |\n\s*---|\n\s*\n.*\S|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            return content
        
        return None
    
    def format_context_for_review(self, context_result: ContextResult) -> str:
        """Format processed context for inclusion in review prompts.
        
        Args:
            context_result: The result from context processing
            
        Returns:
            Formatted context string for reviewers
        """
        return f"""## CONTEXT
{context_result.formatted_context}

---
"""
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this context agent."""
        if not self.persona:
            return {
                'name': 'No Context Agent',
                'role': 'No contextualizer persona available',
                'goal': 'No additional context processing',
                'capabilities': []
            }
        
        return {
            'name': self.persona.name,
            'role': self.persona.role,
            'goal': self.persona.goal,
            'capabilities': [
                'Process contextual information',
                'Format context for reviewers',
                'Create context summaries'
            ]
        }
