"""
Conversation Manager for orchestrating multi-agent reviews.

This module manages conversations between multiple review agents,
collecting and organizing their feedback.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .review_agent import ReviewAgent
from .analysis_agent import AnalysisAgent, AnalysisResult
from ..config.persona_loader import PersonaLoader, PersonaConfig


@dataclass
class ReviewResult:
    """Result from a single agent review."""
    agent_name: str
    agent_role: str
    feedback: str
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class ConversationResult:
    """Complete conversation result with all agent reviews."""
    content: str
    reviews: List[ReviewResult]
    timestamp: datetime
    summary: Optional[str] = None
    analysis: Optional[AnalysisResult] = None


class ConversationManager:
    """Manages multi-agent review conversations."""
    
    def __init__(self, persona_loader: Optional[PersonaLoader] = None, model_provider: str = 'bedrock', model_config: Optional[Dict[str, Any]] = None, enable_analysis: bool = True):
        """Initialize the conversation manager.
        
        Args:
            persona_loader: Optional PersonaLoader instance
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
            enable_analysis: Whether to enable analysis of reviews using analyzer personas
        """
        self.persona_loader = persona_loader or PersonaLoader()
        self.model_provider = model_provider
        self.model_config = model_config or {}
        self.enable_analysis = enable_analysis
        self.agents: List[ReviewAgent] = []
        self.analysis_agent: Optional[AnalysisAgent] = None
        self._load_agents()
        
        if self.enable_analysis:
            self.analysis_agent = AnalysisAgent(
                persona_name='meta_analysis',
                model_provider=model_provider,
                model_config=model_config
            )
    
    def _load_agents(self) -> None:
        """Load all reviewer personas as review agents."""
        try:
            personas = self.persona_loader.load_reviewer_personas()
            self.agents = [
                ReviewAgent(
                    persona, 
                    model_provider=self.model_provider,
                    model_config_override=self.model_config
                ) 
                for persona in personas
            ]
            print(f"✅ Loaded {len(self.agents)} review agents with {self.model_provider} provider")
        except Exception as e:
            print(f"❌ Error loading agents: {e}")
            self.agents = []
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get information about available agents.
        
        Returns:
            List of agent information dictionaries
        """
        return [agent.get_info() for agent in self.agents]
    
    def run_review(self, content: str, selected_agents: Optional[List[str]] = None) -> ConversationResult:
        """Run a synchronous review with selected agents.
        
        Args:
            content: Content to review
            selected_agents: Optional list of agent names to use (uses all if None)
            
        Returns:
            ConversationResult with all reviews
        """
        if not self.agents:
            raise ValueError("No review agents available. Check your persona configurations.")
        
        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)
        
        print(f"🎭 Starting review with {len(agents_to_use)} agents...")
        
        reviews = []
        for agent in agents_to_use:
            print(f"  📝 {agent.persona.name} is reviewing...")
            
            try:
                feedback = agent.review(content)
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback=feedback,
                    timestamp=datetime.now()
                )
                reviews.append(review)
                print(f"  ✅ {agent.persona.name} completed review")
                
            except Exception as e:
                error_review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=str(e)
                )
                reviews.append(error_review)
                print(f"  ❌ {agent.persona.name} failed: {e}")
        
        result = ConversationResult(
            content=content,
            reviews=reviews,
            timestamp=datetime.now()
        )
        
        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agent:
            successful_reviews = [r for r in reviews if not r.error]
            if len(successful_reviews) > 0:  # Always run analysis if we have any successful reviews
                print("🧠 Performing analysis...")
                try:
                    # Convert ReviewResult objects to dictionaries for analysis
                    review_dicts = [
                        {
                            'agent_name': r.agent_name,
                            'agent_role': r.agent_role,
                            'feedback': r.feedback,
                            'error': r.error
                        }
                        for r in successful_reviews
                    ]
                    
                    # Get context length from model config or use default
                    max_context_length = self.model_config.get('max_context_length', None)
                    analysis_result = self.analysis_agent.analyze(review_dicts, max_context_length)
                    result.analysis = analysis_result
                    print("✅ Analysis complete!")
                    
                except Exception as e:
                    print(f"⚠️  Analysis failed: {e}")
        
        print(f"🎉 Review complete! Collected {len([r for r in reviews if not r.error])} successful reviews")
        return result
    
    async def run_review_async(self, content: str, selected_agents: Optional[List[str]] = None) -> ConversationResult:
        """Run an asynchronous review with selected agents.
        
        Args:
            content: Content to review
            selected_agents: Optional list of agent names to use (uses all if None)
            
        Returns:
            ConversationResult with all reviews
        """
        if not self.agents:
            raise ValueError("No review agents available. Check your persona configurations.")
        
        # Filter agents if specific ones are requested
        agents_to_use = self._filter_agents(selected_agents)
        
        print(f"🎭 Starting async review with {len(agents_to_use)} agents...")
        
        # Run all reviews concurrently
        tasks = []
        for agent in agents_to_use:
            task = self._review_with_agent_async(agent, content)
            tasks.append(task)
        
        reviews = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_reviews = []
        for i, result in enumerate(reviews):
            agent = agents_to_use[i]
            
            if isinstance(result, Exception):
                review = ReviewResult(
                    agent_name=agent.persona.name,
                    agent_role=agent.persona.role,
                    feedback="",
                    timestamp=datetime.now(),
                    error=str(result)
                )
                print(f"  ❌ {agent.persona.name} failed: {result}")
            else:
                review = result
                print(f"  ✅ {agent.persona.name} completed review")
            
            processed_reviews.append(review)
        
        result = ConversationResult(
            content=content,
            reviews=processed_reviews,
            timestamp=datetime.now()
        )
        
        # Perform analysis if enabled and we have successful reviews
        if self.enable_analysis and self.analysis_agent:
            successful_reviews = [r for r in processed_reviews if not r.error]
            if len(successful_reviews) > 0:  # Always run analysis if we have any successful reviews
                print("🧠 Performing analysis...")
                try:
                    # Convert ReviewResult objects to dictionaries for analysis
                    review_dicts = [
                        {
                            'agent_name': r.agent_name,
                            'agent_role': r.agent_role,
                            'feedback': r.feedback,
                            'error': r.error
                        }
                        for r in successful_reviews
                    ]
                    
                    # Get context length from model config or use default
                    max_context_length = self.model_config.get('max_context_length', None)
                    analysis_result = await self.analysis_agent.analyze_async(review_dicts, max_context_length)
                    result.analysis = analysis_result
                    print("✅ Analysis complete!")
                    
                except Exception as e:
                    print(f"⚠️  Analysis failed: {e}")
        
        print(f"🎉 Async review complete! Collected {len([r for r in processed_reviews if not r.error])} successful reviews")
        return result
    
    async def _review_with_agent_async(self, agent: ReviewAgent, content: str) -> ReviewResult:
        """Review content with a single agent asynchronously."""
        try:
            feedback = await agent.review_async(content)
            return ReviewResult(
                agent_name=agent.persona.name,
                agent_role=agent.persona.role,
                feedback=feedback,
                timestamp=datetime.now()
            )
        except Exception as e:
            return ReviewResult(
                agent_name=agent.persona.name,
                agent_role=agent.persona.role,
                feedback="",
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _filter_agents(self, selected_agents: Optional[List[str]]) -> List[ReviewAgent]:
        """Filter agents based on selection.
        
        Args:
            selected_agents: List of agent names to include, or None for all
            
        Returns:
            Filtered list of ReviewAgent objects
        """
        if selected_agents is None:
            return self.agents
        
        # Filter by name (case-insensitive)
        selected_lower = [name.lower() for name in selected_agents]
        filtered = [
            agent for agent in self.agents 
            if agent.persona.name.lower() in selected_lower
        ]
        
        if not filtered:
            print(f"⚠️  No agents found matching: {selected_agents}")
            print(f"Available agents: {[agent.persona.name for agent in self.agents]}")
            return self.agents
        
        return filtered
    
    def format_results(self, result: ConversationResult, include_content: bool = True) -> str:
        """Format conversation results for display as clean markdown.
        
        Args:
            result: ConversationResult to format
            include_content: Whether to include the original content
            
        Returns:
            Formatted markdown string
        """
        output = []
        
        # Header
        output.append("# Review-Crew Analysis Results")
        output.append("")
        output.append(f"**Analysis completed:** {result.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}")
        output.append("")
        
        if include_content:
            output.append("## Content Reviewed")
            output.append("")
            # Format content with proper markdown
            content_lines = result.content.split('\n')
            for line in content_lines:
                if line.strip():
                    output.append(line)
                else:
                    output.append("")
            output.append("")
        
        # Summary
        successful_reviews = [r for r in result.reviews if not r.error]
        failed_reviews = [r for r in result.reviews if r.error]
        
        output.append("## Summary")
        output.append("")
        output.append(f"- **Total Reviews:** {len(result.reviews)}")
        output.append(f"- **Successful:** {len(successful_reviews)} ✅")
        if failed_reviews:
            output.append(f"- **Failed:** {len(failed_reviews)} ❌")
        if result.analysis:
            output.append("- **Meta-Analysis:** Included 🧠")
        output.append("")
        
        # Individual Reviews
        if successful_reviews:
            output.append("## Individual Reviews")
            output.append("")
            
            for i, review in enumerate(successful_reviews, 1):
                output.append(f"### {i}. {review.agent_name}")
                output.append(f"**Role:** {review.agent_role}")
                output.append("")
                
                # Extract clean text from feedback (handle both string and dict formats)
                clean_feedback = self._extract_clean_feedback(review.feedback)
                output.append(clean_feedback)
                output.append("")
                output.append("---")
                output.append("")
        
        # Analysis Section
        if result.analysis:
            output.append("## Analysis & Synthesis")
            output.append("")
            output.append(result.analysis.synthesis)
            output.append("")
            output.append("---")
            output.append("")
            
            # Additional analysis context is included in the synthesis above
        
        # Failed Reviews
        if failed_reviews:
            output.append("## Failed Reviews")
            output.append("")
            for review in failed_reviews:
                output.append(f"- **{review.agent_name}:** {review.error}")
            output.append("")
        
        return "\n".join(output)
    
    def _extract_clean_feedback(self, feedback) -> str:
        """Extract clean text from feedback, handling various formats."""
        import json
        import ast
        
        if isinstance(feedback, str):
            # Try to parse as JSON/dict if it looks like one
            if feedback.strip().startswith('{') and feedback.strip().endswith('}'):
                try:
                    # Try JSON first
                    parsed = json.loads(feedback)
                    return self._extract_clean_feedback(parsed)
                except json.JSONDecodeError:
                    try:
                        # Try literal_eval for Python dict format
                        parsed = ast.literal_eval(feedback)
                        return self._extract_clean_feedback(parsed)
                    except (ValueError, SyntaxError):
                        # If parsing fails, return as-is
                        return feedback
            else:
                # If it's already a plain string, return as-is
                return feedback
        elif isinstance(feedback, dict):
            # Handle dictionary format (like from API responses)
            if 'role' in feedback and 'content' in feedback:
                content = feedback['content']
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and 'text' in content[0]:
                        return content[0]['text']
                    else:
                        return str(content[0])
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)
            elif 'text' in feedback:
                return feedback['text']
            else:
                return str(feedback)
        else:
            # Fallback to string representation
            return str(feedback)
    
    # Removed college-specific supplemental context methods - analysis personas now handle context generation automatically
