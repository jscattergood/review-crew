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


class ConversationManager:
    """Manages multi-agent review conversations."""
    
    def __init__(self, persona_loader: Optional[PersonaLoader] = None, model_provider: str = 'bedrock', model_config: Optional[Dict[str, Any]] = None):
        """Initialize the conversation manager.
        
        Args:
            persona_loader: Optional PersonaLoader instance
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
        """
        self.persona_loader = persona_loader or PersonaLoader()
        self.model_provider = model_provider
        self.model_config = model_config or {}
        self.agents: List[ReviewAgent] = []
        self._load_agents()
    
    def _load_agents(self) -> None:
        """Load all available personas as review agents."""
        try:
            personas = self.persona_loader.load_all_personas()
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
        """Format conversation results for display.
        
        Args:
            result: ConversationResult to format
            include_content: Whether to include the original content
            
        Returns:
            Formatted string representation
        """
        output = []
        
        if include_content:
            output.append("📄 CONTENT REVIEWED:")
            output.append("=" * 50)
            output.append(result.content[:200] + "..." if len(result.content) > 200 else result.content)
            output.append("")
        
        output.append("🎭 REVIEW RESULTS:")
        output.append("=" * 50)
        
        successful_reviews = [r for r in result.reviews if not r.error]
        failed_reviews = [r for r in result.reviews if r.error]
        
        output.append(f"✅ {len(successful_reviews)} successful reviews")
        if failed_reviews:
            output.append(f"❌ {len(failed_reviews)} failed reviews")
        output.append("")
        
        # Display successful reviews
        for review in successful_reviews:
            output.append(f"👤 {review.agent_name} ({review.agent_role})")
            output.append("-" * 40)
            output.append(review.feedback)
            output.append("")
        
        # Display failed reviews
        if failed_reviews:
            output.append("❌ FAILED REVIEWS:")
            output.append("-" * 20)
            for review in failed_reviews:
                output.append(f"👤 {review.agent_name}: {review.error}")
            output.append("")
        
        output.append(f"⏰ Completed at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(output)
