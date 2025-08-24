"""
Analysis Agent for performing various types of analysis on review feedback.

This agent can perform different types of analysis based on the loaded persona:
- Meta-analysis and feedback synthesis
- Conflict resolution
- Priority ranking of feedback
- Personal statement summaries for supplemental essay context
- Other analysis types as defined by analyzer personas
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .review_agent import ReviewAgent
from ..config.persona_loader import PersonaConfig


@dataclass
class AnalysisResult:
    """Result from analysis of multiple agent reviews."""
    synthesis: str
    personal_statement_summary: Optional[str] = None
    key_themes: List[str] = None
    conflicting_feedback: List[Dict[str, Any]] = None
    priority_recommendations: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.key_themes is None:
            self.key_themes = []
        if self.conflicting_feedback is None:
            self.conflicting_feedback = []
        if self.priority_recommendations is None:
            self.priority_recommendations = []


class AnalysisAgent:
    """Agent that performs analysis on feedback from multiple review agents based on loaded persona."""
    
    def __init__(self, persona_name: str = 'meta_analysis', model_provider: str = 'bedrock', model_config: Optional[Dict[str, Any]] = None):
        """Initialize the analysis agent.
        
        Args:
            persona_name: Name of the analyzer persona to load (without .yaml extension)
            model_provider: Model provider to use ('bedrock', 'lm_studio', 'ollama')
            model_config: Optional model configuration override
        """
        self.persona_name = persona_name
        self.model_provider = model_provider
        self.model_config = model_config or {}
        
        # Load the specified analyzer persona
        self.persona = self._load_analyzer_persona(persona_name)
        
        # Create the underlying review agent
        self.agent = ReviewAgent(
            self.persona,
            model_provider=model_provider,
            model_config_override=model_config
        )
    
    def _load_analyzer_persona(self, persona_name: str) -> PersonaConfig:
        """Load an analyzer persona configuration."""
        # Try to load from YAML file first
        try:
            from ..config.persona_loader import PersonaLoader
            import os
            loader = PersonaLoader()
            default_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'personas', 'analyzers', f'{persona_name}.yaml')
            if os.path.exists(default_path):
                return loader.load_persona(default_path)
        except Exception:
            pass
        
        # Fallback to hardcoded meta-analysis persona for backwards compatibility
        if persona_name == 'meta_analysis':
            return PersonaConfig(
            name="Meta-Analysis Synthesizer",
            role="Senior Editorial Consultant & Application Strategy Expert",
            goal="Synthesize multiple perspectives into coherent, actionable guidance while creating strategic context for supplemental essays",
            backstory="""You are a senior editorial consultant with 15+ years of experience in college admissions consulting and content strategy. You specialize in synthesizing diverse feedback from multiple reviewers and creating strategic narratives that connect personal statements with supplemental essays. You have deep expertise in identifying thematic consistency, resolving conflicting advice, and prioritizing recommendations for maximum impact.""",
            prompt_template="""Analyze the following collection of reviews to provide a comprehensive meta-analysis.

<!-- Original content placeholder (not used in analysis): {content} -->

INDIVIDUAL REVIEWS:
{reviews}

Please provide a structured analysis with the following sections:

## SYNTHESIS & RECOMMENDATIONS
Synthesize all feedback into coherent, prioritized recommendations. Resolve any conflicts between reviewers by providing balanced guidance.

## PERSONAL STATEMENT SUMMARY (for supplemental essay context)
Create a concise 2-3 sentence summary of the personal statement's core themes, unique angle, and key strengths. This will be used as context for supplemental essays to ensure consistency and avoid repetition.

## KEY THEMES IDENTIFIED
List the 3-5 most important themes that emerged across all reviews.

## CONFLICTING FEEDBACK RESOLUTION
If reviewers provided conflicting advice, explain the conflicts and provide balanced recommendations.

## PRIORITY ACTION ITEMS
Rank the top 5 most important changes/improvements in order of priority and impact.

## STRATEGIC GUIDANCE
Provide strategic advice on how the reviewed content fits into the broader application narrative and what new dimensions should be explored in supplemental essays.

Focus on actionable insights that will genuinely improve the content while maintaining the author's authentic voice.""",
            model_config={
                'temperature': 0.4,  # Slightly higher for creative synthesis
                'max_tokens': 2000   # More tokens for comprehensive analysis
            }
        )
        else:
            raise ValueError(f"Unknown analyzer persona: {persona_name}. Please create {persona_name}.yaml in config/personas/analyzers/")
    
    def analyze(self, reviews: List[Dict[str, Any]], max_context_length: Optional[int] = None) -> AnalysisResult:
        """Perform analysis on multiple agent reviews with optional chunking.
        
        Args:
            reviews: List of review results from other agents
            max_context_length: Maximum context length for chunking (if None, no chunking)
            
        Returns:
            AnalysisResult with synthesized analysis
        """
        # Check if we need to use chunking strategy
        if max_context_length and self._should_chunk(reviews, max_context_length):
            return self._analyze_with_chunking(reviews, max_context_length)
        else:
            return self._analyze_full_context(reviews)
    
    def _analyze_full_context(self, reviews: List[Dict[str, Any]]) -> AnalysisResult:
        """Perform analysis with full context (original method)."""
        # Format reviews for the prompt
        formatted_reviews = self._format_reviews_for_analysis(reviews)
        
        # Create the analysis prompt by manually formatting it
        # (since ReviewAgent.review() only expects {content} placeholder)
        analysis_prompt = self.persona.prompt_template.format(
            reviews=formatted_reviews
        )
        
        # Get the analysis by calling the underlying agent directly
        # We bypass the ReviewAgent.review() method since it expects a different prompt format
        result = self.agent.agent(analysis_prompt)
        
        # Extract the text content from the response
        if hasattr(result, 'message'):
            if isinstance(result.message, dict) and 'content' in result.message:
                if isinstance(result.message['content'], list) and len(result.message['content']) > 0:
                    synthesis = result.message['content'][0].get('text', str(result.message))
                else:
                    synthesis = str(result.message['content'])
            else:
                synthesis = str(result.message)
        else:
            synthesis = str(result)
        
        # Parse the structured response
        return self._parse_meta_analysis_response(synthesis)
    
    async def analyze_async(self, reviews: List[Dict[str, Any]], max_context_length: Optional[int] = None) -> AnalysisResult:
        """Perform asynchronous meta-analysis on multiple agent reviews.
        
        Args:
            reviews: List of review results from other agents
            max_context_length: Maximum context length for chunking (if None, no chunking)
            
        Returns:
            AnalysisResult with synthesized analysis
        """
        # Check if we need to use chunking strategy
        if max_context_length and self._should_chunk(reviews, max_context_length):
            # Note: For async, we'll use the sync chunking method for now
            # In a full implementation, we'd make the chunking async too
            return self._analyze_with_chunking(reviews, max_context_length)
        
        # Format reviews for the prompt
        formatted_reviews = self._format_reviews_for_analysis(reviews)
        
        # Create the analysis prompt by manually formatting it
        # (since ReviewAgent.review_async() only expects {content} placeholder)
        analysis_prompt = self.persona.prompt_template.format(
            reviews=formatted_reviews
        )
        
        # Get the meta-analysis by calling the underlying agent directly
        # We bypass the ReviewAgent.review_async() method since it expects a different prompt format
        result = await self.agent.agent.invoke_async(analysis_prompt)
        
        # Extract the text content from the response
        if hasattr(result, 'message'):
            if isinstance(result.message, dict) and 'content' in result.message:
                if isinstance(result.message['content'], list) and len(result.message['content']) > 0:
                    synthesis = result.message['content'][0].get('text', str(result.message))
                else:
                    synthesis = str(result.message['content'])
            else:
                synthesis = str(result.message)
        else:
            synthesis = str(result)
        
        # Parse the structured response
        return self._parse_meta_analysis_response(synthesis)
    
    def _format_reviews_for_analysis(self, reviews: List[Dict[str, Any]]) -> str:
        """Format individual reviews for meta-analysis."""
        formatted = []
        
        for i, review in enumerate(reviews, 1):
            if review.get('error'):
                continue  # Skip failed reviews
                
            agent_name = review.get('agent_name', f'Reviewer {i}')
            agent_role = review.get('agent_role', 'Unknown Role')
            feedback = review.get('feedback', '')
            
            formatted.append(f"""
### Review {i}: {agent_name} ({agent_role})
{feedback}
""")
        
        return "\n".join(formatted)
    
    def _should_chunk(self, reviews: List[Dict[str, Any]], max_context_length: int) -> bool:
        """Determine if chunking is needed based on estimated token count.
        
        Args:
            reviews: List of reviews
            max_context_length: Maximum context length
            
        Returns:
            True if chunking is needed
        """
        # Rough estimation: 1 token ≈ 4 characters (conservative estimate)
        chars_per_token = 4
        
        # Estimate token count for reviews only
        # content_tokens = 0  # No longer using original content
        reviews_text = self._format_reviews_for_analysis(reviews)
        reviews_tokens = len(reviews_text) // chars_per_token
        
        # Add buffer for prompt template and response
        prompt_buffer = 500  # tokens for prompt template
        response_buffer = 1000  # tokens for response
        
        total_estimated_tokens = reviews_tokens + prompt_buffer + response_buffer
        
        return total_estimated_tokens > max_context_length
    
    def _analyze_with_chunking(self, reviews: List[Dict[str, Any]], max_context_length: int) -> AnalysisResult:
        """Perform analysis using chunking strategy for large review sets.
        
        Args:
            reviews: List of reviews
            max_context_length: Maximum context length
            
        Returns:
            AnalysisResult with synthesized analysis
        """
        print(f"🔄 Using chunking strategy for {len(reviews)} reviews (context limit: {max_context_length})")
        
        # Create chunks of reviews
        review_chunks = self._create_review_chunks(reviews, max_context_length)
        
        # Analyze each chunk
        chunk_analyses = []
        for i, chunk in enumerate(review_chunks, 1):
            print(f"📝 Analyzing chunk {i}/{len(review_chunks)} ({len(chunk)} reviews)")
            
            # Create a modified prompt for chunk analysis
            chunk_prompt = self._create_chunk_analysis_prompt()
            
            # Format the chunk
            formatted_chunk = self._format_reviews_for_analysis(chunk)
            
            # Create analysis prompt for this chunk
            analysis_prompt = chunk_prompt.format(
                reviews=formatted_chunk,
                chunk_number=i,
                total_chunks=len(review_chunks)
            )
            
            # Get analysis for this chunk
            result = self.agent.agent(analysis_prompt)
            
            # Extract text content
            if hasattr(result, 'message'):
                if isinstance(result.message, dict) and 'content' in result.message:
                    if isinstance(result.message['content'], list) and len(result.message['content']) > 0:
                        chunk_analysis = result.message['content'][0].get('text', str(result.message))
                    else:
                        chunk_analysis = str(result.message['content'])
                else:
                    chunk_analysis = str(result.message)
            else:
                chunk_analysis = str(result)
            
            chunk_analyses.append(chunk_analysis)
        
        # Synthesize all chunk analyses into final result
        print("🔄 Synthesizing chunk analyses into final result...")
        return self._synthesize_chunk_analyses(chunk_analyses)
    
    def _parse_meta_analysis_response(self, response: str) -> AnalysisResult:
        """Parse the structured meta-analysis response."""
        # For now, return the full response as synthesis
        # In a more sophisticated implementation, we could parse sections
        
        # Try to extract personal statement summary if it exists
        personal_statement_summary = self._extract_section(response, "PERSONAL STATEMENT SUMMARY")
        
        # Try to extract key themes
        key_themes = self._extract_list_section(response, "KEY THEMES IDENTIFIED")
        
        # Try to extract priority recommendations
        priority_recommendations = self._extract_list_section(response, "PRIORITY ACTION ITEMS")
        
        return AnalysisResult(
            synthesis=response,
            personal_statement_summary=personal_statement_summary,
            key_themes=key_themes,
            priority_recommendations=priority_recommendations
        )
    
    def _extract_section(self, text: str, section_header: str) -> Optional[str]:
        """Extract a specific section from the structured response."""
        import re
        
        # Look for the section header
        pattern = rf"## {section_header}.*?\n(.*?)(?=\n## |\n$|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            # Clean up the content
            content = re.sub(r'\n+', ' ', content)  # Replace multiple newlines with space
            return content
        
        return None
    
    def _extract_list_section(self, text: str, section_header: str) -> List[str]:
        """Extract a list section from the structured response."""
        section_content = self._extract_section(text, section_header)
        
        if not section_content:
            return []
        
        # Split by common list indicators
        import re
        items = re.split(r'[•\-\*]\s*|\d+\.\s*', section_content)
        
        # Clean up and filter items
        cleaned_items = []
        for item in items:
            item = item.strip()
            if item and len(item) > 10:  # Filter out very short items
                cleaned_items.append(item)
        
        return cleaned_items[:5]  # Return top 5 items
    
    def create_supplemental_context(self, personal_statement: str, meta_analysis: AnalysisResult) -> str:
        """Create context summary for supplemental essays.
        
        Args:
            personal_statement: The original personal statement
            meta_analysis: Meta-analysis result containing summary
            
        Returns:
            Formatted context string for supplemental essays
        """
        context_parts = []
        
        # Add personal statement summary
        if meta_analysis.personal_statement_summary:
            context_parts.append(f"**Personal Statement Theme:** {meta_analysis.personal_statement_summary}")
        
        # Add key themes
        if meta_analysis.key_themes:
            themes_str = ", ".join(meta_analysis.key_themes[:3])  # Top 3 themes
            context_parts.append(f"**Key Strengths:** {themes_str}")
        
        # Add strategic guidance
        context_parts.append("**Note:** This supplemental should add new dimensions beyond the personal statement themes")
        
        return "\n".join(context_parts)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this meta-analysis agent."""
        return {
            'name': self.persona.name,
            'role': self.persona.role,
            'goal': self.persona.goal,
            'capabilities': [
                'Synthesize multiple reviewer feedback',
                'Resolve conflicting recommendations',
                'Create personal statement summaries',
                'Prioritize action items',
                'Generate supplemental essay context'
            ]
        }
    
    def _create_review_chunks(self, reviews: List[Dict[str, Any]], max_context_length: int) -> List[List[Dict[str, Any]]]:
        """Create chunks of reviews that fit within context limits.
        
        Args:
            reviews: List of all reviews
            max_context_length: Maximum context length
            
        Returns:
            List of review chunks
        """
        chars_per_token = 4
        # content_tokens = 0  # No longer using original content
        prompt_buffer = 800  # tokens for prompt template and instructions
        response_buffer = 1000  # tokens for response
        
        # Available tokens for reviews in each chunk
        available_tokens = max_context_length - prompt_buffer - response_buffer
        
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        for review in reviews:
            # Estimate tokens for this review
            review_text = self._format_single_review_for_analysis(review)
            review_tokens = len(review_text) // chars_per_token
            
            # If adding this review would exceed the limit, start a new chunk
            if current_chunk and (current_chunk_tokens + review_tokens) > available_tokens:
                chunks.append(current_chunk)
                current_chunk = [review]
                current_chunk_tokens = review_tokens
            else:
                current_chunk.append(review)
                current_chunk_tokens += review_tokens
        
        # Add the last chunk if it has reviews
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk_analysis_prompt(self) -> str:
        """Create a modified prompt template for chunk analysis."""
        return """You are analyzing a subset of reviews. This is chunk {chunk_number} of {total_chunks} total chunks.

## Reviews in This Chunk:
{reviews}

## Your Task:
Analyze ONLY the reviews in this chunk and provide:

### KEY INSIGHTS FROM THIS CHUNK
- Main themes and patterns identified in these specific reviews
- Notable strengths highlighted by these reviewers
- Common concerns or improvement areas mentioned
- Any conflicting opinions within this chunk

### CHUNK SUMMARY
- Brief summary of the overall sentiment in this chunk
- Most important takeaways from these specific reviews
- Any standout comments or unique perspectives

Focus on extracting the key insights from this specific set of reviews. Keep your analysis concise and focused on the most important points."""
    
    def _synthesize_chunk_analyses(self, chunk_analyses: List[str]) -> AnalysisResult:
        """Synthesize multiple chunk analyses into a final result.
        
        Args:
            chunk_analyses: List of analysis results from each chunk
            
        Returns:
            Final AnalysisResult
        """
        # Create a synthesis prompt
        synthesis_prompt = f"""You are synthesizing analysis results from multiple chunks of reviews.

## Chunk Analyses:
"""
        
        for i, analysis in enumerate(chunk_analyses, 1):
            synthesis_prompt += f"\n### Chunk {i} Analysis:\n{analysis}\n"
        
        synthesis_prompt += """

## Your Task:
Synthesize all chunk analyses into a comprehensive final analysis. DO NOT use chunk-specific headers like "KEY INSIGHTS FROM THIS CHUNK" or "CHUNK SUMMARY". Instead, provide a clean final analysis with these sections:

### SYNTHESIS & RECOMMENDATIONS
Combine insights from all reviews into coherent, prioritized recommendations.

### KEY THEMES IDENTIFIED  
List the 3-5 most important themes that emerged across ALL reviews.

### CONSENSUS AREAS
Areas where multiple reviewers agreed.

### CONFLICTING FEEDBACK RESOLUTION
Identify and resolve any conflicts between different reviewers.

### PRIORITY ACTION ITEMS
Rank the top 5 most important changes/improvements based on all feedback.

### STRATEGIC GUIDANCE
Provide strategic advice based on the complete picture from all reviews.

Focus on creating a comprehensive synthesis that captures the full scope of all reviewer feedback. Write as if you analyzed all reviews directly, not as chunks."""
        
        # Get the final synthesis
        result = self.agent.agent(synthesis_prompt)
        
        # Extract text content
        if hasattr(result, 'message'):
            if isinstance(result.message, dict) and 'content' in result.message:
                if isinstance(result.message['content'], list) and len(result.message['content']) > 0:
                    synthesis = result.message['content'][0].get('text', str(result.message))
                else:
                    synthesis = str(result.message['content'])
            else:
                synthesis = str(result.message)
        else:
            synthesis = str(result)
        
        # Parse the final synthesis
        return self._parse_meta_analysis_response(synthesis)
    
    def _format_single_review_for_analysis(self, review: Dict[str, Any]) -> str:
        """Format a single review for analysis (used in chunking)."""
        agent_name = review.get('agent_name', 'Unknown Agent')
        agent_role = review.get('agent_role', 'Unknown Role')
        
        if review.get('error'):
            return f"**{agent_name}** ({agent_role}): ERROR - {review['error']}\n"
        
        feedback = review.get('feedback', review.get('review', ''))
        
        # Handle different feedback formats
        if isinstance(feedback, dict):
            if 'content' in feedback:
                if isinstance(feedback['content'], list) and len(feedback['content']) > 0:
                    feedback_text = feedback['content'][0].get('text', str(feedback))
                else:
                    feedback_text = str(feedback.get('content', feedback))
            else:
                feedback_text = str(feedback)
        else:
            feedback_text = str(feedback)
        
        return f"**{agent_name}** ({agent_role}):\n{feedback_text}\n\n"
