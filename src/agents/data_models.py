"""
Data models for Review-Crew agents.

This module contains the core data structures used throughout the agent system.
"""

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

from .analysis_agent import AnalysisResult
from .context_agent import ContextResult


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
    analysis_results: List[AnalysisResult] = None
    context_results: List[ContextResult] = None
    original_content: Optional[str] = None
    analysis_errors: List[str] = None  # Track analysis failures separately

    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.context_results is None:
            self.context_results = []
        if self.analysis_errors is None:
            self.analysis_errors = []
