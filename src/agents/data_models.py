"""
Data models for Review-Crew agents.

This module contains the core data structures used throughout the agent system.
"""

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
    error: str | None = None


@dataclass
class ConversationResult:
    """Complete conversation result with all agent reviews."""

    content: str
    reviews: list[ReviewResult]
    timestamp: datetime
    summary: str | None = None
    analysis_results: list[AnalysisResult] = None
    context_results: list[ContextResult] = None
    original_content: str | None = None
    analysis_errors: list[str] = None  # Track analysis failures separately

    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.context_results is None:
            self.context_results = []
        if self.analysis_errors is None:
            self.analysis_errors = []
