"""
Tools package for Review-Crew system.

This package contains deterministic analysis tools that enhance persona accuracy
by providing objective, measurable data that LLMs cannot reliably generate.
"""

from .academic_tools import (
    analyze_essay_strength,
    analyze_personal_voice,
    detect_cliches,
)
from .structure_analysis import (
    analyze_document_structure,
    analyze_paragraph_flow,
    detect_essay_components,
)
from .text_metrics import (
    analyze_readability,
    analyze_vocabulary,
    get_text_metrics,
    validate_constraints,
)

__all__ = [
    # Text metrics and constraints
    "get_text_metrics",
    "validate_constraints",
    "analyze_readability",
    "analyze_vocabulary",
    # Structure analysis
    "analyze_document_structure",
    "detect_essay_components",
    "analyze_paragraph_flow",
    # Academic writing tools
    "analyze_essay_strength",
    "detect_cliches",
    "analyze_personal_voice",
]
