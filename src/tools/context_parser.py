"""
Context-aware constraint parsing for dynamic tool configuration.

This module extracts constraints and requirements from content context
rather than requiring static configuration in persona files.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContentConstraints:
    """Dynamically extracted content constraints."""

    # Word/character limits
    word_limit: int | None = None
    character_limit: int | None = None
    min_words: int | None = None

    # Content type and context
    essay_type: str | None = None
    school_name: str | None = None
    prompt: str | None = None

    # Special requirements
    special_requirements: list[str] | None = None
    target_audience: str | None = None

    # Optimal ranges based on type
    optimal_word_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.special_requirements is None:
            self.special_requirements = []


def extract_essay_content(content: str) -> str:
    """
    Extract just the essay content from a document that includes assignment context.

    Looks for common patterns that separate assignment context from essay content:
    - "ESSAY TO REVIEW:"
    - "Essay:"
    - "Content:"
    - Or falls back to content after the first blank line

    Args:
        content: Full content including context and essay text

    Returns:
        Just the essay content without assignment context

    Example:
        >>> content = "Word Limit: 650\\n\\nESSAY TO REVIEW:\\n\\nMy essay starts here..."
        >>> extract_essay_content(content)
        'My essay starts here...'
    """
    if not content or not content.strip():
        return content

    # Look for explicit essay content markers
    essay_markers = [
        r"\*\*ESSAY TO REVIEW:\*\*\s*\n",  # **ESSAY TO REVIEW:**
        r"ESSAY TO REVIEW:\s*\n",  # ESSAY TO REVIEW:
        r"\*\*QUESTIONS TO REVIEW:\*\*\s*\n",  # **QUESTIONS TO REVIEW:**
        r"QUESTIONS TO REVIEW:\s*\n",  # QUESTIONS TO REVIEW:
        r"Essay:\s*\n",
        r"Content:\s*\n",
        r"Text to analyze:\s*\n",
        r"Document:\s*\n",
    ]

    for marker in essay_markers:
        match = re.search(marker, content, re.IGNORECASE)
        if match:
            # Return everything after the marker
            essay_content = content[match.end() :].strip()
            return essay_content

    # Fallback: Look for content after assignment context patterns
    # If we see assignment-like patterns, try to find where the actual content starts
    lines = content.split("\n")
    essay_start_idx = 0

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Skip common assignment context patterns
        if any(
            pattern in line_lower
            for pattern in [
                "assignment context",
                "word limit",
                "character limit",
                "essay type",
                "prompt:",
                "question:",
                "school:",
                "university:",
                "constraint:",
                "requirement:",
            ]
        ):
            continue

        # Skip headers and formatting
        if line.strip().startswith("#") or line.strip().startswith("**"):
            continue

        # Skip empty lines
        if not line.strip():
            continue

        # If we find a line that looks like essay content (starts with quote, letter, etc.)
        if line.strip() and not any(char in line for char in [":", "-", "*", "#"]):
            essay_start_idx = i
            break

    # Return content from the essay start
    if essay_start_idx > 0:
        essay_lines = lines[essay_start_idx:]
        return "\n".join(essay_lines).strip()

    # If no clear separation found, return original content
    return content.strip()


def extract_constraints_from_content(content: str) -> ContentConstraints:
    """
    Extract constraints dynamically from content context.

    Parses assignment context, prompts, and requirements embedded in the content
    to determine appropriate constraints and analysis focus. Includes robust
    error handling and validation.

    Args:
        content: Full content including context and essay text

    Returns:
        ContentConstraints with extracted requirements

    Example:
        >>> content = "Word Limit: 650 words\\nPrompt: Common App essay..."
        >>> constraints = extract_constraints_from_content(content)
        >>> constraints.word_limit
        650
        >>> constraints.essay_type
        'Common Application Personal Statement'
    """
    if not content or not content.strip():
        return ContentConstraints()

    constraints = ContentConstraints()

    try:
        # Extract word limits with validation
        word_limit_patterns = [
            r"Word Limit:\s*(\d+)\s*words?",
            r"Maximum:\s*(\d+)\s*words?",
            r"Limit:\s*(\d+)\s*words?",
            r"(\d+)\s*word\s*limit",
            r"(\d+)\s*words?\s*maximum",
        ]

        for pattern in word_limit_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                word_limit = int(match.group(1))
                # Validate word limit is reasonable
                if 10 <= word_limit <= 50000:
                    constraints.word_limit = word_limit
                    break
    except (ValueError, AttributeError):
        # Continue if word limit extraction fails
        pass

    try:
        # Extract character limits with validation
        char_limit_patterns = [
            r"Character Limit:\s*(\d+)",
            r"(\d+)\s*characters?\s*maximum",
            r"Maximum:\s*(\d+)\s*characters?",
            r"\((\d+)\s*characters?\s*or\s*fewer",  # (100 characters or fewer each)
            r"(\d+)\s*characters?\s*or\s*fewer",  # 100 characters or fewer
        ]

        for pattern in char_limit_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                char_limit = int(match.group(1))
                # Validate character limit is reasonable
                if 50 <= char_limit <= 200000:
                    constraints.character_limit = char_limit
                    break
    except (ValueError, AttributeError):
        # Continue if character limit extraction fails
        pass

    try:
        # Extract essay type with validation
        essay_type_patterns = [
            r"Essay Type:\s*([^\n]+)",
            r"Assignment:\s*([^\n]+)",
            r"Type:\s*([^\n]+)",
        ]

        for pattern in essay_type_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                essay_type = match.group(1).strip()
                # Validate essay type is not empty and reasonable length
                if essay_type and 3 <= len(essay_type) <= 200:
                    constraints.essay_type = essay_type
                    break
    except (AttributeError, TypeError):
        # Continue if essay type extraction fails
        pass

    try:
        # Extract school name with validation
        school_patterns = [
            r"(\w+)\s+Supplemental",
            r"School:\s*([^\n]+)",
            r"University:\s*([^\n]+)",
        ]

        for pattern in school_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                school_name = match.group(1).strip()
                # Validate school name is reasonable
                if school_name and 2 <= len(school_name) <= 100:
                    constraints.school_name = school_name
                    break
    except (AttributeError, TypeError):
        # Continue if school name extraction fails
        pass

    try:
        # Extract prompt with validation
        prompt_patterns = [
            r'Prompt:\s*["\']([^"\']+)["\']',
            r"Prompt:\s*([^\n]+)",
            r"Question:\s*([^\n]+)",
        ]

        for pattern in prompt_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                prompt = match.group(1).strip()
                # Validate prompt is reasonable length
                if prompt and 5 <= len(prompt) <= 1000:
                    constraints.prompt = prompt
                    break
    except (AttributeError, TypeError):
        # Continue if prompt extraction fails
        pass

    try:
        # Extract special requirements
        constraints.special_requirements = _extract_special_requirements(content)
    except Exception:
        # Ensure special_requirements is always a list
        constraints.special_requirements = []

    try:
        # Set optimal ranges based on essay type and limits
        constraints.optimal_word_range = _determine_optimal_range(
            constraints.word_limit, constraints.essay_type
        )
    except Exception:
        # Continue if optimal range calculation fails
        pass

    try:
        # Determine target audience
        constraints.target_audience = _determine_target_audience(constraints.essay_type)
    except Exception:
        # Continue if target audience determination fails
        pass

    return constraints


def get_analysis_types_for_content(constraints: ContentConstraints) -> list[str]:
    """
    Determine appropriate analysis types based on content constraints.

    Different essay types benefit from different analysis focuses:
    - Personal statements: structure, voice, strength, cliches
    - Supplementals: metrics, constraints (strict limits)
    - Technical content: readability, vocabulary

    Args:
        constraints: Extracted content constraints

    Returns:
        List of analysis types most relevant for this content
    """
    if not constraints.essay_type:
        return ["metrics", "readability", "structure"]

    essay_type_lower = constraints.essay_type.lower()

    # College application essays
    if any(
        term in essay_type_lower
        for term in ["common app", "personal statement", "application"]
    ):
        return ["metrics", "constraints", "structure", "strength", "cliches", "voice"]

    # Supplemental essays (usually shorter, strict limits)
    elif "supplemental" in essay_type_lower:
        return ["metrics", "constraints", "readability", "strength"]

    # Academic/research content
    elif any(term in essay_type_lower for term in ["research", "academic", "thesis"]):
        return ["metrics", "readability", "vocabulary", "structure"]

    # Business/professional content
    elif any(
        term in essay_type_lower for term in ["business", "professional", "proposal"]
    ):
        return ["metrics", "readability", "structure", "constraints"]

    # Technical documentation
    elif any(
        term in essay_type_lower for term in ["technical", "documentation", "api"]
    ):
        return ["metrics", "readability", "vocabulary", "structure"]

    # Creative writing
    elif any(term in essay_type_lower for term in ["creative", "story", "narrative"]):
        return ["metrics", "structure", "voice", "strength", "cliches"]

    # Default comprehensive analysis
    return ["metrics", "readability", "structure", "constraints"]


def _extract_special_requirements(content: str) -> list[str]:
    """Extract special requirements from content."""
    requirements = []

    # Common requirements patterns
    requirement_patterns = [
        r"Constraint:\s*([^\n]+)",
        r"Requirement:\s*([^\n]+)",
        r"Note:\s*([^\n]+)",
        r"Must:\s*([^\n]+)",
        r"Cannot:\s*([^\n]+)",
        r"Should:\s*([^\n]+)",
    ]

    for pattern in requirement_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        requirements.extend(matches)

    # Specific requirement detection
    if "cannot mention specific institutions" in content.lower():
        requirements.append(
            "Generic for all schools - no specific institution mentions"
        )

    if "work for all schools" in content.lower():
        requirements.append("Must be applicable to multiple institutions")

    return [req.strip() for req in requirements if req.strip()]


def _determine_optimal_range(
    word_limit: int | None, essay_type: str | None
) -> tuple[int, int] | None:
    """Determine optimal word range based on limit and type."""
    if not word_limit:
        return None

    if not essay_type:
        # Conservative range for unknown types
        return (int(word_limit * 0.85), word_limit)

    essay_type_lower = essay_type.lower()

    # Common App essays - use most of the space
    if "common app" in essay_type_lower or "personal statement" in essay_type_lower:
        return (int(word_limit * 0.92), word_limit)  # 600-650 for 650 limit

    # Short supplementals - be more conservative
    elif "supplemental" in essay_type_lower and word_limit <= 200:
        return (int(word_limit * 0.80), int(word_limit * 0.95))  # Leave some buffer

    # Medium supplementals - use most space
    elif "supplemental" in essay_type_lower:
        return (int(word_limit * 0.85), word_limit)

    # Default range
    return (int(word_limit * 0.85), word_limit)


def _determine_target_audience(essay_type: str | None) -> str | None:
    """Determine target audience based on essay type."""
    if not essay_type:
        return None

    essay_type_lower = essay_type.lower()

    if any(
        term in essay_type_lower for term in ["college", "university", "admissions"]
    ):
        return "College admissions officers"
    elif "supplemental" in essay_type_lower:
        return "University-specific admissions committee"
    elif any(term in essay_type_lower for term in ["business", "professional"]):
        return "Business professionals"
    elif any(term in essay_type_lower for term in ["academic", "research"]):
        return "Academic reviewers"
    elif any(term in essay_type_lower for term in ["technical", "documentation"]):
        return "Technical users and developers"

    return None


def format_constraints_for_prompt(constraints: ContentConstraints) -> str:
    """
    Format extracted constraints for inclusion in persona prompts.

    Args:
        constraints: Extracted content constraints

    Returns:
        Formatted string with constraint information
    """
    if not any(
        [
            constraints.word_limit,
            constraints.essay_type,
            constraints.special_requirements,
        ]
    ):
        return ""

    parts = []

    # Basic constraints
    if constraints.word_limit:
        parts.append(f"Word limit: {constraints.word_limit} words")
        if constraints.optimal_word_range:
            min_words, max_words = constraints.optimal_word_range
            parts.append(f"Optimal range: {min_words}-{max_words} words")

    if constraints.character_limit:
        parts.append(f"Character limit: {constraints.character_limit}")

    # Context information
    if constraints.essay_type:
        parts.append(f"Essay type: {constraints.essay_type}")

    if constraints.school_name:
        parts.append(f"School: {constraints.school_name}")

    if constraints.target_audience:
        parts.append(f"Target audience: {constraints.target_audience}")

    # Special requirements
    if constraints.special_requirements:
        parts.append("Special requirements:")
        for req in constraints.special_requirements:
            parts.append(f"  • {req}")

    return "\n".join(parts) if parts else ""
