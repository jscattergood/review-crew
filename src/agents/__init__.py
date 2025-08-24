"""Agent implementations for Review-Crew."""

# Import core data structures that don't require external dependencies
from .conversation_manager import ReviewResult, ConversationResult

# Import analysis components
try:
    from .analysis_agent import AnalysisAgent, AnalysisResult
    _analysis_available = True
except ImportError:
    _analysis_available = False

# Import main components that may require external dependencies
try:
    from .review_agent import ReviewAgent
    from .conversation_manager import ConversationManager
    _main_components_available = True
except ImportError as e:
    _main_components_available = False
    _import_error = e

# Define what's available for import
__all__ = ["ReviewResult", "ConversationResult"]

if _analysis_available:
    __all__.extend(["AnalysisAgent", "AnalysisResult"])

if _main_components_available:
    __all__.extend(["ReviewAgent", "ConversationManager"])
else:
    # Provide helpful error message for missing dependencies
    def _missing_dependency_error():
        raise ImportError(f"Main components not available due to missing dependencies: {_import_error}")
    
    ReviewAgent = _missing_dependency_error
    ConversationManager = _missing_dependency_error