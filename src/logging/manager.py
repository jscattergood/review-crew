"""
Centralized logging manager for Review-Crew system.

This module provides session-based logging with timestamped directories
for each review process, organizing logs by agents, conversation, and tools.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class LoggingManager:
    """Manages session-based logging for the Review-Crew system."""

    _instance: Optional["LoggingManager"] = None

    def __init__(self, base_log_dir: str = "logs"):
        """Initialize the logging manager.

        Args:
            base_log_dir: Base directory for all logs
        """
        self.base_log_dir = Path(base_log_dir)
        self.session_id: str | None = None
        self.session_dir: Path | None = None
        self.session_info: dict[str, Any] = {}
        self.loggers: dict[str, logging.Logger] = {}

    @classmethod
    def get_instance(cls) -> "LoggingManager":
        """Get the singleton instance of LoggingManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def start_session(
        self,
        content_info: str = "",
        selected_agents: list[str] | None = None,
        model_provider: str = "bedrock",
        model_config: dict[str, Any] | None = None,
    ) -> str:
        """Start a new logging session.

        Args:
            content_info: Information about the content being reviewed
            selected_agents: List of selected agent names
            model_provider: Model provider being used
            model_config: Model configuration

        Returns:
            Session ID
        """
        # Generate session ID with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_id = f"session_{timestamp}"

        # Create session directory structure
        self.session_dir = self.base_log_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "agents").mkdir(exist_ok=True)
        (self.session_dir / "tools").mkdir(exist_ok=True)

        # Store session information
        self.session_info = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "content_info": content_info,
            "selected_agents": selected_agents or [],
            "model_provider": model_provider,
            "model_config": model_config or {},
            "end_time": None,
            "duration_seconds": None,
        }

        # Create session info file
        self._save_session_info()

        # Create symlink to latest session
        self._update_latest_symlink()

        # Set up conversation logger
        self._setup_conversation_logger()

        return self.session_id

    def end_session(self) -> None:
        """End the current logging session."""
        if self.session_info:
            self.session_info["end_time"] = datetime.now().isoformat()

            # Calculate duration
            start_time = datetime.fromisoformat(self.session_info["start_time"])
            end_time = datetime.fromisoformat(self.session_info["end_time"])
            duration = (end_time - start_time).total_seconds()
            self.session_info["duration_seconds"] = duration

            self._save_session_info()

        # Clean up loggers
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        self.loggers.clear()

    def get_agent_logger(self, agent_name: str) -> logging.Logger:
        """Get or create a logger for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Logger instance for the agent
        """
        if not self.session_dir:
            raise RuntimeError("No active logging session. Call start_session() first.")

        logger_key = f"agent_{agent_name}"

        if logger_key not in self.loggers:
            # Create logger
            logger = logging.getLogger(logger_key)
            logger.setLevel(logging.INFO)

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create file handler
            safe_agent_name = agent_name.replace(" ", "_").lower()
            log_file = self.session_dir / "agents" / f"{safe_agent_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(file_handler)

            # Prevent propagation to root logger
            logger.propagate = False

            self.loggers[logger_key] = logger

        return self.loggers[logger_key]

    def get_conversation_logger(self) -> logging.Logger:
        """Get the conversation logger.

        Returns:
            Logger instance for conversation events
        """
        if not self.session_dir:
            raise RuntimeError("No active logging session. Call start_session() first.")

        return self.loggers.get("conversation", logging.getLogger("conversation"))

    def get_tool_logger(self, tool_name: str) -> logging.Logger:
        """Get or create a logger for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Logger instance for the tool
        """
        if not self.session_dir:
            raise RuntimeError("No active logging session. Call start_session() first.")

        logger_key = f"tool_{tool_name}"

        if logger_key not in self.loggers:
            # Create logger
            logger = logging.getLogger(logger_key)
            logger.setLevel(logging.INFO)

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Create file handler
            safe_tool_name = tool_name.replace(" ", "_").lower()
            log_file = self.session_dir / "tools" / f"{safe_tool_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(file_handler)

            # Prevent propagation to root logger
            logger.propagate = False

            self.loggers[logger_key] = logger

        return self.loggers[logger_key]

    def log_conversation_event(
        self, event: str, details: dict[str, Any] | None = None
    ) -> None:
        """Log a conversation-level event.

        Args:
            event: Event description
            details: Optional additional details
        """
        logger = self.get_conversation_logger()
        message = event
        if details:
            message += f" - Details: {json.dumps(details, default=str)}"
        logger.info(message)

    def get_session_dir(self) -> Path | None:
        """Get the current session directory.

        Returns:
            Path to current session directory, or None if no active session
        """
        return self.session_dir

    def get_session_id(self) -> str | None:
        """Get the current session ID.

        Returns:
            Current session ID, or None if no active session
        """
        return self.session_id

    def _setup_conversation_logger(self) -> None:
        """Set up the conversation logger."""
        if not self.session_dir:
            return

        logger = logging.getLogger("conversation")
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create file handler
        log_file = self.session_dir / "conversation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        self.loggers["conversation"] = logger

    def _save_session_info(self) -> None:
        """Save session information to JSON file."""
        if not self.session_dir:
            return

        info_file = self.session_dir / "session_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(self.session_info, f, indent=2, default=str)

    def _update_latest_symlink(self) -> None:
        """Update the 'latest' symlink to point to current session."""
        if not self.session_dir or not self.session_id:
            return

        latest_link = self.base_log_dir / "latest"

        # Remove existing symlink if it exists
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()

        # Create new symlink (relative path for portability)
        try:
            latest_link.symlink_to(self.session_id)
        except OSError:
            # Symlinks might not be supported on all systems
            pass
