"""
Logging utilities for tools in the Review-Crew system.

Provides decorators and utilities to add consistent logging to tool functions.
"""

import functools
import json
import time
from typing import Any, Callable, TypeVar

from ..logging.manager import LoggingManager

F = TypeVar("F", bound=Callable[..., Any])


def log_tool_execution(tool_name: str) -> Callable[[F], F]:
    """Decorator to log tool execution with parameters and results.

    Args:
        tool_name: Name of the tool for logging purposes

    Returns:
        Decorated function with logging
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Get the tool logger
                logging_manager = LoggingManager.get_instance()
                logger = logging_manager.get_tool_logger(tool_name)

                # Log tool execution start
                start_time = time.time()

                # Prepare parameters for logging (truncate large content)
                log_args = []
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 200:
                        log_args.append(
                            f"{arg[:200]}... (truncated, total length: {len(arg)})"
                        )
                    else:
                        log_args.append(str(arg)[:100])

                log_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str) and len(value) > 200:
                        log_kwargs[key] = (
                            f"{value[:200]}... (truncated, total length: {len(value)})"
                        )
                    else:
                        log_kwargs[key] = str(value)[:100]

                logger.info(f"[TOOL_START] {tool_name}")
                logger.info(f"Parameters - Args: {log_args}, Kwargs: {log_kwargs}")

                # Execute the tool
                result = func(*args, **kwargs)

                # Log execution completion
                execution_time = time.time() - start_time
                logger.info(
                    f"[TOOL_COMPLETE] {tool_name} - Execution time: {execution_time:.3f}s"
                )

                # Log result summary (avoid logging huge results)
                if hasattr(result, "__dict__"):
                    # For dataclass objects, log the field names and values (truncated if needed)
                    result_summary = {}
                    for field in result.__dict__.keys():
                        value = getattr(result, field)
                        if isinstance(value, str) and len(value) > 100:
                            result_summary[field] = f"{value[:100]}... (truncated)"
                        elif isinstance(value, (list, dict)) and len(str(value)) > 200:
                            result_summary[field] = (
                                f"{type(value).__name__} with {len(value)} items"
                            )
                        else:
                            result_summary[field] = value
                    logger.info(f"Result summary: {result_summary}")
                else:
                    result_str = str(result)
                    if len(result_str) > 200:
                        logger.info(f"Result: {result_str[:200]}... (truncated)")
                    else:
                        logger.info(f"Result: {result_str}")

                return result

            except RuntimeError:
                # No active logging session - execute without logging
                return func(*args, **kwargs)
            except Exception as e:
                # Log error if possible, then re-raise
                try:
                    logging_manager = LoggingManager.get_instance()
                    logger = logging_manager.get_tool_logger(tool_name)
                    logger.error(f"[TOOL_ERROR] {tool_name} failed: {str(e)}")
                except:
                    pass
                raise

        return wrapper  # type: ignore

    return decorator


def log_tool_result(
    tool_name: str, parameters: dict[str, Any], result: Any, execution_time: float
) -> None:
    """Log tool execution result.

    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        result: Tool result
        execution_time: Execution time in seconds
    """
    try:
        logging_manager = LoggingManager.get_instance()
        logger = logging_manager.get_tool_logger(tool_name)

        # Create log entry
        log_entry = {
            "tool": tool_name,
            "execution_time_seconds": execution_time,
            "parameters": _sanitize_for_logging(parameters),
            "result_type": type(result).__name__,
            "timestamp": time.time(),
        }

        logger.info(f"Tool execution: {json.dumps(log_entry, default=str)}")

    except RuntimeError:
        # No active logging session - skip logging
        pass


def _sanitize_for_logging(obj: Any, max_length: int = 200) -> Any:
    """Sanitize object for logging by truncating long strings.

    Args:
        obj: Object to sanitize
        max_length: Maximum string length before truncation

    Returns:
        Sanitized object
    """
    if isinstance(obj, str):
        if len(obj) > max_length:
            return f"{obj[:max_length]}... (truncated, total length: {len(obj)})"
        return obj
    elif isinstance(obj, dict):
        return {
            key: _sanitize_for_logging(value, max_length) for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_logging(item, max_length) for item in obj]
    else:
        return obj
