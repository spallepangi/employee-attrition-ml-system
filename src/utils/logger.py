"""
Structured logging utility for the Employee Attrition Prediction System.
Used for training, evaluation, and API prediction logs.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a configured logger with optional file and console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to a file.
        format_string: Optional custom format. Default includes timestamp, name, level, message.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding handlers multiple times when called with same name
    if logger.handlers:
        return logger

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_training_start(logger: logging.Logger, model_name: str, **kwargs: Any) -> None:
    """Log start of model training with optional key-value context."""
    logger.info("Training started: model=%s", model_name, extra=kwargs)
    for key, value in kwargs.items():
        logger.info("  %s: %s", key, value)


def log_training_end(
    logger: logging.Logger,
    model_name: str,
    metrics: Optional[dict] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """Log end of model training with metrics and duration."""
    logger.info("Training completed: model=%s", model_name)
    if duration_seconds is not None:
        logger.info("  duration_seconds: %.2f", duration_seconds)
    if metrics:
        for metric_name, value in metrics.items():
            logger.info("  %s: %.4f", metric_name, value)


def log_prediction(
    logger: logging.Logger,
    prediction: int,
    probability: float,
    request_id: Optional[str] = None,
) -> None:
    """Log a single prediction for auditing and debugging."""
    logger.info(
        "Prediction: prediction=%s probability=%.4f request_id=%s",
        prediction,
        probability,
        request_id or "N/A",
    )
