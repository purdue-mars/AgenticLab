"""
Simple logging utility for AgenticLab

Usage:
    from agenticlab.utils.log_utils import info, error, logger

    # Direct usage
    info("Program started")
    error("An error occurred")

    # Or use logger object
    logger.info("Detection started")
    logger.warning("Warning message")

    # Custom logger with no console output
    from agenticlab.utils.log_utils import setup_logger
    custom_logger = setup_logger(console_output=False)
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name="agenticlab", level="INFO", log_dir="logs", console_output=True):
    """
    Setup a simple logger

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Log file directory
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Console output (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File output
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create default logger (with console output)
logger = setup_logger()


# Convenience functions
def debug(msg):
    logger.debug(msg)


def info(msg):
    logger.info(msg)


def warning(msg):
    logger.warning(msg)


def error(msg):
    logger.error(msg)
