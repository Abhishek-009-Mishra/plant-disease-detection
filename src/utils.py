"""
utils.py
--------
Small, shared helper functions used across multiple modules.
Keeping these here avoids duplicating logging setup / formatting code
in every file (dataset_analysis.py, train.py, predictor.py, etc.).
"""

import logging
import sys
from pathlib import Path

from src.config import config


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Every module calls `get_logger(__name__)` instead of setting up its
    own handlers, so log formatting stays consistent project-wide and the
    verbosity is controlled from one place (config.LOG_LEVEL).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated imports
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    return logger


def ensure_dir(path: str) -> Path:
    """Create a directory (including parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_bytes(num_bytes: float) -> str:
    """Human-readable byte size, e.g. 1536 -> '1.5 KB'. Used in reports."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def print_banner(title: str, width: int = 60) -> None:
    """Consistent CLI section banner used by main.py and other CLI commands."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)
