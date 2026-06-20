"""Application logging setup.

Purpose
-------
Configures consistent, application-wide logging and provides a helper to fetch
named loggers, so every module logs in the same format.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging.

    This function is safe to call during FastAPI startup. It avoids duplicate
    handlers when the app reloads during development.
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name:
            Usually __name__ from the calling module.
    """

    return logging.getLogger(name)