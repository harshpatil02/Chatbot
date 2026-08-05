from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None, logfile: Optional[str] = None) -> None:
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, level, logging.INFO)

    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=numeric_level, format=fmt, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().hasHandlers():
        configure_logging()
    return logging.getLogger(name)
