"""Guardrail RAG application package."""

from .logging_config import configure_logging

# Configure logging on package import (can be overridden by env vars)
configure_logging()
