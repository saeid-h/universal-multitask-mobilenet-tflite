"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

# Add project root to sys.path so `from src.utils...` and `from models...` work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
