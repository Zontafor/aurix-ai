# tests/conftest.py
"""
Ensures pytest can import the `src` package regardless of where
pytest is executed from. This is required for imports such as:

    from src.models.profit_loss import ...

Without this file, pytest uses its own working directory and
fails to resolve the project root.
"""

import os
import sys
import pytest
import requests
from pathlib import Path

# Path to project root: .../project/code/
ROOT = Path(__file__).resolve().parents[2]

root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

@pytest.fixture(scope="session")
def base_url() -> str:
    # Example: export AURIX_API_BASE_URL="http://127.0.0.1:8000"
    return os.getenv("AURIX_API_BASE_URL", "http://127.0.0.1:8080")

@pytest.fixture()
def client():
    return requests.Session()
