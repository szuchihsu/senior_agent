"""
config.py

Central configuration for the entire project.
All API keys, file paths, and tunable parameters live here.

We use python-dotenv to load secrets from a .env file so
they never get hardcoded into the source code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root into environment variables
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────

# Your Anthropic API key — used by the Explanation Agent to call Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Your GitHub personal access token — used by the Data Collection Agent
# Needs: repo (read) + actions (read) scopes
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ── Target Repository ─────────────────────────────────────────────────────────

# The open-source repo we're collecting CI history from.
# Format: "owner/repo"  e.g. "pallets/flask" or "django/django"
TARGET_REPO = os.getenv("TARGET_REPO", "pallets/flask")

# ── File Paths ────────────────────────────────────────────────────────────────

# Root of this project
PROJECT_ROOT = Path(__file__).parent

# Where raw CI data (JSON) gets saved after collection
RAW_DATA_DIR = PROJECT_ROOT / "storage" / "raw_data"

# Where ChromaDB stores its vector index
VECTOR_DB_DIR = PROJECT_ROOT / "storage" / "vector_db"

# ── Data Collection Settings ──────────────────────────────────────────────────

# How many commits to pull from GitHub history
# Start small (50-100) while testing, go bigger (500+) for real evaluation
MAX_COMMITS_TO_COLLECT = int(os.getenv("MAX_COMMITS", "100"))

# ── Retrieval Settings ────────────────────────────────────────────────────────

# How many similar historical diffs to retrieve for a new diff
# More = more evidence, but also more noise and slower ranking
TOP_K_SIMILAR = int(os.getenv("TOP_K", "20"))

# ── Ranking Settings ──────────────────────────────────────────────────────────

# How many tests to include in the final ranked report
# (we don't show all tests, just the most suspicious ones)
TOP_N_TESTS_TO_REPORT = int(os.getenv("TOP_N", "12"))

# Minimum failure score to include a test in the report
# Tests below this threshold are considered "safe to skip"
MIN_FAILURE_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE", "0.1"))

# ── Model Settings ────────────────────────────────────────────────────────────

# Claude model used by the Explanation Agent
# claude-opus-4-6 is the most capable; use claude-haiku-4-5 if cost is a concern
CLAUDE_MODEL = "claude-opus-4-6"

# Embedding model — we use Claude's embedding endpoint
# This converts a diff (text) into a fixed-size vector of floats
EMBEDDING_MODEL = "voyage-code-3"  # Voyage AI's code-specific embedding model
# Alternative: use OpenAI's text-embedding-3-small if you prefer
