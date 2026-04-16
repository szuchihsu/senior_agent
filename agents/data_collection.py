"""
agents/data_collection.py

The Data Collection Agent.

Responsibility: Fetch historical commit + CI data from GitHub and
save it as raw JSON to disk. This is a one-time (or periodic) setup step —
you run this before you can make any predictions.

The agent works like this:
  1. Fetch a list of recent commits from the target repo
  2. For each commit, fetch its diff and its CI test results
  3. Save everything as a list of JSON records to disk

Why save to disk first (instead of directly to ChromaDB)?
  - Easier to inspect and debug the raw data
  - Allows re-embedding without re-fetching from GitHub (saves API quota)
  - If something fails mid-run, you don't lose everything
"""

import json
import time
from pathlib import Path
from datetime import datetime

from tools.github_tools import get_recent_commits, get_commit_diff, get_test_results_for_commit
from tools.diff_tools import clean_diff
from config import TARGET_REPO, MAX_COMMITS_TO_COLLECT, RAW_DATA_DIR


def run(repo: str = None, max_commits: int = None) -> list[dict]:
    """
    Main entry point for the Data Collection Agent.

    Fetches commits from GitHub, collects their diffs and CI results,
    and saves them to storage/raw_data/<repo_name>.json.

    Args:
        repo:        Override the target repo (default: from config.py)
        max_commits: Override how many commits to collect

    Returns:
        List of raw record dicts (also saved to disk).

    Usage:
        from agents import data_collection
        records = data_collection.run()
    """
    repo = repo or TARGET_REPO
    max_commits = max_commits or MAX_COMMITS_TO_COLLECT

    print(f"[DataCollection] Starting collection for {repo} (max {max_commits} commits)")

    # Step 1: Get a list of commit SHAs and metadata
    print(f"[DataCollection] Fetching commit list...")
    commits = get_recent_commits(repo, max_count=max_commits)
    print(f"[DataCollection] Found {len(commits)} commits")

    records = []

    for i, commit in enumerate(commits):
        sha = commit["sha"]
        print(f"[DataCollection] [{i+1}/{len(commits)}] Processing {sha[:8]}...")

        try:
            # Step 2a: Fetch the raw git diff for this commit
            raw_diff = get_commit_diff(repo, sha)

            # Step 2b: Clean the diff (remove noise, truncate)
            # We store BOTH raw and cleaned so we can re-clean later
            cleaned_diff = clean_diff(raw_diff)

            # Skip commits with no meaningful diff (e.g. merge commits, doc-only changes)
            if not cleaned_diff.strip():
                print(f"[DataCollection]   Skipping — empty diff after cleaning")
                continue

            # Step 2c: Fetch CI test results for this commit
            ci_results = get_test_results_for_commit(repo, sha)

            # Skip commits with no CI data (e.g. very old commits before CI was set up)
            if not ci_results["tests_run"]:
                print(f"[DataCollection]   Skipping — no CI data")
                continue

            record = {
                "commit_sha": sha,
                "message": commit["message"],
                "author": commit["author"],
                "timestamp": commit["date"],
                "diff_text": cleaned_diff,       # cleaned diff (for embedding)
                "raw_diff": raw_diff,            # original (for reference)
                "tests_run": ci_results["tests_run"],
                "tests_failed": ci_results["tests_failed"],
            }

            records.append(record)

            failure_count = len(ci_results["tests_failed"])
            run_count = len(ci_results["tests_run"])
            print(f"[DataCollection]   {run_count} tests run, {failure_count} failed")

        except Exception as e:
            # Don't crash the whole run if one commit fails
            print(f"[DataCollection]   Error processing {sha[:8]}: {e}")
            continue

        # GitHub API has rate limits: 5000 requests/hour for authenticated users
        # With 3 API calls per commit, that's ~1666 commits/hour
        # Sleep briefly to be polite and avoid hitting limits
        time.sleep(0.5)

    # Step 3: Save to disk
    _save_records(repo, records)

    print(f"[DataCollection] Done. Collected {len(records)} usable records.")
    return records


def _save_records(repo: str, records: list[dict]) -> Path:
    """
    Saves collected records to a JSON file.

    The filename is based on the repo name so you can collect from
    multiple repos without overwriting each other.

    Args:
        repo:    "owner/repo" string
        records: List of record dicts to save

    Returns:
        Path to the saved file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # "pallets/flask" → "pallets_flask.json"
    filename = repo.replace("/", "_") + ".json"
    filepath = RAW_DATA_DIR / filename

    with open(filepath, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"[DataCollection] Saved {len(records)} records to {filepath}")
    return filepath


def load_records(repo: str = None) -> list[dict]:
    """
    Loads previously collected records from disk.

    Called by the Embedding Agent — it doesn't re-fetch from GitHub,
    it just reads what the Data Collection Agent already saved.

    Args:
        repo: "owner/repo" string (default: from config.py)

    Returns:
        List of record dicts, or empty list if no data found.
    """
    repo = repo or TARGET_REPO
    filename = repo.replace("/", "_") + ".json"
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        print(f"[DataCollection] No data found at {filepath}. Run data_collection.run() first.")
        return []

    with open(filepath) as f:
        records = json.load(f)

    print(f"[DataCollection] Loaded {len(records)} records from {filepath}")
    return records
