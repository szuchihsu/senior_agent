"""
tools/github_tools.py

Low-level functions for talking to the GitHub API.
These are "tools" in the agent sense — small, focused functions
that do one thing and return structured data.

The Data Collection Agent calls these functions to:
  1. Get a list of recent commits from a repo
  2. Get the diff for each commit
  3. Get the CI test results for each commit
"""

import requests
from datetime import datetime
from config import GITHUB_TOKEN, TARGET_REPO


# ── Base HTTP helper ──────────────────────────────────────────────────────────

def _github_get(endpoint: str, params: dict = None) -> dict | list:
    """
    Makes an authenticated GET request to the GitHub REST API.

    All GitHub API calls go through this function so we handle
    auth and error checking in one place.

    Args:
        endpoint: The API path, e.g. "/repos/pallets/flask/commits"
        params:   Optional query parameters, e.g. {"per_page": 50}

    Returns:
        Parsed JSON response as a dict or list.
    """
    base_url = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(base_url + endpoint, headers=headers, params=params or {})
    response.raise_for_status()  # raises HTTPError if status >= 400
    return response.json()


# ── Commit fetching ───────────────────────────────────────────────────────────

def get_recent_commits(repo: str, max_count: int = 100) -> list[dict]:
    """
    Fetches recent commits from a GitHub repository.

    GitHub's API returns commits newest-first, paginated at up to
    100 per page. We fetch pages until we hit max_count.

    Args:
        repo:      "owner/repo" string, e.g. "pallets/flask"
        max_count: Maximum number of commits to return

    Returns:
        List of commit summaries: [{sha, message, date, author}, ...]

    Example return value:
        [
          {
            "sha": "a3f9c21...",
            "message": "Fix auth token expiry bug",
            "date": "2024-01-15T10:30:00Z",
            "author": "alice"
          },
          ...
        ]
    """
    commits = []
    page = 1

    while len(commits) < max_count:
        # GitHub returns at most 100 per page
        batch_size = min(100, max_count - len(commits))
        data = _github_get(
            f"/repos/{repo}/commits",
            params={"per_page": batch_size, "page": page}
        )

        if not data:  # empty page means we've reached the end
            break

        for item in data:
            commits.append({
                "sha": item["sha"],
                "message": item["commit"]["message"].split("\n")[0],  # first line only
                "date": item["commit"]["author"]["date"],
                "author": item["commit"]["author"]["name"],
            })

        page += 1

    return commits[:max_count]


def get_commit_diff(repo: str, commit_sha: str) -> str:
    """
    Fetches the full git diff for a single commit.

    This is the raw "what changed" text — the same thing you'd see
    running `git show <sha>` in the terminal.

    We use a special Accept header to get the diff format instead of JSON.

    Args:
        repo:       "owner/repo"
        commit_sha: The full or partial commit hash

    Returns:
        Raw diff text as a string. Empty string if the commit has no diff
        (e.g. merge commits with no changes).
    """
    base_url = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        # This special media type tells GitHub to return the diff format
        "Accept": "application/vnd.github.diff",
    }
    response = requests.get(
        f"{base_url}/repos/{repo}/commits/{commit_sha}",
        headers=headers
    )
    response.raise_for_status()
    return response.text


# ── CI / GitHub Actions data ──────────────────────────────────────────────────

def get_test_results_for_commit(repo: str, commit_sha: str) -> dict:
    """
    Fetches CI test results associated with a commit from GitHub Actions.

    GitHub Actions organizes CI data as:
      Workflow Run → Jobs → Steps

    We look for workflow runs triggered by this commit, then parse
    the job results to find which tests passed and failed.

    This is the trickiest part of data collection because:
      - Not all repos parse individual test names from CI logs
      - Some repos only report "job passed/failed", not test-level detail
      - We need to parse job names as a proxy for test names

    Args:
        repo:       "owner/repo"
        commit_sha: The commit hash to look up CI results for

    Returns:
        {
          "tests_run": ["job_name_1", "job_name_2", ...],
          "tests_failed": ["job_name_2", ...]
        }

        Note: For simplicity, we treat CI job names as "tests".
        A future improvement would parse actual pytest output from logs.
    """
    # Get workflow runs associated with this commit
    runs_data = _github_get(
        f"/repos/{repo}/actions/runs",
        params={"head_sha": commit_sha, "per_page": 10}
    )

    tests_run = []
    tests_failed = []

    workflow_runs = runs_data.get("workflow_runs", [])

    for run in workflow_runs:
        run_id = run["id"]

        # Get the individual jobs within this workflow run
        jobs_data = _github_get(f"/repos/{repo}/actions/runs/{run_id}/jobs")

        for job in jobs_data.get("jobs", []):
            job_name = job["name"]
            tests_run.append(job_name)

            # A job "failed" if its conclusion is failure or cancelled
            if job["conclusion"] in ("failure", "cancelled", "timed_out"):
                tests_failed.append(job_name)

    return {
        "tests_run": list(set(tests_run)),      # deduplicate
        "tests_failed": list(set(tests_failed))
    }
