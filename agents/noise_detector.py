"""
agents/noise_detector.py

The CI Noise Detector Agent.

Responsibility: Analyze all CI job names collected from a repo and use Claude
to identify which ones are infrastructure or bot jobs whose failures are NOT
caused by code changes. Save the result as a repo-specific blocklist.

Why this matters:
  Hardcoded regex filters (like FLAKY_JOB_PATTERNS in github_tools.py) are
  generic guesses. Every repo names its jobs differently:
    - "docs-build" vs "check-links" vs "CI / Documentation"
    - "Dependabot" vs "renovate" vs "update-deps"
  This agent reads the actual job names from your collected data and asks
  Claude to make the call — no manual pattern-writing needed.

When it runs:
  After data_collection.run(), before embedding.run().
  The blocklist is saved to storage/noise_blocklists/ and automatically
  applied whenever records are loaded for embedding or evaluation.

What it detects:
  - Documentation link checkers
  - Dependency update bots (Dependabot, Renovate)
  - Stale issue / PR bots
  - Label managers, greeting bots
  - Spell / prose checkers
  - Platform-specific flaky combinations (e.g. PyPy on Windows)
  - Any job that clearly runs on a schedule, not triggered by code changes

What it cannot detect:
  - Statistically flaky jobs (randomly fail ~5% of the time regardless of diff)
    — that would require a separate analysis of failure rates vs. similarity scores
  - Jobs that are real tests but happen to have infrastructure-sounding names
"""

import json
from pathlib import Path

import anthropic

from agents.data_collection import load_records
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TARGET_REPO


BLOCKLIST_DIR = Path("storage/noise_blocklists")


def run(repo: str = None) -> list[str]:
    """
    Main entry point for the Noise Detector Agent.

    Loads collected records, extracts all unique job names, asks Claude to
    classify them, and saves the resulting blocklist to disk.

    Args:
        repo: "owner/repo" (default: from config.py)

    Returns:
        List of job name strings identified as noise/infra jobs.

    Usage:
        from agents import noise_detector
        noisy = noise_detector.run()
        print(f"Blocked {len(noisy)} noisy jobs")
    """
    repo = repo or TARGET_REPO

    print(f"\n[NoiseDetector] Analyzing CI job names for {repo}...")

    records = load_records(repo)
    if not records:
        print("[NoiseDetector] No records found — skipping noise detection.")
        return []

    # Count how often each job ran and failed across all commits
    job_stats: dict[str, dict] = {}
    for record in records:
        for job in record.get("tests_run", []):
            if job not in job_stats:
                job_stats[job] = {"runs": 0, "failures": 0}
            job_stats[job]["runs"] += 1
        for job in record.get("tests_failed", []):
            if job in job_stats:
                job_stats[job]["failures"] += 1

    if not job_stats:
        print("[NoiseDetector] No job names found in records.")
        return []

    print(f"[NoiseDetector] Found {len(job_stats)} unique job names across {len(records)} commits")

    # Ask Claude to classify the job names
    try:
        noisy_jobs = _classify_with_claude(job_stats)
    except Exception as e:
        print(f"[NoiseDetector] Warning: Claude classification failed ({e})")
        print("[NoiseDetector] Falling back to empty blocklist — no jobs filtered.")
        noisy_jobs = []

    # Save the blocklist to disk for future use
    _save_blocklist(repo, noisy_jobs)

    if noisy_jobs:
        print(f"[NoiseDetector] Identified {len(noisy_jobs)} noisy jobs to exclude:")
        for job in noisy_jobs:
            print(f"[NoiseDetector]   - {job}")
    else:
        print("[NoiseDetector] No noisy jobs identified — all jobs look like real tests.")

    return noisy_jobs


def _classify_with_claude(job_stats: dict[str, dict]) -> list[str]:
    """
    Sends the job name list to Claude and asks it to identify noise jobs.

    We include failure rates in the prompt so Claude can spot jobs that
    fail very frequently (likely infra) or never (maybe not a real test).

    Returns:
        List of job names Claude identified as infrastructure/noise.
    """
    # Format the job list with stats for Claude
    job_lines = []
    for job_name, stats in sorted(job_stats.items()):
        failure_rate = stats["failures"] / stats["runs"] if stats["runs"] > 0 else 0
        job_lines.append(
            f"- {job_name}  "
            f"(ran {stats['runs']}x, failed {stats['failures']}x, {failure_rate:.0%} failure rate)"
        )
    job_list = "\n".join(job_lines)

    prompt = f"""You are analyzing CI/CD job names from a GitHub repository.
Your task: identify which jobs are infrastructure, bot, or maintenance jobs
whose failures are NOT caused by code changes.

Here are all the CI jobs seen across recent commits:

{job_list}

Jobs to flag as noise include:
- Documentation link checkers or doc build jobs that check for broken URLs
- Dependency update bots (Dependabot, Renovate, update-deps)
- Stale issue/PR bots, label managers, auto-assign, greeting bots
- Spell checkers, prose linters
- Scheduled maintenance jobs
- Platform-specific combinations known to be flaky (e.g. PyPy on Windows)
- Any job whose name clearly indicates it runs on a schedule, not on code push

Do NOT flag:
- Unit tests, integration tests, end-to-end tests
- Build and compile jobs
- Linting and type checking jobs (flake8, mypy, pylint)
- Code coverage jobs
- Security scanning jobs that check code (not just links)

Return ONLY a JSON array of the exact job names to exclude.
If all jobs look like real test/build/lint jobs, return an empty array [].

Return only the JSON array with no explanation, preamble, or markdown fences."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude included them despite being told not to
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    noisy_jobs: list[str] = json.loads(raw)

    # Safety check: only keep names that actually exist in our job list
    # (prevents Claude from hallucinating job names)
    known_jobs = set(job_stats.keys())
    noisy_jobs = [j for j in noisy_jobs if j in known_jobs]

    return noisy_jobs


def _save_blocklist(repo: str, noisy_jobs: list[str]) -> None:
    """Saves the blocklist to storage/noise_blocklists/<repo>_blocklist.json."""
    BLOCKLIST_DIR.mkdir(parents=True, exist_ok=True)
    filename = repo.replace("/", "_") + "_blocklist.json"
    filepath = BLOCKLIST_DIR / filename
    with open(filepath, "w") as f:
        json.dump({"repo": repo, "noisy_jobs": noisy_jobs}, f, indent=2)
    print(f"[NoiseDetector] Blocklist saved to {filepath}")


def load_blocklist(repo: str = None) -> set[str]:
    """
    Loads the saved blocklist for a repo.

    Called by the Embedding Agent and evaluate.py to filter records
    before training/evaluation.

    Returns:
        Set of noisy job name strings. Empty set if no blocklist exists.
    """
    repo = repo or TARGET_REPO
    filename = repo.replace("/", "_") + "_blocklist.json"
    filepath = BLOCKLIST_DIR / filename

    if not filepath.exists():
        return set()

    with open(filepath) as f:
        data = json.load(f)

    return set(data.get("noisy_jobs", []))


def apply_blocklist(records: list[dict], blocklist: set[str]) -> list[dict]:
    """
    Removes noisy jobs from the tests_run and tests_failed fields of each record.

    Called after load_records() in the embedding and evaluation pipelines.

    Args:
        records:   List of raw record dicts from data_collection
        blocklist: Set of job names to exclude (from load_blocklist())

    Returns:
        Filtered list of records with noisy jobs removed.
    """
    if not blocklist:
        return records

    filtered = []
    for record in records:
        r = dict(record)  # shallow copy so we don't mutate the original
        r["tests_run"] = [t for t in r.get("tests_run", []) if t not in blocklist]
        r["tests_failed"] = [t for t in r.get("tests_failed", []) if t not in blocklist]
        # Keep the record only if it still has CI data after filtering
        if r["tests_run"]:
            filtered.append(r)

    removed = len(records) - len(filtered)
    if removed:
        print(f"[NoiseDetector] Filtered out {len(blocklist)} noisy job types; "
              f"dropped {removed} records with no remaining CI data")

    return filtered
