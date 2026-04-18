"""
agents/orchestrator.py

The Orchestrator Agent.

Responsibility: Coordinate all the other agents to produce a final
prediction report from a raw diff.

This is the entry point for the "predict" phase (not the "build" phase).
It assumes the vector DB is already populated (data_collection + embedding
have been run).

The orchestrator:
  1. Receives a diff (from CLI, GitHub webhook, or git hook)
  2. Calls retrieval → ranking → explanation in sequence
  3. Returns a PredictionReport with the final ranked + explained test list

Why have an orchestrator?
  - Single entry point: callers don't need to know about the internal agents
  - Error handling in one place: if retrieval finds nothing, we return early
  - Easy to add steps later (e.g. caching, logging, Slack notifications)
"""

from datetime import datetime, timezone
from models.schemas import PredictionReport
from tools.vector_tools import get_collection_size
from agents import retrieval, ranking, explanation
from config import TOP_K_SIMILAR, TOP_N_TESTS_TO_REPORT, TARGET_REPO


def predict(diff_text: str, top_k: int = None, top_n: int = None, repo: str = None) -> PredictionReport:
    """
    Main entry point: given a diff, predict which tests are most likely to fail.

    This is the function you call from main.py, a GitHub Action, or a git hook.

    Args:
        diff_text: Raw git diff string. Get this by running:
                   `git diff main...HEAD` or `git show <sha>`
        top_k:     How many similar historical diffs to retrieve (default: config)
        top_n:     How many tests to include in the final report (default: config)

    Returns:
        A PredictionReport with ranked tests and explanations.
        Returns an empty report if the DB is empty or the diff has no meaningful changes.

    Example:
        import subprocess
        diff = subprocess.check_output(["git", "diff", "main...HEAD"]).decode()

        from agents.orchestrator import predict
        report = predict(diff)

        for test in report.top_n(5):
            print(f"{test.failure_score:.0%} - {test.test_name}")
            print(f"  → {test.explanation}")
    """
    top_k = top_k or TOP_K_SIMILAR
    top_n = top_n or TOP_N_TESTS_TO_REPORT
    repo = repo or TARGET_REPO

    print("\n" + "="*60)
    print("REGRESSION PREDICTOR — Analyzing diff...")
    print("="*60)

    # Sanity check: make sure we have data to search against
    db_size = get_collection_size(repo)
    if db_size == 0:
        print("[Orchestrator] ERROR: Vector DB is empty.")
        print("[Orchestrator] Run: python main.py build")
        return _empty_report(diff_text)

    print(f"[Orchestrator] DB contains {db_size} historical diffs")

    # ── Step 1: Retrieval ──────────────────────────────────────────────────────
    # Find the most similar historical diffs
    print("\n[Orchestrator] Step 1/3: Retrieving similar historical diffs...")
    similar_diffs = retrieval.run(diff_text, top_k=top_k, repo=repo)

    if not similar_diffs:
        print("[Orchestrator] No similar diffs found. Cannot make predictions.")
        return _empty_report(diff_text)

    # ── Step 2: Ranking ────────────────────────────────────────────────────────
    # Compute failure scores for each test
    print("\n[Orchestrator] Step 2/3: Ranking tests by failure likelihood...")
    predictions = ranking.run(similar_diffs, top_n=top_n)

    if not predictions:
        print("[Orchestrator] No tests predicted to fail above threshold.")
        return _empty_report(diff_text)

    # ── Step 3: Explanation ────────────────────────────────────────────────────
    # Ask Claude to explain each prediction
    print("\n[Orchestrator] Step 3/3: Generating explanations via Claude...")
    predictions_with_explanations = explanation.run(diff_text, predictions)

    # ── Build the final report ─────────────────────────────────────────────────
    # Count all unique tests seen across the retrieved diffs
    all_tests_seen = set()
    for diff in similar_diffs:
        all_tests_seen.update(diff["tests_run"])

    report = PredictionReport(
        diff_text=diff_text,
        ranked_tests=predictions_with_explanations,
        total_tests_in_db=len(all_tests_seen),
        similar_commits_used=len(similar_diffs),
        generated_at=datetime.now(timezone.utc),
    )

    _print_report(report, top_n)
    return report


def _print_report(report: PredictionReport, top_n: int) -> None:
    """Prints a human-readable summary of the prediction report."""
    print("\n" + "="*60)
    print(f"PREDICTION REPORT")
    print(f"Based on {report.similar_commits_used} similar historical commits")
    print(f"Showing top {min(top_n, len(report.ranked_tests))} of "
          f"{report.total_tests_in_db} known tests")
    print("="*60)

    for i, test in enumerate(report.ranked_tests[:top_n]):
        bar = "█" * int(test.failure_score * 10)  # visual score bar
        print(f"\n{i+1}. [{bar:<10}] {test.failure_score:.0%} — {test.test_name}")
        if test.explanation:
            print(f"   → {test.explanation}")

    print("\n" + "="*60)
    print(f"Run these {len(report.ranked_tests)} tests first instead of the full suite.")
    print("="*60 + "\n")


def _empty_report(diff_text: str) -> PredictionReport:
    """Returns an empty report when no predictions can be made."""
    return PredictionReport(
        diff_text=diff_text,
        ranked_tests=[],
        total_tests_in_db=0,
        similar_commits_used=0,
        generated_at=datetime.now(timezone.utc),
    )
