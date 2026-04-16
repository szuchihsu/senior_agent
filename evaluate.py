"""
evaluate.py

Backtesting agent for the Regression Predictor.

How evaluation works:
  We use a "leave-one-out" strategy:
    - Take a commit from our collected history
    - Temporarily remove it from the DB (or just query excluding it)
    - Ask the predictor: "which tests will fail on this commit?"
    - Compare the predicted failing tests to the actual failing tests
    - Record precision, recall, and time-savings

Why this matters for a senior project:
  A prediction system is only useful if we can MEASURE how good it is.
  This script produces the numbers you put in your paper/demo:
    "Our system achieves 72% recall while reducing test runs by 63%"

Metrics explained:
  Precision = of the tests we flagged, what fraction actually failed?
              High precision = few false alarms
  Recall    = of the tests that actually failed, what fraction did we catch?
              High recall = we don't miss real failures (this matters more)
  Time savings = (total tests - tests we'd run) / total tests
                 If we predict 5 tests and 45 are in the suite → 89% savings
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agents.data_collection import load_records
from agents import retrieval, ranking
from tools.vector_tools import get_collection_size
from config import TARGET_REPO, TOP_K_SIMILAR, TOP_N_TESTS_TO_REPORT


@dataclass
class CommitEvalResult:
    """Stores the evaluation result for a single commit."""
    commit_sha: str
    actual_failed: list[str]      # tests that actually failed
    predicted_failed: list[str]   # tests our model flagged
    all_tests: list[str]          # all tests that ran
    precision: float              # of predicted, fraction correct
    recall: float                 # of actual failures, fraction caught
    time_savings: float           # fraction of test suite we skipped


@dataclass
class EvalSummary:
    """Aggregated metrics across all evaluated commits."""
    total_commits: int = 0
    commits_with_failures: int = 0  # commits where at least one test failed
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_time_savings: float = 0.0
    results: list[CommitEvalResult] = field(default_factory=list)


def evaluate(
    repo: str = None,
    sample_size: int = 20,
    top_k: int = None,
    top_n: int = None,
    seed: int = 42,
) -> EvalSummary:
    """
    Run backtesting evaluation over a sample of historical commits.

    For each sampled commit:
      1. Get its diff (already in raw_data JSON)
      2. Query the predictor (retrieval + ranking, skip explanation for speed)
      3. Compare predictions to actual CI results
      4. Record metrics

    Args:
        repo:        "owner/repo" to evaluate (default: from config)
        sample_size: How many commits to evaluate (more = more accurate metrics)
        top_k:       How many similar diffs to retrieve per prediction
        top_n:       How many tests to include in each prediction
        seed:        Random seed for reproducible sampling

    Returns:
        EvalSummary with per-commit results and aggregate metrics.

    Usage:
        from evaluate import evaluate
        summary = evaluate(sample_size=30)
        print(f"Recall: {summary.avg_recall:.1%}")
        print(f"Time savings: {summary.avg_time_savings:.1%}")
    """
    repo = repo or TARGET_REPO
    top_k = top_k or TOP_K_SIMILAR
    top_n = top_n or TOP_N_TESTS_TO_REPORT

    print(f"\n{'='*60}")
    print(f"EVALUATION — {repo}")
    print(f"Sample size: {sample_size} commits | top_k={top_k} | top_n={top_n}")
    print(f"{'='*60}\n")

    # Load all collected records from disk
    all_records = load_records(repo)
    if not all_records:
        print("No records found. Run data_collection.run() first.")
        return EvalSummary()

    # Only evaluate on commits that had at least one test failure
    # (commits where nothing failed aren't interesting to evaluate)
    records_with_failures = [r for r in all_records if r["tests_failed"]]
    print(f"Records with failures: {len(records_with_failures)} / {len(all_records)}")

    # Sample randomly for faster evaluation
    random.seed(seed)
    sample = random.sample(
        records_with_failures,
        min(sample_size, len(records_with_failures))
    )

    summary = EvalSummary()
    summary.total_commits = len(sample)
    summary.commits_with_failures = len(sample)

    for i, record in enumerate(sample):
        sha = record["commit_sha"]
        print(f"[Eval] [{i+1}/{len(sample)}] Evaluating {sha[:8]}...")

        actual_failed = set(record["tests_failed"])
        all_tests = set(record["tests_run"])

        # Run the predictor pipeline (skip explanation — it calls Claude and we want speed)
        similar_diffs = retrieval.run(record["diff_text"], top_k=top_k)

        # Exclude the commit being evaluated from the similar diffs
        # (otherwise we'd be "cheating" by finding the exact same commit)
        similar_diffs = [d for d in similar_diffs if d["commit_sha"] != sha]

        predictions = ranking.run(similar_diffs, top_n=top_n)
        predicted_failed = set(p.test_name for p in predictions)

        # Calculate metrics
        precision, recall = _compute_precision_recall(predicted_failed, actual_failed)
        time_savings = _compute_time_savings(predicted_failed, all_tests)

        result = CommitEvalResult(
            commit_sha=sha,
            actual_failed=list(actual_failed),
            predicted_failed=list(predicted_failed),
            all_tests=list(all_tests),
            precision=precision,
            recall=recall,
            time_savings=time_savings,
        )
        summary.results.append(result)

        print(f"[Eval]   Actual failures: {len(actual_failed)} | "
              f"Predicted: {len(predicted_failed)} | "
              f"Recall: {recall:.0%} | Precision: {precision:.0%} | "
              f"Saved: {time_savings:.0%}")

    # Aggregate metrics
    if summary.results:
        summary.avg_precision = sum(r.precision for r in summary.results) / len(summary.results)
        summary.avg_recall = sum(r.recall for r in summary.results) / len(summary.results)
        summary.avg_time_savings = sum(r.time_savings for r in summary.results) / len(summary.results)

    _print_summary(summary)
    _save_results(summary, repo)
    return summary


def _compute_precision_recall(
    predicted: set[str],
    actual: set[str]
) -> tuple[float, float]:
    """
    Computes precision and recall for a single commit prediction.

    Precision = True Positives / (True Positives + False Positives)
              = tests we flagged that actually failed / all tests we flagged

    Recall    = True Positives / (True Positives + False Negatives)
              = tests we flagged that actually failed / all tests that actually failed

    Edge cases:
      - If we predicted nothing: precision=0, recall=0
      - If nothing actually failed: precision=1 (nothing to get wrong), recall=1
    """
    if not predicted:
        return 0.0, 0.0

    if not actual:
        return 1.0, 1.0  # nothing failed, nothing to miss

    true_positives = predicted & actual  # tests we correctly flagged

    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(actual) if actual else 0.0

    return round(precision, 4), round(recall, 4)


def _compute_time_savings(predicted: set[str], all_tests: set[str]) -> float:
    """
    Estimates what fraction of the test suite we avoided running.

    If the full suite has 50 tests and we only flagged 8, we saved
    running 42 tests = 84% time savings.

    In reality, you'd still run the flagged tests — the savings come
    from skipping the un-flagged ones.
    """
    if not all_tests:
        return 0.0

    # Tests we'd run = flagged tests (capped at total test count)
    tests_to_run = min(len(predicted), len(all_tests))
    return round(1.0 - (tests_to_run / len(all_tests)), 4)


def _print_summary(summary: EvalSummary) -> None:
    """Prints a human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Commits evaluated:    {summary.total_commits}")
    print(f"Commits with failures:{summary.commits_with_failures}")
    print(f"")
    print(f"Average Precision:    {summary.avg_precision:.1%}")
    print(f"  (of flagged tests, how many actually failed)")
    print(f"")
    print(f"Average Recall:       {summary.avg_recall:.1%}")
    print(f"  (of real failures, how many did we catch)")
    print(f"")
    print(f"Average Time Savings: {summary.avg_time_savings:.1%}")
    print(f"  (fraction of test suite we skipped)")
    print(f"{'='*60}\n")


def _save_results(summary: EvalSummary, repo: str) -> None:
    """Saves detailed results to a JSON file for later analysis."""
    output_path = Path("storage") / "eval_results.json"
    output_path.parent.mkdir(exist_ok=True)

    data = {
        "repo": repo,
        "total_commits": summary.total_commits,
        "avg_precision": summary.avg_precision,
        "avg_recall": summary.avg_recall,
        "avg_time_savings": summary.avg_time_savings,
        "per_commit_results": [
            {
                "commit_sha": r.commit_sha,
                "precision": r.precision,
                "recall": r.recall,
                "time_savings": r.time_savings,
                "actual_failed_count": len(r.actual_failed),
                "predicted_count": len(r.predicted_failed),
            }
            for r in summary.results
        ]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate regression predictor accuracy")
    parser.add_argument("--sample", type=int, default=20, help="Number of commits to evaluate")
    parser.add_argument("--repo", default=TARGET_REPO)
    args = parser.parse_args()

    evaluate(repo=args.repo, sample_size=args.sample)
