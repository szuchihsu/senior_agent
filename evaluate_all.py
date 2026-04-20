"""
evaluate_all.py

Runs evaluation across multiple repos and prints a comparison table.

Usage:
    python evaluate_all.py
    python evaluate_all.py --sample 30
"""

import argparse
from evaluate import evaluate


REPOS = [
    "pallets/flask",
    "psf/requests",
    "pytest-dev/pytest",
]


def main(sample_size: int = 20):
    results = []

    for repo in REPOS:
        print(f"\n{'#'*60}")
        print(f"# Evaluating {repo}")
        print(f"{'#'*60}")
        try:
            summary = evaluate(repo=repo, sample_size=sample_size)
            results.append({
                "repo": repo,
                "commits": summary.total_commits,
                "precision": summary.avg_precision,
                "recall": summary.avg_recall,
                "time_savings": summary.avg_time_savings,
            })
        except Exception as e:
            print(f"[EvalAll] Skipping {repo}: {e}")
            results.append({
                "repo": repo,
                "commits": 0,
                "precision": None,
                "recall": None,
                "time_savings": None,
            })

    _print_table(results)


def _print_table(results: list[dict]) -> None:
    print("\n")
    print("=" * 65)
    print("MULTI-REPO EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Repo':<25} {'Commits':>8} {'Precision':>10} {'Recall':>8} {'Time Saved':>11}")
    print("-" * 65)
    for r in results:
        if r["precision"] is None:
            print(f"{r['repo']:<25} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>11}")
        else:
            print(
                f"{r['repo']:<25} "
                f"{r['commits']:>8} "
                f"{r['precision']:>9.1%} "
                f"{r['recall']:>8.1%} "
                f"{r['time_savings']:>10.1%}"
            )
    print("=" * 65)
    print()
    print("Precision   = of flagged tests, fraction that actually failed")
    print("Recall      = of real failures, fraction we caught")
    print("Time Saved  = fraction of test suite we skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate across all repos")
    parser.add_argument("--sample", type=int, default=20,
                        help="Commits to evaluate per repo (default: 20)")
    args = parser.parse_args()
    main(sample_size=args.sample)
