"""
main.py

Command-line interface for the Regression Predictor Agent.

Two modes:

  1. BUILD — collect historical CI data and build the vector index
     Run this once before making any predictions.

     python main.py build
     python main.py build --repo pallets/flask --max-commits 200

  2. PREDICT — given a diff, predict which tests are likely to fail
     Run this before merging a PR.

     python main.py predict --diff path/to/my.diff
     git diff main...HEAD | python main.py predict --stdin
     python main.py predict --sha a3f9c21   (analyze a specific commit)
"""

import argparse
import subprocess
import sys

from agents import data_collection, embedding
from agents.orchestrator import predict
from config import TARGET_REPO, MAX_COMMITS_TO_COLLECT, TOP_N_TESTS_TO_REPORT


def cmd_build(args):
    """
    Build phase: collect CI data → embed diffs → populate vector DB.

    This typically takes a few minutes depending on how many commits
    you're collecting and how fast the GitHub API responds.
    """
    print(f"Building vector index for {args.repo}...")
    print(f"Step 1/2: Collecting CI history ({args.max_commits} commits)...")
    data_collection.run(repo=args.repo, max_commits=args.max_commits)

    print("\nStep 2/2: Embedding diffs and storing in vector DB...")
    count = embedding.run(repo=args.repo)

    print(f"\nBuild complete. {count} diffs embedded and indexed.")
    print("You can now run: python main.py predict --diff <path>")


def cmd_predict(args):
    """
    Predict phase: given a diff, output a ranked list of likely-failing tests.

    Three ways to provide the diff:
      --diff <file>  : read from a file
      --stdin        : read from stdin (pipe from git)
      --sha <hash>   : fetch from GitHub directly
    """
    # Get the diff text from the user's chosen source
    if args.stdin:
        print("Reading diff from stdin...")
        diff_text = sys.stdin.read()

    elif args.diff:
        print(f"Reading diff from {args.diff}...")
        with open(args.diff) as f:
            diff_text = f.read()

    elif args.sha:
        print(f"Fetching diff for commit {args.sha}...")
        from tools.github_tools import get_commit_diff
        diff_text = get_commit_diff(TARGET_REPO, args.sha)

    else:
        # Default: run `git diff main...HEAD` in the current directory
        print("Running `git diff main...HEAD`...")
        try:
            diff_text = subprocess.check_output(
                ["git", "diff", "main...HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8")
        except subprocess.CalledProcessError:
            print("Error: Could not run git diff. Are you in a git repo?")
            print("Use --diff <file> or --sha <hash> instead.")
            sys.exit(1)

    if not diff_text.strip():
        print("Error: Empty diff. Nothing to analyze.")
        sys.exit(1)

    # Run the full prediction pipeline
    report = predict(diff_text, top_n=args.top_n)

    # Optionally save the report to a JSON file
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        print(f"Report saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Regression Predictor Agent — predict which tests will break before merging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build command ──────────────────────────────────────────────────────────
    build_parser = subparsers.add_parser("build", help="Collect CI data and build vector index")
    build_parser.add_argument("--repo", default=TARGET_REPO,
                              help=f"GitHub repo to collect from (default: {TARGET_REPO})")
    build_parser.add_argument("--max-commits", type=int, default=MAX_COMMITS_TO_COLLECT,
                              help=f"Max commits to collect (default: {MAX_COMMITS_TO_COLLECT})")
    build_parser.set_defaults(func=cmd_build)

    # ── predict command ────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser("predict", help="Predict failing tests for a diff")
    predict_parser.add_argument("--diff", type=str, help="Path to a .diff file")
    predict_parser.add_argument("--sha", type=str, help="GitHub commit SHA to analyze")
    predict_parser.add_argument("--stdin", action="store_true",
                                help="Read diff from stdin (pipe from git)")
    predict_parser.add_argument("--top-n", type=int, default=TOP_N_TESTS_TO_REPORT,
                                help=f"Number of tests to show (default: {TOP_N_TESTS_TO_REPORT})")
    predict_parser.add_argument("--output", type=str,
                                help="Save report as JSON to this file path")
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
