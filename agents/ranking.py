"""
agents/ranking.py

The Ranking Agent.

Responsibility: Given a list of similar historical diffs, compute a
failure score for each test and return a ranked list.

The core idea:
  If a test failed in commits that look similar to the new diff,
  it's likely to fail in the new diff too.

Scoring formula (per test):
  failure_score = sum(similarity × 1.0 for each similar commit where this test failed)
                / sum(similarity for each similar commit where this test was run)

This is a weighted failure rate:
  - A test that failed in commits with 0.9 similarity scores higher
    than one that failed in commits with 0.3 similarity
  - Tests that were never run in similar commits get score 0.0
"""

from collections import defaultdict
from models.schemas import TestPrediction
from config import TOP_N_TESTS_TO_REPORT, MIN_FAILURE_SCORE_THRESHOLD


def run(similar_diffs: list[dict], top_n: int = None) -> list[TestPrediction]:
    """
    Main entry point for the Ranking Agent.

    Aggregates test failure patterns from similar historical diffs
    and produces a ranked list of tests most likely to fail.

    Args:
        similar_diffs: Output from retrieval.run() — list of similar diffs
                       with their test results and similarity scores
        top_n:         How many tests to return (default: from config)

    Returns:
        List of TestPrediction objects sorted by failure_score descending.
        Only includes tests above MIN_FAILURE_SCORE_THRESHOLD.

    Example:
        similar = retrieval.run(my_diff)
        predictions = ranking.run(similar)
        for p in predictions[:5]:
            print(f"{p.test_name}: {p.failure_score:.2f}")
        # → test_auth_login: 0.87
        #   test_token_expiry: 0.72
        #   test_session_create: 0.61
        #   ...
    """
    top_n = top_n or TOP_N_TESTS_TO_REPORT

    if not similar_diffs:
        return []

    print(f"[Ranking] Aggregating failure patterns from {len(similar_diffs)} similar diffs...")

    # Accumulators for computing weighted failure rate per test
    # weighted_failures[test] = sum of similarity scores where this test FAILED
    # weighted_runs[test]     = sum of similarity scores where this test RAN
    weighted_failures = defaultdict(float)
    weighted_runs = defaultdict(float)

    # supporting_commits[test] = list of commit SHAs that contribute to its score
    supporting_commits = defaultdict(list)

    for diff_record in similar_diffs:
        similarity = diff_record["similarity"]
        tests_run = set(diff_record["tests_run"])
        tests_failed = set(diff_record["tests_failed"])
        sha = diff_record["commit_sha"]

        for test in tests_run:
            # This test ran in a commit similar to ours — add to the denominator
            weighted_runs[test] += similarity

            if test in tests_failed:
                # This test also FAILED in that similar commit — add to numerator
                weighted_failures[test] += similarity
                supporting_commits[test].append(sha)

    # Compute failure scores and build TestPrediction objects
    predictions = []

    for test in weighted_runs:
        if weighted_runs[test] == 0:
            continue

        # Weighted failure rate: how often did this test fail, weighted by similarity
        score = weighted_failures[test] / weighted_runs[test]

        if score < MIN_FAILURE_SCORE_THRESHOLD:
            continue  # too low to bother reporting

        predictions.append(TestPrediction(
            test_name=test,
            failure_score=round(score, 4),
            supporting_commits=supporting_commits[test],
            explanation="",  # filled in later by the Explanation Agent
        ))

    # Sort by failure score, highest first
    predictions.sort(key=lambda p: p.failure_score, reverse=True)

    # Return only the top N
    result = predictions[:top_n]

    print(f"[Ranking] Produced {len(result)} test predictions (from {len(predictions)} candidates)")
    if result:
        print(f"[Ranking] Top test: {result[0].test_name} (score: {result[0].failure_score})")

    return result
