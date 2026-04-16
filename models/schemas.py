"""
models/schemas.py

Defines the core data structures used across all agents.
Using Pydantic so we get automatic type validation and easy
serialization to/from JSON (for saving to disk and loading back).
"""

from datetime import datetime
from pydantic import BaseModel


class DiffRecord(BaseModel):
    """
    Represents one historical commit that we've collected and embedded.

    This is the main unit of "training data" for our predictor.
    Each DiffRecord stores:
      - the raw code diff (what changed)
      - its vector embedding (a list of floats representing the diff's meaning)
      - which tests were run and which failed

    We store this so that later, when we see a new diff, we can:
      1. Embed the new diff
      2. Find DiffRecords whose embeddings are close (similar changes)
      3. Look at which tests failed in those records
    """

    commit_sha: str              # unique ID for the commit, e.g. "a3f9c21..."
    diff_text: str               # raw git diff output
    diff_embedding: list[float]  # vector representation of the diff (from embedding model)
    tests_run: list[str]         # all tests that ran in this CI run
    tests_failed: list[str]      # subset of tests_run that failed
    timestamp: datetime          # when the commit happened (useful for weighting recent data higher)
    pr_number: int | None = None # optional: which PR this commit came from


class TestPrediction(BaseModel):
    """
    Represents a single test's predicted likelihood of failing.

    After we retrieve similar historical diffs and aggregate their
    failure patterns, we produce one TestPrediction per test.

    The failure_score is a float between 0 and 1:
      - 0.0 = very unlikely to break
      - 1.0 = almost certainly will break

    We also keep track of *why* we think it'll break (supporting_commits)
    so the Explanation Agent can justify the prediction to the user.
    """

    test_name: str                       # e.g. "tests/auth/test_login.py::test_valid_user"
    failure_score: float                 # 0.0 to 1.0, higher = more likely to fail
    supporting_commits: list[str]        # commit SHAs that contribute to this score
    explanation: str = ""                # filled in by the Explanation Agent


class PredictionReport(BaseModel):
    """
    The final output of the entire pipeline.

    Contains the diff that was analyzed, the ranked list of tests
    (most likely to fail first), and summary statistics.

    This is what gets printed to the user or sent to a GitHub comment.
    """

    diff_text: str                        # the input diff that was analyzed
    ranked_tests: list[TestPrediction]    # sorted by failure_score descending
    total_tests_in_db: int                # how many unique tests exist in historical data
    similar_commits_used: int             # how many historical commits informed this prediction
    generated_at: datetime = None         # timestamp of when this report was generated

    def top_n(self, n: int) -> list[TestPrediction]:
        """Returns the top N tests most likely to fail."""
        return self.ranked_tests[:n]
