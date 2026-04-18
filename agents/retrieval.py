"""
agents/retrieval.py

The Retrieval Agent.

Responsibility: Given a new diff, find the most similar historical diffs
from the vector database.

This is essentially a semantic search:
  - Input:  a new diff (as text)
  - Output: the top-K most similar historical diffs

The "similarity" here is cosine similarity between embeddings.
High similarity means the two diffs changed similar parts of the code
in similar ways — which suggests they might break similar tests.
"""

from tools.diff_tools import clean_diff
from tools.vector_tools import query_similar_diffs, get_collection_size
from agents.embedding import embed_query
from config import TOP_K_SIMILAR, TARGET_REPO


def run(diff_text: str, top_k: int = None, repo: str = None) -> list[dict]:
    """
    Main entry point for the Retrieval Agent.

    Takes a raw git diff and returns the most similar historical diffs
    from the vector database.

    Args:
        diff_text: Raw git diff string (e.g. from `git diff main...HEAD`)
        top_k:     How many similar diffs to return (default: from config)

    Returns:
        List of similar diff records, sorted by similarity descending.
        Each record looks like:
          {
            "commit_sha":   "a3f9c21...",
            "diff_text":    "diff --git a/...",
            "tests_run":    ["test_auth", "test_login", ...],
            "tests_failed": ["test_login"],
            "timestamp":    "2024-01-15T10:30:00",
            "similarity":   0.87   ← higher = more similar
          }

    Example:
        diff = open("my_pr.diff").read()
        similar = retrieval.run(diff, top_k=15)
        # → 15 historical diffs most similar to my_pr.diff
    """
    top_k = top_k or TOP_K_SIMILAR
    repo = repo or TARGET_REPO

    db_size = get_collection_size(repo)
    if db_size == 0:
        print("[Retrieval] Vector DB is empty. Run embedding.run() first.")
        return []

    print(f"[Retrieval] Searching {db_size} historical diffs for similar changes...")

    # Step 1: Clean the new diff (same preprocessing we did during storage)
    # Important: use the same cleaning logic so embeddings are comparable
    cleaned_diff = clean_diff(diff_text)

    if not cleaned_diff.strip():
        print("[Retrieval] Warning: diff is empty after cleaning. No results.")
        return []

    # Step 2: Embed the cleaned diff as a QUERY (not a document)
    # The query embedding is slightly optimized for search vs storage
    query_vector = embed_query(cleaned_diff)

    # Step 3: Query ChromaDB for the top_k most similar embeddings
    similar_diffs = query_similar_diffs(query_vector, top_k=top_k, repo=repo)

    print(f"[Retrieval] Found {len(similar_diffs)} similar diffs")

    if similar_diffs:
        top_score = similar_diffs[0]["similarity"]
        print(f"[Retrieval] Top similarity score: {top_score:.3f}")

    return similar_diffs
