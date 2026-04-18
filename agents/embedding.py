"""
agents/embedding.py

The Embedding Agent.

Responsibility: Convert diff text into vector embeddings and store them
in ChromaDB. This bridges raw text data and the vector database.

What is an embedding?
  An embedding is a fixed-size list of floats that captures the *meaning*
  of a piece of text. Two semantically similar texts (e.g. two diffs that
  both modify authentication logic) will have embeddings that are close
  together in vector space.

  This is what makes retrieval work — instead of keyword matching,
  we find diffs that are semantically similar to the new diff.

We use sentence-transformers with the "all-MiniLM-L6-v2" model.
It runs fully locally (no API key, no rate limits, no cost) and produces
384-dimensional embeddings. The model is downloaded once on first use
(~90MB) and cached locally.

For even better code-specific embeddings in the future, you could swap
this out for "jinaai/jina-embeddings-v2-base-code" or Voyage AI's
voyage-code-3 (requires API key).
"""

from sentence_transformers import SentenceTransformer
from datetime import datetime

from models.schemas import DiffRecord
from tools.vector_tools import store_diff_record, get_collection_size
from agents.data_collection import load_records
from agents.noise_detector import load_blocklist, apply_blocklist
from config import TARGET_REPO


# Load the model once at module import time so it isn't reloaded on every call.
# First run downloads ~90MB to ~/.cache/huggingface/; subsequent runs load from cache.
print("[Embedding] Loading local embedding model (all-MiniLM-L6-v2)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    """
    Converts a text string into a vector embedding using a local model.

    Runs entirely on your machine — no API calls, no rate limits.
    Produces a 384-dimensional vector for the input text.

    Args:
        text: The text to embed (usually a cleaned git diff)

    Returns:
        A list of 384 floats representing the semantic meaning of the text.

    Example:
        embed_text("- def login(user, password):\\n+ def login(user, password, mfa_code):")
        → [0.023, -0.11, 0.87, ...] (384 floats)
    """
    # encode() returns a numpy array; .tolist() converts to plain Python list
    # for compatibility with ChromaDB and Pydantic
    return _model.encode(text, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """
    Embeds a query diff for retrieval.

    With sentence-transformers there's no distinction between document and
    query embeddings (unlike Voyage AI), so this is identical to embed_text().
    We keep the separate function so the rest of the codebase doesn't need
    to change if we swap embedding backends later.

    Args:
        text: The query diff text

    Returns:
        A list of 384 floats (query embedding)
    """
    return _model.encode(text, normalize_embeddings=True).tolist()


def run(repo: str = None, force_reembed: bool = False) -> int:
    """
    Main entry point for the Embedding Agent.

    Loads raw records from disk (collected by Data Collection Agent),
    embeds each diff, and stores the results in ChromaDB.

    Args:
        repo:          "owner/repo" (default: from config.py)
        force_reembed: If True, re-embed even if already in the DB.
                       Useful if you change embedding models.

    Returns:
        Number of new records embedded and stored.

    Usage:
        from agents import embedding
        count = embedding.run()
    """
    repo = repo or TARGET_REPO

    print(f"[Embedding] Loading raw records...")
    records = load_records(repo)

    if not records:
        print("[Embedding] No records to embed. Run data_collection.run() first.")
        return 0

    # Apply noise blocklist if one exists for this repo
    blocklist = load_blocklist(repo)
    if blocklist:
        records = apply_blocklist(records, blocklist)

    current_db_size = get_collection_size(repo)
    print(f"[Embedding] Vector DB currently has {current_db_size} records")

    embedded_count = 0

    for i, record in enumerate(records):
        sha = record["commit_sha"]
        print(f"[Embedding] [{i+1}/{len(records)}] Embedding {sha[:8]}...")

        try:
            # Generate the embedding for this diff
            # This is an API call — takes ~100ms per diff
            embedding_vector = embed_text(record["diff_text"])

            # Build a DiffRecord (our schema) from the raw dict
            diff_record = DiffRecord(
                commit_sha=sha,
                diff_text=record["diff_text"],
                diff_embedding=embedding_vector,
                tests_run=record["tests_run"],
                tests_failed=record["tests_failed"],
                timestamp=datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")),
                pr_number=record.get("pr_number"),
            )

            # Store in ChromaDB — use repo-specific collection
            # ChromaDB will skip duplicates if the same ID is added twice
            store_diff_record(diff_record, repo)
            embedded_count += 1

        except Exception as e:
            print(f"[Embedding]   Error embedding {sha[:8]}: {e}")
            continue

    new_db_size = get_collection_size(repo)
    print(f"[Embedding] Done. Embedded {embedded_count} records. DB now has {new_db_size} entries.")
    return embedded_count
