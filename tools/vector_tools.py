"""
tools/vector_tools.py

Wrapper around ChromaDB for storing and querying diff embeddings.

ChromaDB is a vector database — it stores embeddings (lists of floats)
and lets you find the most similar ones given a query embedding.

Think of it like a nearest-neighbor search:
  - We store 500 historical diffs, each represented as a point in 1536-dimensional space
  - When a new diff comes in, we find the 20 closest points
  - "Close" means semantically similar code changes

ChromaDB handles all the math (cosine similarity) and persistence for us.
"""

import json
import chromadb
from chromadb.config import Settings
from models.schemas import DiffRecord
from config import VECTOR_DB_DIR


# ── Client setup ──────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Creates (or opens) a persistent ChromaDB client.

    "Persistent" means the data is saved to disk at VECTOR_DB_DIR.
    If you restart the program, your embeddings are still there.

    Returns:
        A ChromaDB client connected to local storage.
    """
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False)  # don't send usage data
    )


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Gets or creates the 'diffs' collection in ChromaDB.

    A "collection" is like a table in a SQL database — it groups
    related embeddings together. We use one collection for all diffs.

    The collection stores:
      - embeddings: the float vectors
      - documents: the raw diff text (so we can retrieve it later)
      - metadatas: structured data (commit SHA, failed tests, etc.)

    Returns:
        The ChromaDB collection object.
    """
    return client.get_or_create_collection(
        name="diffs",
        # cosine similarity is better than euclidean distance for text embeddings
        # because it measures the angle between vectors, not their magnitude
        metadata={"hnsw:space": "cosine"}
    )


# ── Storage ───────────────────────────────────────────────────────────────────

def store_diff_record(record: DiffRecord) -> None:
    """
    Stores a single DiffRecord in the vector database.

    ChromaDB needs four things per entry:
      - id:        unique string identifier
      - embedding: the vector (list of floats)
      - document:  the raw text (for retrieval/display)
      - metadata:  structured fields (must be str, int, float, or bool)

    We serialize lists (tests_run, tests_failed) to JSON strings
    because ChromaDB metadata doesn't support list values.

    Args:
        record: A DiffRecord with embedding already computed.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    collection.add(
        ids=[record.commit_sha],
        embeddings=[record.diff_embedding],
        documents=[record.diff_text],
        metadatas=[{
            "commit_sha": record.commit_sha,
            # JSON-encode lists because ChromaDB metadata only supports primitives
            "tests_run": json.dumps(record.tests_run),
            "tests_failed": json.dumps(record.tests_failed),
            "timestamp": record.timestamp.isoformat(),
        }]
    )


def query_similar_diffs(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    """
    Finds the most similar historical diffs to a query embedding.

    This is the core retrieval operation:
      Given the embedding of a new diff, find the top_k diffs from history
      whose embeddings are most similar (smallest cosine distance).

    Args:
        query_embedding: The embedding vector of the new diff we're predicting for.
        top_k:           How many similar diffs to return.

    Returns:
        List of result dicts, each containing:
          {
            "commit_sha":    str,
            "diff_text":     str,
            "tests_run":     list[str],
            "tests_failed":  list[str],
            "timestamp":     str,
            "similarity":    float  (1.0 = identical, 0.0 = completely different)
          }
        Sorted from most similar to least similar.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        return []  # no data yet

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),  # can't ask for more than we have
        include=["documents", "metadatas", "distances"]
    )

    similar_diffs = []

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite)
        # Convert to similarity score (1 = identical, 0 = completely different)
        similarity = 1.0 - (distance / 2.0)

        similar_diffs.append({
            "commit_sha": metadata["commit_sha"],
            "diff_text": results["documents"][0][i],
            "tests_run": json.loads(metadata["tests_run"]),
            "tests_failed": json.loads(metadata["tests_failed"]),
            "timestamp": metadata["timestamp"],
            "similarity": round(similarity, 4),
        })

    return similar_diffs


def get_collection_size() -> int:
    """Returns how many diffs are currently stored in the vector DB."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return collection.count()
