"""
tests/test_vector_store.py — Unit tests for the VectorStore class.
Run with: pytest tests/ -v
"""

import pytest
from vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh in-memory-like VectorStore backed by a temp directory."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chroma_test"),
    )


def test_upsert_and_count(store):
    store.upsert("id1", [0.1, 0.2, 0.3], "Hello world", {"source": "test"})
    store.upsert("id2", [0.4, 0.5, 0.6], "Another doc", {"source": "test"})
    assert store.count() == 2


def test_query_returns_results(store):
    store.upsert("id1", [1.0, 0.0, 0.0], "Document about cats", {})
    store.upsert("id2", [0.0, 1.0, 0.0], "Document about dogs", {})
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "id1"


def test_delete(store):
    store.upsert("id1", [0.1, 0.2, 0.3], "Delete me", {})
    assert store.count() == 1
    store.delete("id1")
    assert store.count() == 0


def test_reset(store):
    for i in range(5):
        store.upsert(f"id{i}", [float(i), 0.0, 0.0], f"Doc {i}", {})
    assert store.count() == 5
    store.reset()
    assert store.count() == 0
