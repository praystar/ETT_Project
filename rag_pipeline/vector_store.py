"""
vector_store.py — ChromaDB wrapper for storing and querying embeddings.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """
    A thin wrapper around ChromaDB that handles document upserts
    and similarity search.
    """

    def __init__(self, collection_name: str = "rag_docs", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        print(f"🗄️  VectorStore ready — collection: '{collection_name}' at '{persist_directory}'")

    def upsert(
        self,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a document with its embedding."""
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def upsert_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Batch insert or update multiple documents at once."""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{} for _ in ids],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar documents to the query embedding.

        Returns a list of dicts with keys: id, text, metadata, distance.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def delete(self, doc_id: str) -> None:
        """Remove a document from the collection by ID."""
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
