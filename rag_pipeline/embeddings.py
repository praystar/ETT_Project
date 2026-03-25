"""
embeddings.py — Generates text embeddings using a local sentence-transformer model.

By default uses 'all-MiniLM-L6-v2' (fast, 384-dim).
Swap for 'text-embedding-3-small' via OpenAI if you prefer a hosted model.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from config import settings


class EmbeddingModel:
    """
    Wraps a SentenceTransformer to produce fixed-size vector embeddings
    for arbitrary text inputs.
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        print(f"🔠 Loading embedding model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> List[float]:
        """Return a single embedding vector for the given text."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts (more efficient than one-by-one)."""
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]
