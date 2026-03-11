"""
config.py — Centralised settings loaded from environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="your-api-key-here", description="OpenAI API key")
    OPENAI_BASE_URL: str | None = Field(
        default=None,
        description="Optional custom base URL (e.g. for local Ollama or Azure OpenAI)",
    )
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="Chat model to use")
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name (local) or 'openai' to use OpenAI embeddings",
    )

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", description="Directory for ChromaDB persistence")
    CHROMA_COLLECTION: str = Field(default="rag_docs", description="ChromaDB collection name")


# Singleton — import `settings` everywhere
settings = Settings()
