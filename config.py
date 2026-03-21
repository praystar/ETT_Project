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
    LLM_PROVIDER: str = Field(default="grok", description="LLM provider: 'grok' or 'gemini'")
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    LLM_MODEL: str = Field(default="grok-3-70b", description="Chat model to use")
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name (local) or 'openai' to use OpenAI embeddings",
    )

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_db", description="Directory for ChromaDB persistence")
    CHROMA_COLLECTION: str = Field(default="rag_docs", description="ChromaDB collection name")

    # ── Document Loading ───────────────────────────────────────────────────────
    DOCUMENTS_DIR: str = Field(default="./documents", description="Directory where user places PDFs/DOCX/TXT files")
    CHUNK_SIZE: int = Field(default=500, description="Approximate words per document chunk")
    CHUNK_OVERLAP: int = Field(default=50, description="Words to overlap between chunks")
    WATCH_ENABLED: bool = Field(default=True, description="Enable folder watching for new documents")


# Singleton — import `settings` everywhere
settings = Settings()
