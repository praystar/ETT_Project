# LLM + Vector Database — RAG Pipeline

A simple, well-structured **Retrieval-Augmented Generation (RAG)** project that combines:

| Component | Technology |
|---|---|
| 🤖 Large Language Model | OpenAI GPT-4o-mini (swappable) |
| 🔠 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| 🗄️ Vector Database | ChromaDB (embedded, persistent) |
| ⚙️ Config | `pydantic-settings` + `.env` |

---

## Project Structure

```
llm-vector-project/
├── main.py           # RAG pipeline entry point
├── vector_store.py   # ChromaDB wrapper (upsert, query, delete)
├── embeddings.py     # SentenceTransformer embedding model
├── llm_client.py     # OpenAI chat completions wrapper
├── config.py         # Centralised settings (env vars)
├── requirements.txt  # All dependencies
├── .env.example      # Environment variable template
└── tests/
    └── test_vector_store.py
```

---

## Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd llm-vector-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the pipeline
```bash
# Demo mode (predefined queries)
python main.py

# Interactive mode
python main.py --interactive
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## How It Works

```
User Query
    │
    ▼
EmbeddingModel.embed(query)          ← sentence-transformers (local)
    │
    ▼
VectorStore.query(embedding, top_k)  ← ChromaDB cosine similarity search
    │
    ▼
Build prompt with retrieved context
    │
    ▼
LLMClient.complete(prompt)           ← OpenAI GPT
    │
    ▼
Answer
```

---

## Swapping Components

**Different LLM** — change `LLM_MODEL` in `.env`:
```
LLM_MODEL=gpt-4o          # More capable, higher cost
LLM_MODEL=gpt-3.5-turbo   # Faster, cheaper
```

**Local LLM via Ollama** — set the base URL:
```
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3
OPENAI_API_KEY=ollama       # dummy value
```

**Different embedding model**:
```
EMBEDDING_MODEL=all-mpnet-base-v2   # Stronger, 768-dim
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2  # Faster, smaller
```

---

## Virtualization / Cloud (Tertiary Topic)

For production deployment, consider:
- **Docker** — containerise the app with `python:3.12-slim`
- **ChromaDB server mode** — run ChromaDB as a standalone service
- **AWS / GCP / Azure** — host the container on ECS, Cloud Run, or AKS
- **Managed Vector DBs** — swap ChromaDB for Pinecone, Weaviate, or Qdrant

---

## Dependencies

See `requirements.txt` for exact versions. Key packages:

| Package | Purpose |
|---|---|
| `openai` | LLM API client |
| `sentence-transformers` | Local embedding models |
| `chromadb` | Embedded vector database |
| `pydantic-settings` | Environment-based config |
| `torch` | PyTorch backend for embeddings |
