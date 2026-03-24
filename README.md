# LLM + Vector Database — RAG Pipeline

A simple, well-structured **Retrieval-Augmented Generation (RAG)** project that combines:

| Component | Technology |
|---|---|
| 🤖 Large Language Model | Groq or Google Gemini (swappable) |
| 🔠 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| 🗄️ Vector Database | ChromaDB (embedded, persistent) |
| ⚙️ Config | `pydantic-settings` + `.env` |

---

## Project Structure

```
ETT_Project/
├── main.py                 # RAG pipeline entry point
├── document_loader.py      # Extract & chunk PDFs/DOCX/TXT
├── watch_documents.py      # Optional: continuous folder monitoring
├── vector_store.py         # ChromaDB wrapper (upsert, query, delete)
├── embeddings.py           # SentenceTransformer embedding model
├── llm_client.py           # Groq/Gemini chat completions wrapper
├── config.py               # Centralised settings (env vars)
├── requirements.txt        # All dependencies
├── documents/              # 📁 Drop your PDFs/DOCX/TXT files here
│   └── (your documents)
├── chroma_db/              # ChromaDB persistence
├── .env.example            # Environment variable template
├── SETUP_GEMINI.md         # Gemini setup guide
└── tests/
    └── test_vector_store.py
```

---

## Quick Start

### 1. Clone & install
```bash
git clone <your-repo>
cd ETT_Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your API key (GROQ_API_KEY or GEMINI_API_KEY)
```

### 3. Run the pipeline
```bash
# Demo mode (predefined queries)
python main.py

# Interactive mode
python main.py --interactive

# Use sample documents if folder is empty
python main.py --samples
```

### 4. Add your own documents

Place PDF, DOCX, or TXT files in the `./documents/` directory:

```bash
# Example: Add documents to the folder
cp ~/Downloads/my_paper.pdf ./documents/
cp ~/Downloads/my_notes.docx ./documents/
echo "Some important text" > ./documents/notes.txt

# Run the pipeline — it will automatically load them
python main.py
```

The system will:
1. ✅ Scan the `./documents/` folder recursively
2. ✅ Extract text from PDFs, DOCX, and TXT files
3. ✅ Split large documents into chunks (~500 words each)
4. ✅ Generate embeddings and store in ChromaDB
5. ✅ Make them searchable via RAG queries

**Optional: Continuous watching** — monitor the folder for new files and auto-import:
```bash
# This watches the document folder in real-time
python watch_documents.py
```

### 5. Run tests
```bash
pytest tests/ -v
```

---

## Document Ingestion

### Supported Formats
- **PDF** (`.pdf`) — text extraction via `pypdf`
- **DOCX** (`.docx`) — text extraction via `python-docx`
- **TXT** (`.txt`) — plain text files

### Folder Structure
```
.
├── documents/              # Drop your files here
│   ├── research_paper.pdf
│   ├── notes.docx
│   └── summary.txt
├── main.py
└── watch_documents.py
```

### Configuration

Adjust document processing in `.env`:
```bash
DOCUMENTS_DIR=./documents          # Where to scan for files
CHUNK_SIZE=500                     # Words per chunk (≈270 tokens)
CHUNK_OVERLAP=50                   # Overlap between chunks for context
WATCH_ENABLED=True                 # Enable folder watching
```

### How Ingestion Works

```
Document → Extract Text → Chunk → Embed → Store in ChromaDB
  .pdf      pypdf        500w     MiniLM      ↓
  .docx     python-docx   +50     384-dim   Query &
  .txt      raw read      overlap   ↓       Retrieve
                                  [vec1,
                                   vec2, ...]
```

Each chunk gets a unique ID: `{filename}_{chunk_index}_{hash}` for tracking.

### Troubleshooting Documents

**Issue: "No documents found"**
- Verify files are in `./documents/` (check path)
- Check supported formats (`.pdf`, `.docx`, `.txt` only)
- Ensure files are readable (not corrupted)

**Issue: PDF text extraction is poor**
- Some PDFs are image-based (scanned). Current version uses text extraction only.
- For scanned PDFs, consider OCR (future enhancement with Tesseract)

**Issue: Large files take too long**
- Chunking happens automatically. Very large PDFs may take time to process.
- Adjust `CHUNK_SIZE` to balance context granularity vs speed.

---

## How It Works

```
User Document (PDF/DOCX/TXT)
    │
    ▼
DocumentLoader.load_files_from_directory()
    │
    ├─→ Extract text
    ├─→ Split into chunks (~500w)
    └─→ Generate embeddings
         │
         ▼
    VectorStore.upsert() → ChromaDB
         │
         ▼
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
LLMClient.complete(prompt)           ← Groq or Gemini
    │
    ▼
Answer
```

---

## Swapping Components

**Different LLM provider** — change `LLM_PROVIDER` in `.env`:
```
LLM_PROVIDER=groq          # Use Groq API
LLM_PROVIDER=gemini        # Use Google Gemini API
```

**Different LLM model** — change `LLM_MODEL` in `.env`:
```
# For Groq
LLM_MODEL=gpt-oss-120b     # Default Groq model
LLM_MODEL=llama3-70b-8192  # Alternative Groq model

# For Gemini
LLM_MODEL=gemini-2.0-flash-exp  # Default Gemini model
```

**Different embedding model**:
```
EMBEDDING_MODEL=all-MiniLM-L6-v2         # Default, 384-dim
EMBEDDING_MODEL=all-mpnet-base-v2        # Stronger, 768-dim
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
| `groq` | Groq LLM API client |
| `google-genai` | Google Gemini API client |
| `sentence-transformers` | Local embedding models |
| `chromadb` | Embedded vector database |
| `pydantic-settings` | Environment-based config |
| `torch` | PyTorch backend for embeddings |

Ingestion of documents works with both Groq and Gemini.
