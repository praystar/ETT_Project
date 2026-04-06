# Directory Structure Overview

```
ETT_Project/
│
├── 🔧 BACKEND (Python)
│   └── backend/
│       ├── __init__.py
│       ├── app.py              # Flask REST API server
│       ├── config.py            # Settings & environment config
│       ├── vector_store.py      # ChromaDB wrapper
│       ├── embeddings.py        # SentenceTransformer wrapper
│       ├── llm_client.py        # Groq/Gemini client
│       ├── document_loader.py   # PDF/DOCX/TXT loader
│       └── main.py              # CLI interface (original)
│
├── 🌐 FRONTEND (Web UI)
│   └── frontend/
│       └── index.html           # Chat interface (vanilla JS)
│
├── 🧪 TESTS
│   └── tests/
│       ├── __init__.py
│       └── test_vector_store.py # Unit tests
│
├── 📚 DOCUMENTATION
│   └── docs/
│       ├── QUICKSTART.md        # Quick start guide
│       ├── API.md               # API documentation 
│       ├── DEVELOPMENT.md       # Development guide
│       └── STRUCTURE.md         # This file
│
├── 📁 DATA & CONFIG
│   ├── documents/               # Your documents (PDF/DOCX/TXT)
│   ├── chroma_db/               # Vector database storage
│   ├── .env                     # API keys (create from .env.example)
│   ├── .env.example             # Template for .env
│   └── .gitignore               # Git ignore patterns
│
├── 🚀 STARTUP SCRIPTS
│   ├── start.sh                 # Linux/macOS startup
│   └── start.bat                # Windows startup
│
├── 📦 DEPENDENCIES
│   └── requirements.txt          # Python packages
│
└── 📄 PROJECT INFO
    ├── README.md                # Project overview
    ├── LICENSE
    └── SETUP_GEMINI.md          # Gemini setup guide
```

## Folder Purposes

### `backend/` - Python Backend
Core RAG pipeline implementation:
- **app.py** - Flask REST API (run this to start server)
- **config.py** - Loads environment variables
- **embeddings.py** - Vector representations  
- **vector_store.py** - ChromaDB persistence
- **llm_client.py** - LLM API clients
- **document_loader.py** - File processing
- **main.py** - CLI version (original)

Start with: `cd backend && python app.py`

### `frontend/` - Web Interface
Single HTML file with embedded CSS/JavaScript:
- **index.html** - Interactive chat UI
- No build tools needed
- Direct browser file:// protocol

Usage: Open in browser → http://localhost:5000 (backend API)

### `tests/` - Unit Tests
Quality assurance:
- **test_vector_store.py** - ChromaDB & retrieval tests

Run with: `pytest tests/ -v`

### `docs/` - Documentation
User and developer guides:
- **QUICKSTART.md** - Setup & usage
- **API.md** - REST API reference
- **DEVELOPMENT.md** - Development guide
- **STRUCTURE.md** - This file

### `documents/` - User's Documents
Directory for adding your own:
- Place PDF, DOCX, TXT files here
- Click "Ingest Documents" to load
- Auto-discovered on startup

### `chroma_db/` - Vector Database
Persistent storage (auto-created):
- ChromaDB data
- Embeddings & document chunks
- Metadata

### Configuration Files
- **.env** - API keys (REQUIRED - create from .env.example)
- **.env.example** - Template
- **.gitignore** - Git exclusions
- **requirements.txt** - Python dependencies

### Startup Scripts
- **start.sh** - Automated startup (Linux/macOS)
- **start.bat** - Automated startup (Windows)

Handles:
1. Dependency checks
2. Backend initialization
3. Browser opening
4. Sample document loading

## Data Flow

```
User Question (Browser)
    ↓
frontend/index.html (vanilla JS)
    ↓ HTTP POST to /api/query
backend/app.py (Flask)
    ↓
1. Embed query (embeddings.py)
2. Search (vector_store.py)
3. Generate (llm_client.py)
    ↓
Answer + Sources
    ↓
JSON Response
    ↓
Display in Browser
```

## File Dependencies

```
frontend/index.html
    ← Calls → backend/app.py (HTTP)

backend/app.py
    → imports → config.py
    → imports → vector_store.py
    → imports → embeddings.py
    → imports → llm_client.py
    → imports → document_loader.py

backend/main.py (CLI)
    → imports → [same as app.py]

tests/test_vector_store.py
    → imports → backend/vector_store.py
```

## Installation & Setup

1. **Create virtual environment**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Create .env**: `cp .env.example .env` + add API key
4. **Run startup script**: `./start.sh` or `start.bat`

## Key Differences from Original

✅ **Organized structure** - Separated backend/frontend/docs/tests  
✅ **Cleaner imports** - All Python modules in `backend/`  
✅ **Flask API** - REST endpoints for web UI  
✅ **Web interface** - Interactive browser UI  
✅ **Test support** - Proper test directory  
✅ **Documentation** - Comprehensive docs in `docs/`  
✅ **No changes to RAG logic** - Core functionality preserved  

## What's Preserved

- ✅ Exact same RAG pipeline logic
- ✅ Vector store persistence (chroma_db/)
- ✅ Document processing
- ✅ LLM integration
- ✅ Configuration system
- ✅ CLI mode (backend/main.py)

## Quick Navigation

| Goal | File |
|------|------|
| Start server | `cd backend && python app.py` |
| Open UI | Open `frontend/index.html` in browser |
| CLI mode | `cd backend && python main.py` |
| Tests | `pytest tests/ -v` |
| Config | `backend/config.py` |
| API docs | `docs/API.md` |
| Help | `docs/QUICKSTART.md` |
