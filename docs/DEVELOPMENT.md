# Development Guide

## Project Architecture

```
Browser (frontend/index.html)
    ↓ HTTP/JSON
Flask Backend (backend/app.py)
    ↓
Embeddings + LLM + Vector Store
    ↓
ChromaDB
```

## File Organization

### Backend (`backend/`)
- `config.py` - Centralized configuration from `.env`
- `embeddings.py` - SentenceTransformer wrapper
- `vector_store.py` - ChromaDB wrapper
- `llm_client.py` - Groq/Gemini client
- `document_loader.py` - PDF/DOCX/TXT loading
- `app.py` - Flask REST API
- `main.py` - CLI interface

### Frontend (`frontend/`)
- `index.html` - Single-page chat application (vanilla JS)

### Tests (`tests/`)
- `test_vector_store.py` - Unit tests
- Run with: `pytest tests/ -v`

### Docs (`docs/`)
- `QUICKSTART.md` - Getting started
- `API.md` - API documentation
- `DEVELOPMENT.md` - This file

## Running Tests

```bash
# Install pytest
pip install pytest

# Run tests
cd tests
pytest -v
```

## Running CLI Version

```bash
cd backend
python main.py --samples      # With sample documents
python main.py --interactive  # Interactive mode
```

## Configuration

All settings in `backend/config.py`:
- LLM provider and model
- Embedding model
- Vector store path
- Document processing parameters

Environment variables in `.env`:
- API keys
- Model names
- Paths

## Adding New Features

### Adding API Endpoints

Edit `backend/app.py`:
```python
@app.route("/api/new_feature", methods=["POST"])
def new_feature():
    # Your code here
    return jsonify({
        "status": "success",
        "data": result
    }), 200
```

### Modifying RAG Pipeline

Edit `backend/app.py` in `query_chatbot()` function to change:
- Number of retrieved documents (top_k)
- System prompt
- Context building

### Adding Document Types

Edit `backend/document_loader.py`:
1. Add file extension to `SUPPORTED_FORMATS`
2. Implement `_extract_text_from_xyz()` method
3. Test with sample files

## Performance Optimization

### Faster Embeddings
- Use smaller model: `all-MiniLM-L6-v2` (default, 22M)
- Or CPU-only PyTorch (already configured)

### Faster LLM Responses
- Use Groq (faster than Gemini)
- Reduce `top_k` in queries
- Optimize prompts

### Vector Store Optimization
- Use HNSW index (default in ChromaDB)
- Reduce document size/chunk_size
- Filter results with metadata

## Import Paths

All Python modules are in `backend/`:
```python
# When running from backend/:
from config import settings
from vector_store import VectorStore
from embeddings import EmbeddingModel
```

## Debugging

### Enable Flask Debug Mode
```python
# In backend/app.py
if __name__ == "__main__":
    app.run(debug=True, ...)  # Already enabled
```

### Check Logs
- Backend logs: Terminal where `python app.py` runs
- Frontend logs: Browser console (F12)

### Test Individual Components
```python
# Test embeddings
from backend.embeddings import EmbeddingModel
model = EmbeddingModel()
embedding = model.embed("test text")

# Test vector store
from backend.vector_store import VectorStore
store = VectorStore()
store.upsert("id1", [0.1, 0.2], "text", {})
results = store.query([0.1, 0.2])
```

## Deployment

### Local Network
```bash
# Backend listens on 0.0.0.0:5000 (accessible from other machines)
cd backend
python app.py
```

### Production
- Use production WSGI server (Gunicorn/uWSGI)
- Add authentication
- Use environment variables for sensitive data
- Enable HTTPS
- Add rate limiting
- Use managed vector database

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```
