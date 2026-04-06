# RAG Chatbot Frontend & API Guide

## Overview

The project now includes:
- **Backend API** (`app.py`): Flask server that exposes REST endpoints for the RAG pipeline
- **Frontend** (`frontend.html`): Modern web UI for interacting with the chatbot

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Set up your `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env and add:
# GROQ_API_KEY=your_key  (or)
# GEMINI_API_KEY=your_key
```

### 3. Start the Backend API

```bash
python app.py
```

The API will start on `http://localhost:5000`

You should see:
```
 * Running on http://0.0.0.0:5000
```

### 4. Open the Frontend

Open `frontend.html` in your browser:
- **From terminal**: `open frontend.html` (macOS) or `xdg-open frontend.html` (Linux)
- **Or**: Simply double-click the file from file explorer
- **Or**: Navigate to the file path in your browser location bar

## API Endpoints

### `POST /api/initialize`
Initialize the RAG system (embeddings, vector store, LLM).

**Response:**
```json
{
  "status": "success",
  "message": "RAG system initialized"
}
```

### `POST /api/ingest`
Ingest documents from the `documents/` directory (or load sample documents if empty).

**Response:**
```json
{
  "status": "success",
  "message": "Ingested X document chunks",
  "added_samples": false,
  "count": 5
}
```

### `POST /api/query`
Query the chatbot with a question.

**Request:**
```json
{
  "query": "What is RAG?",
  "top_k": 3
}
```

**Response:**
```json
{
  "status": "success",
  "query": "What is RAG?",
  "answer": "RAG combines...",
  "sources": [
    {"source": "sample_rag_paper.txt", "file_type": "txt"}
  ],
  "context_chunks": 3
}
```

### `POST /api/clear`
Clear the vector database.

**Response:**
```json
{
  "status": "success",
  "message": "Vector database cleared"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "initialized": true
}
```

## Using the Web UI

1. **Initialize System**: The system auto-initializes when you open the page
2. **Load Documents**: Click "📥 Ingest Documents" to load documents from the `documents/` folder
   - If the folder is empty, sample documents will be loaded automatically
3. **Ask Questions**: Type your question and press Enter or click "Send"
4. **View Sources**: Each response shows the source documents used for context
5. **Clear Database**: Click "🗑️ Clear Database" to reset (requires confirmation)

## File Structure

```
ETT_Project/
├── app.py                  # ← NEW: Flask API server
├── frontend.html           # ← NEW: Web UI
├── main.py                 # Original CLI interface
├── document_loader.py
├── embeddings.py
├── llm_client.py
├── vector_store.py
├── config.py
├── requirements.txt        # Updated with Flask dependencies
├── documents/              # Place your documents here
└── chroma_db/              # Vector database persistence
```

## Features

### Backend (Flask API)
- ✅ REST API for RAG pipeline
- ✅ Document ingestion from directory
- ✅ Query processing with retrieved context
- ✅ Source attribution
- ✅ CORS support for frontend communication
- ✅ Error handling and status reporting

### Frontend (Web UI)
- ✅ Modern, responsive chat interface
- ✅ Real-time message streaming
- ✅ Document source attribution
- ✅ Document management (ingest/clear)
- ✅ Loading states and error messages
- ✅ Mobile-friendly design
- ✅ Session state management

## Troubleshooting

### "Failed to fetch" errors
- **Cause**: Backend not running or CORS issue
- **Fix**: 
  1. Ensure `python app.py` is running
  2. Check that Flask is accessible at `http://localhost:5000`
  3. Try `curl http://localhost:5000/health` to verify

### "System not initialized"
- **Cause**: API failed to start RAG components
- **Fix**: 
  1. Check `.env` file has valid API keys
  2. Check terminal for error messages
  3. Ensure all dependencies are installed: `pip install -r requirements.txt`

### "No documents found"
- **Cause**: `documents/` folder is empty
- **Solution**: 
  - Add PDF, DOCX, or TXT files to the `documents/` folder
  - Or click "📥 Ingest Documents" to auto-load sample documents

### Slow responses
- **Cause**: Large documents, slow internet, or LLM provider delays
- **Solution**: 
  1. Start with fewer, smaller documents
  2. Try Groq instead of Gemini (faster)
  3. Reduce `top_k` in queries

## Development Notes

- **Backend**: Flask with CORS enabled for frontend communication
- **Frontend**: Vanilla JavaScript (no build tools required)
- **Database**: ChromaDB persists to `chroma_db/` directory
- **Embeddings**: Local SentenceTransformer model (no API key needed)
- **LLM**: Configurable (Groq or Gemini) via `.env`

## Next Steps

- Add your own documents to `documents/` folder
- Customize the system prompt in `app.py` (see `query_chatbot()` function)
- Deploy to production with proper authentication
- Add document upload UI feature
- Implement conversation history
