# RAG Chatbot Frontend & API Guide

## Overview

The project now includes:
- **Backend API** (`backend/app.py`): Flask server that exposes REST endpoints for the RAG pipeline
- **Frontend** (`frontend/index.html`): Modern web UI for interacting with the chatbot

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
cd backend
python app.py
```

The API will start on `http://localhost:5000`

### 4. Open the Frontend

Open `frontend/index.html` in your browser or use the startup script.

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
Ingest documents from the `documents/` directory.

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

## Project Structure

```
ETT_Project/
├── backend/                    # Python backend modules
│   ├── app.py                 # Flask API server
│   ├── config.py              # Configuration
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── embeddings.py          # Embeddings model
│   ├── llm_client.py          # LLM client
│   ├── document_loader.py     # Document processing
│   └── main.py                # CLI interface
├── frontend/                   # Web UI
│   └── index.html             # Chat interface
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── documents/                  # User documents folder
├── chroma_db/                  # Vector database
├── requirements.txt
├── .env                        # API keys
└── start.sh / start.bat        # Startup scripts
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
