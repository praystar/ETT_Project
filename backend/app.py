"""
Flask API server for the RAG Chatbot
Exposes REST endpoints to ingest documents and query the chatbot.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from vector_store import VectorStore
from llm_client import LLMClient
from embeddings import EmbeddingModel
from document_loader import DocumentLoader
from config import settings
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Get the relative path to frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

# Global state
_embedding_model = None
_vector_store = None
_llm = None
_initialized = False


def initialize_rag_system():
    """Initialize the RAG system components."""
    global _embedding_model, _vector_store, _llm, _initialized
    
    if _initialized:
        return
    
    try:
        _embedding_model = EmbeddingModel()
        _vector_store = VectorStore(
            collection_name=settings.CHROMA_COLLECTION,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        _llm = LLMClient(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )
        _initialized = True
        print("✅ RAG system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")
        raise


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "initialized": _initialized
    }), 200


@app.route("/api/initialize", methods=["POST"])
def initialize():
    """Initialize the RAG system."""
    try:
        initialize_rag_system()
        return jsonify({
            "status": "success",
            "message": "RAG system initialized"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    """Ingest documents from the documents directory."""
    if not _initialized:
        initialize_rag_system()
    
    try:
        loader = DocumentLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        # Load documents from directory
        documents = loader.load_files_from_directory(settings.DOCUMENTS_DIR)
        
        if not documents:
            # Load sample documents if folder is empty
            documents = _get_sample_documents()
            added_samples = True
        else:
            added_samples = False
        
        # Ingest documents into vector store
        ingested_count = 0
        for doc in documents:
            embedding = _embedding_model.embed(doc["text"])
            _vector_store.upsert(doc["id"], embedding, doc["text"], doc["metadata"])
            ingested_count += 1
        
        return jsonify({
            "status": "success",
            "message": f"Ingested {ingested_count} document chunks",
            "added_samples": added_samples,
            "count": ingested_count
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/query", methods=["POST"])
def query_chatbot():
    """Query the chatbot with a question."""
    if not _initialized:
        initialize_rag_system()
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        top_k = data.get("top_k", 3)
        
        if not query:
            return jsonify({
                "status": "error",
                "message": "Query cannot be empty"
            }), 400
        
        # 1. Embed the query
        query_embedding = _embedding_model.embed(query)
        
        # 2. Retrieve top-k similar chunks
        results = _vector_store.query(query_embedding, top_k=top_k)
        context_chunks = [r["text"] for r in results]
        sources = [r["metadata"] for r in results]
        
        # 3. Build prompt with retrieved context
        context = "\n\n".join(context_chunks)
        prompt = (
            f"You are a helpful assistant. Use only the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        
        # 4. Generate answer with the LLM
        answer = _llm.complete(prompt)
        
        return jsonify({
            "status": "success",
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_chunks": len(context_chunks)
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/clear", methods=["POST"])
def clear_database():
    """Clear the vector database."""
    if not _initialized:
        initialize_rag_system()
    
    try:
        _vector_store.delete_all()
        return jsonify({
            "status": "success",
            "message": "Vector database cleared"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def _get_sample_documents():
    """Return sample documents for demo purposes."""
    return [
        {
            "id": "sample_doc1_0_demo001",
            "text": "Large language models (LLMs) are neural networks trained on massive text corpora. "
                    "They learn statistical patterns in language and can generate coherent, context-aware text.",
            "metadata": {"source": "sample_ai_overview.txt", "file_type": "txt", "chunk_index": 0},
        },
        {
            "id": "sample_doc2_0_demo002",
            "text": "Vector databases store high-dimensional embeddings and support fast similarity search "
                    "using algorithms like HNSW or IVF. Popular options include ChromaDB, Pinecone, and Weaviate.",
            "metadata": {"source": "sample_vector_db_overview.txt", "file_type": "txt", "chunk_index": 0},
        },
        {
            "id": "sample_doc3_0_demo003",
            "text": "Retrieval-Augmented Generation (RAG) combines a retriever (vector search) with a generator "
                    "(LLM) to produce factually grounded answers without retraining the model.",
            "metadata": {"source": "sample_rag_paper.txt", "file_type": "txt", "chunk_index": 0},
        },
        {
            "id": "sample_doc4_0_demo004",
            "text": "Transformer architecture, introduced in 'Attention Is All You Need' (2017), uses "
                    "self-attention mechanisms and is the backbone of modern LLMs like GPT, BERT, and LLaMA.",
            "metadata": {"source": "sample_transformers.txt", "file_type": "txt", "chunk_index": 0},
        },
        {
            "id": "sample_doc5_0_demo005",
            "text": "Embeddings are dense numerical representations of text. Similar texts have vectors "
                    "that are close together in high-dimensional space, enabling semantic search.",
            "metadata": {"source": "sample_embeddings_guide.txt", "file_type": "txt", "chunk_index": 0},
        },
    ]


# ── Frontend Routes ───────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    """Serve the frontend HTML."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files from the frontend directory."""
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
