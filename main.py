"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines a Vector Database (ChromaDB) with an LLM (OpenAI) to answer
questions grounded in your own documents.
"""

from vector_store import VectorStore
from llm_client import LLMClient
from embeddings import EmbeddingModel
from config import settings
import sys


def ingest_documents(vector_store: VectorStore, embedding_model: EmbeddingModel):
    """Load sample documents into the vector database."""
    documents = [
        {
            "id": "doc1",
            "text": "Large language models (LLMs) are neural networks trained on massive text corpora. "
                    "They learn statistical patterns in language and can generate coherent, context-aware text.",
            "metadata": {"source": "ai_overview.txt", "topic": "LLM"},
        },
        {
            "id": "doc2",
            "text": "Vector databases store high-dimensional embeddings and support fast similarity search "
                    "using algorithms like HNSW or IVF. Popular options include ChromaDB, Pinecone, and Weaviate.",
            "metadata": {"source": "vector_db_overview.txt", "topic": "VectorDB"},
        },
        {
            "id": "doc3",
            "text": "Retrieval-Augmented Generation (RAG) combines a retriever (vector search) with a generator "
                    "(LLM) to produce factually grounded answers without retraining the model.",
            "metadata": {"source": "rag_paper.txt", "topic": "RAG"},
        },
        {
            "id": "doc4",
            "text": "Transformer architecture, introduced in 'Attention Is All You Need' (2017), uses "
                    "self-attention mechanisms and is the backbone of modern LLMs like GPT, BERT, and LLaMA.",
            "metadata": {"source": "transformers.txt", "topic": "LLM"},
        },
        {
            "id": "doc5",
            "text": "Embeddings are dense numerical representations of text. Similar texts have vectors "
                    "that are close together in high-dimensional space, enabling semantic search.",
            "metadata": {"source": "embeddings_guide.txt", "topic": "VectorDB"},
        },
    ]

    print(f"📥 Ingesting {len(documents)} documents into vector store...")
    for doc in documents:
        embedding = embedding_model.embed(doc["text"])
        vector_store.upsert(doc["id"], embedding, doc["text"], doc["metadata"])

    print("✅ Ingestion complete.\n")


def answer_question(
    query: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    llm: LLMClient,
    top_k: int = 3,
) -> str:
    """Run the full RAG pipeline for a single query."""
    print(f"🔍 Query: {query}")

    # 1. Embed the query
    query_embedding = embedding_model.embed(query)

    # 2. Retrieve top-k similar chunks from the vector store
    results = vector_store.query(query_embedding, top_k=top_k)
    context_chunks = [r["text"] for r in results]

    print(f"📚 Retrieved {len(context_chunks)} relevant chunks.")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"  [{i}] {chunk[:80]}...")

    # 3. Build a prompt with retrieved context
    context = "\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Use only the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # 4. Generate answer with the LLM
    answer = llm.complete(prompt)
    return answer


def main():
    print("=" * 60)
    print("  LLM + Vector Database RAG Pipeline")
    print("=" * 60)
    print()

    # Initialize components
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    llm = LLMClient(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )

    # Ingest sample documents
    ingest_documents(vector_store, embedding_model)

    # Sample queries to demonstrate the pipeline
    queries = [
        "What is a large language model?",
        "How does vector similarity search work?",
        "What is RAG and why is it useful?",
    ]

    for query in queries:
        print("-" * 60)
        answer = answer_question(query, vector_store, embedding_model, llm)
        print(f"\n💬 Answer:\n{answer}\n")

    # Interactive mode
    if "--interactive" in sys.argv:
        print("=" * 60)
        print("  Interactive Mode — type 'exit' to quit")
        print("=" * 60)
        while True:
            user_query = input("\nYour question: ").strip()
            if user_query.lower() in ("exit", "quit"):
                break
            if user_query:
                print()
                answer = answer_question(user_query, vector_store, embedding_model, llm)
                print(f"\n💬 Answer:\n{answer}")


if __name__ == "__main__":
    main()
