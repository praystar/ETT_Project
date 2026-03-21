"""
watch_documents.py — Continuously monitor the documents folder for new files
and automatically ingest them into the vector store.

Usage:
    python watch_documents.py
"""

import time
import sys
from pathlib import Path
from datetime import datetime

from document_loader import DocumentLoader
from vector_store import VectorStore
from embeddings import EmbeddingModel
from config import settings


class DocumentWatcher:
    """
    Monitors the documents directory for file changes and automatically
    ingests new or modified documents into the vector store.
    """

    def __init__(self):
        self.loader = DocumentLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.vector_store = VectorStore(
            collection_name=settings.CHROMA_COLLECTION,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        self.embedding_model = EmbeddingModel()
        self.documents_dir = Path(settings.DOCUMENTS_DIR)
        self.tracked_files = {}  # {file_path: last_modified_time}

    def start(self):
        """Start watching the documents directory."""
        print("=" * 60)
        print("  Document Watcher — Continuous Folder Monitoring")
        print("=" * 60)
        print(f"\n👀 Watching directory: {self.documents_dir.absolute()}")
        print("   Supported formats: .pdf, .docx, .txt")
        print("   Press Ctrl+C to stop.\n")

        # Ensure directory exists
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        # Perform initial scan
        self._scan_and_ingest()

        # Continuous monitoring loop
        try:
            while True:
                time.sleep(2)  # Check for changes every 2 seconds
                self._scan_and_ingest()
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("  Watcher stopped. Goodbye!")
            print("=" * 60)
            sys.exit(0)

    def _scan_and_ingest(self):
        """Scan for new or modified files and ingest them."""
        current_files = {}

        # Collect all supported files in the directory
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.loader.SUPPORTED_FORMATS:
                last_modified = file_path.stat().st_mtime
                current_files[str(file_path)] = last_modified

        # Check for new or modified files
        for file_path_str, last_modified in current_files.items():
            if file_path_str not in self.tracked_files or self.tracked_files[file_path_str] < last_modified:
                self._ingest_file(Path(file_path_str))
                self.tracked_files[file_path_str] = last_modified

        # Check for deleted files (optional logging)
        deleted_files = set(self.tracked_files.keys()) - set(current_files.keys())
        for deleted_path in deleted_files:
            print(f"🗑️  File removed: {Path(deleted_path).name}")
            del self.tracked_files[deleted_path]

    def _ingest_file(self, file_path: Path):
        """Extract and ingest a single file."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️  Processing: {file_path.name}...")

        try:
            text = self.loader._extract_text(file_path)
            if not text or not text.strip():
                print(f"           ⚠️  File is empty, skipped")
                return

            # Chunk the text
            chunks = self.loader._chunk_text(text)

            # Ingest each chunk
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = self.loader._generate_doc_id(file_path, chunk_idx)
                embedding = self.embedding_model.embed(chunk)
                metadata = {
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip("."),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "imported_at": datetime.now().isoformat(),
                }
                self.vector_store.upsert(doc_id, embedding, chunk, metadata)

            total_docs = self.vector_store.count()
            print(f"           ✅ Ingested ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''}) — Total: {total_docs} docs in store")

        except Exception as e:
            print(f"           ❌ Error: {e}")


def main():
    """Main entry point."""
    if not settings.WATCH_ENABLED:
        print("⚠️  Document watching is disabled in config (WATCH_ENABLED=False)")
        print("   To enable, set WATCH_ENABLED=True in .env or config.py")
        sys.exit(1)

    watcher = DocumentWatcher()
    watcher.start()


if __name__ == "__main__":
    main()
