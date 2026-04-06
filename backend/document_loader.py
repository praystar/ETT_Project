"""
document_loader.py — Loads and processes PDF, DOCX, and TXT files from a directory.

Handles text extraction, chunking, and unique ID generation for document ingestion.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class DocumentLoader:
    """
    Loads documents from a directory, extracts text, and chunks them
    for embedding and vector store ingestion.
    """

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document loader.

        Args:
            chunk_size: Approximate number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_files_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        Recursively scan a directory and extract text from all supported files.

        Returns:
            A list of document dicts, each with keys:
            - id: unique document ID (format: {filename}_{chunk_index}_{hash})
            - text: extracted and chunked text
            - metadata: dict with source, file_type, chunk_index, etc.
        """
        directory = Path(directory)
        if not directory.exists():
            print(f"⚠️  Directory '{directory}' does not exist. Creating it...")
            directory.mkdir(parents=True, exist_ok=True)
            return []

        documents = []
        supported_files = list(directory.rglob("*"))
        supported_files = [
            f for f in supported_files if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]

        if not supported_files:
            print(f"ℹ️  No supported documents found in '{directory}'")
            print(f"   Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
            return []

        print(f"📂 Found {len(supported_files)} document(s) in '{directory}'")

        for file_path in supported_files:
            print(f"   📄 Processing: {file_path.name}...", end=" ")
            try:
                # Extract text based on file type
                text = self._extract_text(file_path)
                if not text or not text.strip():
                    print("⚠️  (empty, skipped)")
                    continue

                # Chunk the text
                chunks = self._chunk_text(text)

                # Generate documents for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = self._generate_doc_id(file_path, chunk_idx)
                    documents.append(
                        {
                            "id": doc_id,
                            "text": chunk,
                            "metadata": {
                                "source": file_path.name,
                                "file_type": file_path.suffix.lower().lstrip("."),
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks),
                                "imported_at": datetime.now().isoformat(),
                            },
                        }
                    )

                print(f"✅ ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")

            except Exception as e:
                print(f"❌ (error: {e})")
                continue

        return documents

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF, DOCX, or TXT file."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        elif suffix == ".docx":
            return self._extract_text_from_docx(file_path)
        elif suffix == ".txt":
            return self._extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF processing. Install with: pip install pypdf")

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")

        doc = Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text

    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from a TXT file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on word count.

        Args:
            text: Raw text to chunk

        Returns:
            List of text chunks
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def _generate_doc_id(self, file_path: Path, chunk_idx: int) -> str:
        """
        Generate a unique document ID.

        Format: {sanitized_filename}_{chunk_index}_{hash}
        """
        # Sanitize filename (remove extension, replace special chars)
        safe_name = file_path.stem.replace(" ", "_").replace("-", "_")

        # Create a short hash from the full path to avoid collisions
        path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]

        return f"{safe_name}_{chunk_idx}_{path_hash}"
