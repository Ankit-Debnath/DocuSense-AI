"""
Document Indexing Service Module
Coordinates text extraction, semantic chunking, batched Google GenAI embedding generation,
and atomic persistence in SQLite for semantic RAG.

Failure Safety:
  - API calls to Google GenAI are executed entirely outside database write transactions.
  - Previous valid indexes are preserved if embedding generation fails.
  - Index replacement in SQLite is atomic within a short write transaction.
  - Documents with no extractable text are rejected safely without producing misleading indexes.
  - Sensitive document contents and vector BLOBs are never logged.
"""

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from pdf_processing.extractor import extract_text_from_pdf
from pdf_processing.chunker import chunk_extracted_pages
from ai.embedding_client import embed_documents, get_embedding_model_name, EmbeddingError
from ai.analyzer import _format_error_message
from database.models import (
    get_document_by_id,
    save_document_chunks,
    count_document_chunks,
    get_document_chunks,
    delete_document_chunks
)

logger = logging.getLogger("pdf_analyzer.indexing")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Fine-grained lock registry to prevent concurrent simultaneous indexing of the same document
_indexing_locks_mutex = threading.Lock()
_document_indexing_locks: Dict[int, threading.Lock] = {}


def _get_document_lock(document_id: int) -> threading.Lock:
    """Retrieves or creates a thread-safe mutex for a specific document ID."""
    with _indexing_locks_mutex:
        if document_id not in _document_indexing_locks:
            _document_indexing_locks[document_id] = threading.Lock()
        return _document_indexing_locks[document_id]


class IndexingError(Exception):
    """Raised when document indexing fails."""
    pass


def get_document_indexing_status(
    document_id: int,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Checks whether a document has been successfully indexed with embeddings in SQLite.

    Parameters:
        document_id: Integer primary key ID of the document.
        db_path: Optional database path override.

    Returns:
        Dict[str, Any]: Status dict with 'document_id', 'indexed', and 'chunk_count'.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return {
            "document_id": document_id,
            "indexed": False,
            "chunk_count": 0,
            "error": "Invalid document ID."
        }

    count = count_document_chunks(document_id, db_path=db_path)
    return {
        "document_id": document_id,
        "indexed": count > 0,
        "chunk_count": count,
        "error": None
    }


def ensure_document_indexed(
    document_id: int,
    file_path: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ensures that a document is indexed in SQLite vector storage.
    If already indexed, reuses the existing index immediately without re-indexing.
    If not yet indexed, executes index_document() under a document-level lock
    to safely prevent redundant simultaneous Gemini embedding calls.

    Parameters:
        document_id: Integer document ID.
        file_path: Optional path to PDF file.
        client: Optional Gemini client.
        model_name: Optional embedding model name override.
        batch_size: Optional embedding batch size.
        chunk_size: Optional target chunk size in characters.
        chunk_overlap: Optional chunk overlap in characters.
        db_path: Optional SQLite database path.

    Returns:
        Dict[str, Any]: Index status dict with 'document_id', 'indexed', 'chunk_count',
                        'freshly_indexed', and 'error'.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return {
            "document_id": document_id,
            "indexed": False,
            "chunk_count": 0,
            "freshly_indexed": False,
            "error": f"Invalid document ID: {document_id!r}"
        }

    # 1. Fast check without acquiring document lock
    current_status = get_document_indexing_status(document_id, db_path=db_path)
    if current_status.get("indexed") and current_status.get("chunk_count", 0) > 0:
        return {
            "document_id": document_id,
            "indexed": True,
            "chunk_count": current_status["chunk_count"],
            "freshly_indexed": False,
            "error": None
        }

    # 2. Acquire per-document lock to prevent concurrent redundant indexing
    doc_lock = _get_document_lock(document_id)
    with doc_lock:
        # Double check inside the lock
        current_status = get_document_indexing_status(document_id, db_path=db_path)
        if current_status.get("indexed") and current_status.get("chunk_count", 0) > 0:
            return {
                "document_id": document_id,
                "indexed": True,
                "chunk_count": current_status["chunk_count"],
                "freshly_indexed": False,
                "error": None
            }

        # 3. Perform on-demand indexing
        index_res = index_document(
            document_id=document_id,
            file_path=file_path,
            client=client,
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db_path=db_path
        )

        return {
            "document_id": document_id,
            "indexed": index_res.get("success", False),
            "chunk_count": index_res.get("chunk_count", 0),
            "freshly_indexed": index_res.get("success", False),
            "error": index_res.get("error")
        }


def index_document(
    document_id: int,
    file_path: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Indexes a document by extracting text, chunking, generating embeddings in bounded batches,
    and atomically persisting the chunks to SQLite.

    Transaction Safety Guarantee:
        1. Document existence is checked.
        2. Text is extracted and chunked.
        3. All embeddings are generated and validated outside of any database transaction.
           If the API call fails or times out, the existing index in the database is untouched.
        4. The index is replaced atomically inside SQLite (delete old + insert new in one transaction).

    Parameters:
        document_id: Integer primary key of the document record.
        file_path: Optional path to PDF. If None, resolved from document record and upload folder.
        client: Optional pre-configured Gemini client.
        model_name: Optional embedding model name override.
        batch_size: Optional batch size override for embedding calls.
        chunk_size: Optional character target for chunking.
        chunk_overlap: Optional character overlap for chunking.
        db_path: Optional database path override.

    Returns:
        Dict[str, Any]: Result dictionary containing 'success', 'document_id', 'chunk_count',
                        'embedding_dim', 'embedding_model', and 'error'.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"Invalid document ID: {document_id!r}"
        }

    # 1. Verify document record exists
    doc_row = get_document_by_id(document_id, db_path=db_path)
    if not doc_row:
        logger.warning("Indexing failed: Document id=%d not found in database.", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"Document with ID {document_id} does not exist."
        }

    # 2. Resolve PDF file path
    resolved_path = file_path
    if not resolved_path:
        upload_folder = None
        try:
            from flask import current_app
            if current_app:
                upload_folder = current_app.config.get("UPLOAD_FOLDER")
        except Exception:
            pass

        if not upload_folder:
            try:
                from config import Config
                upload_folder = getattr(Config, "UPLOAD_FOLDER", "uploads")
            except Exception:
                upload_folder = "uploads"

        resolved_path = os.path.join(upload_folder, doc_row["filename"])

    if not os.path.isfile(resolved_path):
        logger.warning(
            "Indexing failed: PDF file '%s' for document id=%d not found on disk.",
            resolved_path,
            document_id
        )
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"PDF file not found on disk: {resolved_path}"
        }

    # 3. Extract text from PDF
    extraction = extract_text_from_pdf(resolved_path)
    if not extraction.get("success"):
        err_msg = extraction.get("error") or "PDF text extraction failed."
        logger.warning("Indexing failed: extraction error for document id=%d: %s", document_id, err_msg)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": err_msg
        }

    pages = extraction.get("pages") or []
    total_chars = extraction.get("total_characters", 0)
    if not pages or total_chars <= 0:
        logger.warning("Indexing failed: Document id=%d contains no extractable text.", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": "Document contains no extractable text to index."
        }

    # 4. Chunk extracted text
    try:
        chunks = chunk_extracted_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    except Exception as exc:
        clean_msg = _format_error_message(exc)
        logger.warning("Indexing failed: chunking error for document id=%d: %s", document_id, clean_msg)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"Chunking failed: {clean_msg}"
        }

    if not chunks:
        logger.warning("Indexing failed: No chunks generated for document id=%d.", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": "No chunks could be produced from document text."
        }

    # 5. Generate embeddings in bounded batches completely OUTSIDE the database transaction
    chunk_texts = [c["text"] for c in chunks]
    resolved_model = model_name or get_embedding_model_name()

    try:
        embeddings = embed_documents(
            texts=chunk_texts,
            client=client,
            model_name=resolved_model,
            batch_size=batch_size
        )
    except Exception as exc:
        clean_msg = _format_error_message(exc)
        logger.warning(
            "Indexing failed: embedding generation failed for document id=%d (%d chunks): %s",
            document_id,
            len(chunk_texts),
            clean_msg
        )
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"Embedding generation failed: {clean_msg}"
        }

    if len(embeddings) != len(chunks):
        logger.warning(
            "Indexing failed: count mismatch for document id=%d (chunks=%d, embeddings=%d).",
            document_id,
            len(chunks),
            len(embeddings)
        )
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks."
        }

    # 6. Prepare chunk records for atomic persistence
    embedding_dim = len(embeddings[0]) if embeddings else 0
    prepared_records = []
    for chunk, emb in zip(chunks, embeddings):
        prepared_records.append({
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk["page_number"],
            "chunk_text": chunk["text"],
            "char_count": chunk.get("char_count", len(chunk["text"])),
            "embedding": emb,
            "embedding_dim": len(emb),
            "embedding_model": resolved_model
        })

    # 7. Atomically persist to SQLite
    persisted = save_document_chunks(
        document_id=document_id,
        chunks=prepared_records,
        db_path=db_path
    )

    if not persisted:
        logger.error("Indexing failed: Atomic database persistence failed for document id=%d.", document_id)
        return {
            "success": False,
            "document_id": document_id,
            "chunk_count": 0,
            "error": "Failed to persist document chunks into database."
        }

    logger.info(
        "Document id=%d successfully indexed: %d chunks, dimension=%d, model='%s'.",
        document_id,
        len(prepared_records),
        embedding_dim,
        resolved_model
    )

    return {
        "success": True,
        "document_id": document_id,
        "chunk_count": len(prepared_records),
        "embedding_dim": embedding_dim,
        "embedding_model": resolved_model,
        "error": None
    }
