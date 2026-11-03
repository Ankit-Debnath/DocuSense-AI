"""
Database Models & Query Helpers
Provides clean, parameterized data-access functions for document records, summaries, and chunks.
Features binary embedding vector serialization via Python struct.
"""

import logging
import math
import struct
from typing import Any, Dict, List, Optional
import sqlite3
from database.db import get_db

logger = logging.getLogger("pdf_analyzer.database")


def insert_document(
    filename: str,
    original_filename: str,
    page_count: int,
    character_count: int,
    status: str = "processed",
    content_hash: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """
    Inserts a newly processed document record into the SQLite database.
    Uses parameterized queries ('?') to completely prevent SQL injection.

    Returns:
        int: The primary key ID of the newly inserted record.
    """
    conn = get_db(db_path)
    try:
        cursor = conn.cursor()

        query = """
            INSERT INTO documents (filename, original_filename, page_count, character_count, status, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (filename, original_filename, page_count, character_count, status, content_hash))
        conn.commit()

        doc_id = cursor.lastrowid
        return doc_id
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_all_documents(db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """
    Retrieves all previously uploaded documents, ordered newest first.
    """
    conn = get_db(db_path)
    try:
        cursor = conn.cursor()

        query = """
            SELECT id, filename, original_filename, upload_time, page_count, character_count, status, content_hash
            FROM documents
            ORDER BY upload_time DESC, id DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_by_id(doc_id: int, db_path: Optional[str] = None) -> Optional[sqlite3.Row]:
    """
    Retrieves a single document record by its primary key ID.
    Uses parameterized query to prevent SQL injection.
    """
    conn = get_db(db_path)
    try:
        cursor = conn.cursor()

        query = """
            SELECT id, filename, original_filename, upload_time, page_count, character_count, status, content_hash
            FROM documents
            WHERE id = ?
        """
        cursor.execute(query, (doc_id,))
        row = cursor.fetchone()
        return row
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_by_content_hash(
    content_hash: str,
    db_path: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """
    Retrieves an existing processed document matching a content hash (SHA-256).
    Returns the most recent matching record, or None if no match.
    """
    if not content_hash:
        return None

    conn = get_db(db_path)
    try:
        cursor = conn.cursor()

        query = """
            SELECT id, filename, original_filename, upload_time, page_count, character_count, status, content_hash
            FROM documents
            WHERE content_hash = ? AND status = 'processed'
            ORDER BY id DESC
            LIMIT 1
        """
        cursor.execute(query, (content_hash,))
        row = cursor.fetchone()
        return row
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_by_filename(
    filename: str,
    db_path: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """
    Retrieves a single document record by its stored filename.
    """
    if not filename:
        return None

    conn = get_db(db_path)
    try:
        cursor = conn.cursor()

        query = """
            SELECT id, filename, original_filename, upload_time, page_count, character_count, status, content_hash
            FROM documents
            WHERE filename = ?
            ORDER BY id DESC
            LIMIT 1
        """
        cursor.execute(query, (filename,))
        row = cursor.fetchone()
        return row
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def save_document_summary(
    filename: str,
    summary: str,
    model_used: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Saves or updates a generated summary for a document in the cache table.
    Gracefully catches and logs database errors so cache failures never crash the app.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = """
            INSERT INTO document_summaries (filename, summary, model_used, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(filename) DO UPDATE SET
                summary = excluded.summary,
                model_used = excluded.model_used,
                created_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (filename, summary, model_used))
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Cache save failed for '%s': %s", filename, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_summary(
    filename: str,
    db_path: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """
    Retrieves a cached summary from the database for a document.
    Returns None if no cached summary exists.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = """
            SELECT filename, summary, model_used, created_at
            FROM document_summaries
            WHERE filename = ?
        """
        cursor.execute(query, (filename,))
        row = cursor.fetchone()
        return row
    except Exception as exc:
        logger.warning("Cache lookup failed for '%s': %s", filename, exc)
        return None
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def delete_document_summary(
    filename: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Deletes a cached summary for a document (used when explicitly regenerating).
    Gracefully catches and logs database errors so cache failures never crash the app.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = "DELETE FROM document_summaries WHERE filename = ?"
        cursor.execute(query, (filename,))
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Cache deletion failed for '%s': %s", filename, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def save_page_explanation(
    filename: str,
    page_number: int,
    explanation: str,
    model_used: str,
    db_path: Optional[str] = None
) -> bool:
    """
    Saves or updates a structured technical explanation for a specific page in SQLite.
    Gracefully catches and logs errors so database issues never crash the app.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = """
            INSERT INTO page_explanations (filename, page_number, explanation, model_used, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(filename, page_number) DO UPDATE SET
                explanation = excluded.explanation,
                model_used = excluded.model_used,
                created_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (filename, page_number, explanation, model_used))
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Failed to save page explanation for '%s' p%d: %s", filename, page_number, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_page_explanation(
    filename: str,
    page_number: int,
    db_path: Optional[str] = None
) -> Optional[sqlite3.Row]:
    """
    Retrieves a cached page explanation from SQLite for a specific document and page.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = """
            SELECT filename, page_number, explanation, model_used, created_at
            FROM page_explanations
            WHERE filename = ? AND page_number = ?
        """
        cursor.execute(query, (filename, page_number))
        return cursor.fetchone()
    except Exception as exc:
        logger.warning("Failed to get page explanation for '%s' p%d: %s", filename, page_number, exc)
        return None
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_page_explanations(
    filename: str,
    db_path: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Retrieves all cached page explanations for a document as a dict keyed by page_number.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = """
            SELECT page_number, explanation, model_used, created_at
            FROM page_explanations
            WHERE filename = ?
            ORDER BY page_number ASC
        """
        cursor.execute(query, (filename,))
        rows = cursor.fetchall()
        result: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            result[row["page_number"]] = {
                "explanation": row["explanation"],
                "model_used": row["model_used"],
                "created_at": row["created_at"]
            }
        return result
    except Exception as exc:
        logger.warning("Failed to get all page explanations for '%s': %s", filename, exc)
        return {}
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def delete_page_explanation(
    filename: str,
    page_number: int,
    db_path: Optional[str] = None
) -> bool:
    """
    Deletes a specific page explanation cache entry.
    """
    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()

        query = "DELETE FROM page_explanations WHERE filename = ? AND page_number = ?"
        cursor.execute(query, (filename, page_number))
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Failed to delete page explanation for '%s' p%d: %s", filename, page_number, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Vector Embedding Serialization (Python struct IEEE 754 32-bit float '<f')
# ---------------------------------------------------------------------------

def serialize_embedding(vector: List[float]) -> bytes:
    """
    Serializes a floating-point embedding vector into a compact binary BLOB using Python's struct module.

    Representation:
        Uses standard IEEE 754 32-bit single-precision floating point ('<f' little-endian).
        Serializing Python 64-bit double-precision floats to 32-bit floats introduces minor precision
        loss beyond 7 significant decimal digits (e.g. ~1e-7 relative difference), which is the standard,
        native representation for deep learning embedding vectors while halving the storage size compared to 64-bit doubles.

    Parameters:
        vector: Non-empty sequence of finite numeric float values.

    Returns:
        bytes: Binary representation (4 bytes per dimension).

    Raises:
        ValueError: If vector is empty, contains non-numeric values, or contains non-finite values (NaN/Inf).
    """
    if not isinstance(vector, (list, tuple)) or len(vector) == 0:
        raise ValueError("Embedding vector must be a non-empty sequence of numbers.")

    clean_vector: List[float] = []
    for idx, val in enumerate(vector):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Embedding value at index {idx} is non-numeric: {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val):
            raise ValueError(f"Embedding value at index {idx} is non-finite (NaN or Inf): {val!r}")
        clean_vector.append(f_val)

    return struct.pack(f"<{len(clean_vector)}f", *clean_vector)


def deserialize_embedding(blob: bytes, expected_dimension: Optional[int] = None) -> List[float]:
    """
    Deserializes a binary BLOB into a list of Python floats using IEEE 754 32-bit float unpack ('<f').

    Parameters:
        blob: Binary bytes object containing serialized 32-bit floats.
        expected_dimension: Optional expected dimension to validate.

    Returns:
        List[float]: The recovered embedding vector.

    Raises:
        TypeError: If blob is not a bytes-like object.
        ValueError: If blob is empty, length is not a multiple of 4, dimension mismatches,
                    or unpacked values contain non-finite numbers.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"Embedding BLOB must be bytes-like object, got {type(blob).__name__}.")

    if len(blob) == 0:
        raise ValueError("Embedding BLOB cannot be empty.")

    if len(blob) % 4 != 0:
        raise ValueError(f"Malformed embedding BLOB length ({len(blob)} bytes): must be a multiple of 4.")

    dimension = len(blob) // 4
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, but BLOB contains {dimension} dimensions."
        )

    raw_floats = struct.unpack(f"<{dimension}f", blob)
    clean_vector = list(raw_floats)
    for idx, val in enumerate(clean_vector):
        if not math.isfinite(val):
            raise ValueError(f"Malformed embedding BLOB: value at index {idx} unpacked to non-finite float.")

    return clean_vector


# ---------------------------------------------------------------------------
# Document Chunks Persistence Helpers
# ---------------------------------------------------------------------------

def save_document_chunks(
    document_id: int,
    chunks: List[Dict[str, Any]],
    db_path: Optional[str] = None
) -> bool:
    """
    Atomically saves or replaces chunks for a document in SQLite.
    Executes within a single transaction with automatic rollback on error.

    Parameters:
        document_id: Existing document integer ID.
        chunks: List of chunk dictionaries. Each dict should contain:
            - chunk_index: int (0-based stable index)
            - page_number: int (1-based PDF page)
            - chunk_text: str (non-empty chunk text)
            - char_count: Optional[int] (defaults to len(chunk_text))
            - embedding: bytes (serialized BLOB) or List[float] (will be serialized)
            - embedding_dim: int (e.g. 768)
            - embedding_model: str (e.g. 'models/gemini-embedding-001')
        db_path: Optional custom database path for testing.

    Returns:
        bool: True if chunks were successfully persisted, False otherwise.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        logger.error("Invalid document_id for saving chunks: %r", document_id)
        return False

    if not isinstance(chunks, list):
        logger.error("Chunks must be a list for document_id=%d", document_id)
        return False

    conn = None
    try:
        conn = get_db(db_path)
        with conn:
            cursor = conn.cursor()

            # Verify document exists to preserve foreign-key integrity
            cursor.execute("SELECT id FROM documents WHERE id = ?", (document_id,))
            if not cursor.fetchone():
                logger.error("Cannot save chunks: Document id=%d does not exist.", document_id)
                return False

            # Delete any previous chunks for this document to ensure clean re-indexing
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))

            if not chunks:
                return True

            prepared_rows = []
            for item in chunks:
                chunk_idx = item["chunk_index"]
                page_num = item["page_number"]
                chunk_txt = item["chunk_text"]
                char_count = item.get("char_count", len(chunk_txt))
                raw_emb = item["embedding"]
                dim = item["embedding_dim"]
                model_name = item.get("embedding_model") or "gemini-embedding-001"

                # Serialize if list of floats passed
                if isinstance(raw_emb, (list, tuple)):
                    blob_emb = serialize_embedding(raw_emb)
                elif isinstance(raw_emb, (bytes, bytearray, memoryview)):
                    blob_emb = bytes(raw_emb)
                else:
                    raise TypeError(f"Invalid embedding type: {type(raw_emb).__name__}")

                prepared_rows.append((
                    document_id,
                    chunk_idx,
                    page_num,
                    chunk_txt,
                    char_count,
                    blob_emb,
                    dim,
                    model_name
                ))

            insert_query = """
                INSERT INTO document_chunks (
                    document_id,
                    chunk_index,
                    page_number,
                    chunk_text,
                    char_count,
                    embedding,
                    embedding_dim,
                    embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.executemany(insert_query, prepared_rows)
        return True
    except Exception as exc:
        logger.error("Failed to save chunks for document_id=%d: %s", document_id, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def get_document_chunks(
    document_id: int,
    db_path: Optional[str] = None
) -> List[sqlite3.Row]:
    """
    Retrieves all stored chunks and embeddings for a document, ordered by chunk_index.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return []

    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()
        query = """
            SELECT id, document_id, chunk_index, page_number, chunk_text,
                   char_count, embedding, embedding_dim, embedding_model, created_at
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
        """
        cursor.execute(query, (document_id,))
        return cursor.fetchall()
    except Exception as exc:
        logger.error("Failed to retrieve chunks for document_id=%d: %s", document_id, exc)
        return []
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def count_document_chunks(
    document_id: int,
    db_path: Optional[str] = None
) -> int:
    """
    Returns the count of indexed chunks stored for a document.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return 0

    conn = None
    try:
        conn = get_db(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE document_id = ?", (document_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception as exc:
        logger.error("Failed to count chunks for document_id=%d: %s", document_id, exc)
        return 0
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def delete_document_chunks(
    document_id: int,
    db_path: Optional[str] = None
) -> bool:
    """
    Deletes all stored chunks for a specific document.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return False

    conn = None
    try:
        conn = get_db(db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        return True
    except Exception as exc:
        logger.error("Failed to delete chunks for document_id=%d: %s", document_id, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()


def delete_document(
    document_id: int,
    db_path: Optional[str] = None
) -> bool:
    """
    Deletes a document record from the documents table.
    Foreign key CASCADE automatically removes associated document_chunks when enabled.
    Also explicitly cleans up associated document_chunks, summaries, and page explanations.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return False

    conn = None
    try:
        conn = get_db(db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                return False
            stored_filename = row["filename"]

            # Explicit cascade cleanup for robustness across all SQLite connection modes
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM document_summaries WHERE filename = ?", (stored_filename,))
            cursor.execute("DELETE FROM page_explanations WHERE filename = ?", (stored_filename,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return True
    except Exception as exc:
        logger.error("Failed to delete document_id=%d: %s", document_id, exc)
        return False
    finally:
        if db_path is not None and conn is not None:
            conn.close()
