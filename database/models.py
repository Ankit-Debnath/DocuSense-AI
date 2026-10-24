"""
Database Models & Query Helpers
Provides clean, parameterized data-access functions for document records and summaries.
"""

import logging
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
    db_path: Optional[str] = None
) -> int:
    """
    Inserts a newly processed document record into the SQLite database.
    Uses parameterized queries ('?') to completely prevent SQL injection.

    Returns:
        int: The primary key ID of the newly inserted record.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    query = """
        INSERT INTO documents (filename, original_filename, page_count, character_count, status)
        VALUES (?, ?, ?, ?, ?)
    """
    cursor.execute(query, (filename, original_filename, page_count, character_count, status))
    conn.commit()

    doc_id = cursor.lastrowid
    if db_path is not None:
        conn.close()

    return doc_id


def get_all_documents(db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """
    Retrieves all previously uploaded documents, ordered newest first.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    query = """
        SELECT id, filename, original_filename, upload_time, page_count, character_count, status
        FROM documents
        ORDER BY upload_time DESC, id DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if db_path is not None:
        conn.close()

    return rows


def get_document_by_id(doc_id: int, db_path: Optional[str] = None) -> Optional[sqlite3.Row]:
    """
    Retrieves a single document record by its primary key ID.
    Uses parameterized query to prevent SQL injection.
    """
    conn = get_db(db_path)
    cursor = conn.cursor()

    query = """
        SELECT id, filename, original_filename, upload_time, page_count, character_count, status
        FROM documents
        WHERE id = ?
    """
    cursor.execute(query, (doc_id,))
    row = cursor.fetchone()

    if db_path is not None:
        conn.close()

    return row


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
    Retrieves a cached document summary if it was previously generated.
    Gracefully catches and logs database errors so cache failures never crash the app.
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
