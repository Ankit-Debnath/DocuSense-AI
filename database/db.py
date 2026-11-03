"""
Database Connection & Setup
Manages the SQLite database connection, table creation, and teardown.
"""

import os
import sqlite3
from flask import g, current_app

DEFAULT_DB_TIMEOUT = 15.0
DEFAULT_BUSY_TIMEOUT_MS = 15000


def _configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """
    Configures row factory, explicit busy timeout, and foreign key enforcement on a SQLite connection.
    """
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_db(db_path=None):
    """
    Returns an active SQLite database connection.
    If running within a Flask application context, stores and reuses the connection on flask.g.
    If db_path is provided directly (e.g., in unit tests), connects to that path directly.
    """
    if db_path is not None:
        conn = sqlite3.connect(db_path, timeout=DEFAULT_DB_TIMEOUT)
        return _configure_connection(conn)

    if "db" not in g:
        path = current_app.config.get("DATABASE_PATH")
        # Ensure instance directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, timeout=DEFAULT_DB_TIMEOUT)
        g.db = _configure_connection(conn)

    return g.db


def close_db(e=None):
    """
    Closes the database connection at the end of the Flask request lifecycle.
    """
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(db_path=None):
    """
    Initializes the SQLite database and creates necessary tables if they do not exist.
    """
    if db_path is None:
        db_path = current_app.config.get("DATABASE_PATH")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=DEFAULT_DB_TIMEOUT)
    _configure_connection(conn)
    try:
        # Enable Write-Ahead Logging (WAL) for superior reader/writer concurrency
        cursor = conn.execute("PRAGMA journal_mode = WAL;")
        result = cursor.fetchone()
        actual_mode = str(result[0]).strip().lower() if result else ""
        if actual_mode != "wal":
            raise RuntimeError(
                f"Failed to enable SQLite WAL mode on '{db_path}'. "
                f"Expected 'wal', but got '{actual_mode}'."
            )

        with conn:
            cursor = conn.cursor()

            # Create the documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_count INTEGER NOT NULL,
                    character_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    content_hash TEXT
                );
            """)

            # Safe backward-compatible migration: add content_hash column if missing in legacy databases
            cursor.execute("PRAGMA table_info(documents);")
            columns = [col[1] for col in cursor.fetchall()]
            if "content_hash" not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);")

            # Create document_summaries cache table to prevent redundant re-summarization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_summaries (
                    filename TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create page_explanations cache table for page-by-page technical walkthroughs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS page_explanations (
                    filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    explanation TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (filename, page_number)
                );
            """)

            # Create document_chunks table for persistent vector storage and semantic RAG
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    embedding_model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, chunk_index)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(document_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(document_id, page_number);")
    finally:
        conn.close()


def init_app(app):
    """
    Registers database functions with the Flask application:
    - Automatically closes database connection on request teardown
    - Initializes the database tables at application startup
    """
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
