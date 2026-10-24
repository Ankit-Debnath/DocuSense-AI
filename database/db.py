"""
Database Connection & Setup
Manages the SQLite database connection, table creation, and teardown.
"""

import os
import sqlite3
from flask import g, current_app

def get_db(db_path=None):
    """
    Returns an active SQLite database connection.
    If running within a Flask application context, stores and reuses the connection on flask.g.
    If db_path is provided directly (e.g., in unit tests), connects to that path directly.
    """
    if db_path is not None:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    if "db" not in g:
        path = current_app.config.get("DATABASE_PATH")
        # Ensure instance directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        g.db = sqlite3.connect(path)
        # Enable column name access like dictionaries: row['filename']
        g.db.row_factory = sqlite3.Row

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

    conn = sqlite3.connect(db_path)
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
            status TEXT NOT NULL
        );
    """)

    # Create document_summaries cache table to prevent redundant re-summarization
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_summaries (
            filename TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            model_used TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
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
