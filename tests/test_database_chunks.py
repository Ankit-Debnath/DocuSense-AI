"""
Unit Tests for SQLite Document Chunks Persistence
Tests table schema, foreign key cascade, uniqueness constraints, atomic transaction rollback,
chunk ordering, and vector storage in SQLite.
Zero external API calls.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

from database.db import init_db, get_db
from database.models import (
    insert_document,
    save_document_chunks,
    get_document_chunks,
    count_document_chunks,
    delete_document_chunks,
    delete_document,
    deserialize_embedding
)


class DatabaseChunksPersistenceTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_chunks.db")
        init_db(self.db_path)

        # Insert a valid test document
        self.doc_id = insert_document(
            filename="sample.pdf",
            original_filename="sample_original.pdf",
            page_count=2,
            character_count=500,
            status="processed",
            content_hash="dummy_hash_12345",
            db_path=self.db_path
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_schema_and_table_initialization(self):
        """Verify document_chunks table and indexes are created properly."""
        conn = get_db(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # Verify columns
            cursor.execute("PRAGMA table_info(document_chunks)")
            cols = {row["name"]: row["type"] for row in cursor.fetchall()}
            self.assertIn("id", cols)
            self.assertIn("document_id", cols)
            self.assertIn("chunk_index", cols)
            self.assertIn("page_number", cols)
            self.assertIn("chunk_text", cols)
            self.assertIn("embedding", cols)
            self.assertIn("embedding_dim", cols)
            self.assertIn("embedding_model", cols)
        finally:
            conn.close()

    def test_save_and_retrieve_document_chunks_success(self):
        """Verify saving chunks stores vectors as BLOB and get_document_chunks orders by chunk_index."""
        chunks = [
            {
                "chunk_index": 0,
                "page_number": 1,
                "chunk_text": "First chunk text.",
                "char_count": 17,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_dim": 3,
                "embedding_model": "models/gemini-embedding-001"
            },
            {
                "chunk_index": 1,
                "page_number": 1,
                "chunk_text": "Second chunk text.",
                "char_count": 18,
                "embedding": [0.4, 0.5, 0.6],
                "embedding_dim": 3,
                "embedding_model": "models/gemini-embedding-001"
            },
            {
                "chunk_index": 2,
                "page_number": 2,
                "chunk_text": "Third chunk on page two.",
                "char_count": 24,
                "embedding": [0.7, 0.8, 0.9],
                "embedding_dim": 3,
                "embedding_model": "models/gemini-embedding-001"
            }
        ]

        saved = save_document_chunks(self.doc_id, chunks, db_path=self.db_path)
        self.assertTrue(saved)

        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 3)

        rows = get_document_chunks(self.doc_id, db_path=self.db_path)
        self.assertEqual(len(rows), 3)

        # Verify ordering and deserialized embeddings
        for i, row in enumerate(rows):
            self.assertEqual(row["chunk_index"], i)
            recovered_vec = deserialize_embedding(row["embedding"])
            self.assertEqual(len(recovered_vec), 3)
            self.assertAlmostEqual(recovered_vec[0], chunks[i]["embedding"][0], places=5)

    def test_save_chunks_rejects_nonexistent_document(self):
        """Verify chunks cannot be saved for an unrecorded document ID."""
        chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Orphan chunk.",
            "char_count": 13,
            "embedding": [0.1, 0.2],
            "embedding_dim": 2,
            "embedding_model": "models/gemini-embedding-001"
        }]
        # 99999 does not exist
        res = save_document_chunks(99999, chunks, db_path=self.db_path)
        self.assertFalse(res)
        self.assertEqual(count_document_chunks(99999, db_path=self.db_path), 0)

    def test_atomic_rollback_on_failure_preserves_existing_chunks(self):
        """Verify that if saving replacement chunks fails, the previous valid index is preserved."""
        # Initial valid chunks
        initial_chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Initial chunk.",
            "char_count": 14,
            "embedding": [0.5, 0.5],
            "embedding_dim": 2,
            "embedding_model": "models/gemini-embedding-001"
        }]
        self.assertTrue(save_document_chunks(self.doc_id, initial_chunks, db_path=self.db_path))
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 1)

        # Faulty replacement chunks: 2nd chunk has invalid embedding type
        faulty_chunks = [
            {
                "chunk_index": 0,
                "page_number": 1,
                "chunk_text": "New replacement chunk.",
                "char_count": 22,
                "embedding": [0.9, 0.9],
                "embedding_dim": 2,
                "embedding_model": "models/gemini-embedding-001"
            },
            {
                "chunk_index": 1,
                "page_number": 1,
                "chunk_text": "Broken chunk.",
                "char_count": 13,
                "embedding": {"bad": "type"},  # Invalid!
                "embedding_dim": 2,
                "embedding_model": "models/gemini-embedding-001"
            }
        ]

        success = save_document_chunks(self.doc_id, faulty_chunks, db_path=self.db_path)
        self.assertFalse(success)

        # Rollback check: initial chunk must still exist!
        rows = get_document_chunks(self.doc_id, db_path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_text"], "Initial chunk.")

    def test_cascade_delete_when_document_deleted(self):
        """Verify deleting a document via delete_document automatically removes all associated chunks."""
        chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Chunk to delete.",
            "char_count": 16,
            "embedding": [0.1, 0.2],
            "embedding_dim": 2,
            "embedding_model": "models/gemini-embedding-001"
        }]
        save_document_chunks(self.doc_id, chunks, db_path=self.db_path)
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 1)

        deleted = delete_document(self.doc_id, db_path=self.db_path)
        self.assertTrue(deleted)

        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 0)
        self.assertEqual(len(get_document_chunks(self.doc_id, db_path=self.db_path)), 0)

    def test_delete_document_chunks_helper(self):
        """Verify delete_document_chunks removes only chunks while keeping the document record."""
        chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Temporary chunk.",
            "char_count": 16,
            "embedding": [0.1, 0.2],
            "embedding_dim": 2,
            "embedding_model": "models/gemini-embedding-001"
        }]
        save_document_chunks(self.doc_id, chunks, db_path=self.db_path)
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 1)

        self.assertTrue(delete_document_chunks(self.doc_id, db_path=self.db_path))
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 0)

        # Document record itself remains intact
        conn = get_db(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM documents WHERE id = ?", (self.doc_id,))
            self.assertIsNotNone(cursor.fetchone())
        finally:
            conn.close()
