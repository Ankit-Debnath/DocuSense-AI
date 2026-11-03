"""
Unit Tests for Document Indexing Service
Tests coordination of text extraction, semantic chunking, batched embedding generation,
atomic persistence in SQLite, failure-safety, and status reporting.
Zero external API calls are made; Gemini client is mocked.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pypdf
from database.db import init_db
from database.models import (
    insert_document,
    count_document_chunks,
    get_document_chunks
)
from ai.indexing_service import index_document, get_document_indexing_status


class DocumentIndexingServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.upload_dir = os.path.join(self.test_dir, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_indexing.db")
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_pdf_with_text(self, filepath: str):
        """Creates a minimal valid PDF file with actual text content."""
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length 44 >> stream
BT
/F1 24 Tf
100 700 Td
(Hello World from PDF) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000262 00000 n 
0000000341 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
434
%%EOF"""
        with open(filepath, "wb") as f:
            f.write(pdf_content)

    def _create_blank_pdf(self, filepath: str):
        """Creates a valid PDF with blank pages (no extractable text)."""
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(filepath, "wb") as f:
            writer.write(f)

    def _create_mock_client(self, dimension: int = 768):
        """Creates a mock Gemini client returning dummy vectors of specified dimension."""
        mock_client = MagicMock()

        def side_effect(model, contents, config):
            resp = MagicMock()
            items = []
            for _ in contents:
                item = MagicMock()
                item.values = [0.01 * (i + 1) for i in range(dimension)]
                items.append(item)
            resp.embeddings = items
            return resp

        mock_client.models.embed_content.side_effect = side_effect
        return mock_client

    def test_successful_document_indexing(self):
        """Verify successful end-to-end indexing of a valid PDF document."""
        pdf_path = os.path.join(self.upload_dir, "doc1.pdf")
        self._create_pdf_with_text(pdf_path)

        doc_id = insert_document(
            filename="doc1.pdf",
            original_filename="doc1_orig.pdf",
            page_count=1,
            character_count=21,
            status="processed",
            content_hash="hash1",
            db_path=self.db_path
        )

        mock_client = self._create_mock_client(dimension=768)

        result = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=mock_client,
            db_path=self.db_path
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["document_id"], doc_id)
        self.assertGreaterEqual(result["chunk_count"], 1)
        self.assertEqual(result["embedding_dim"], 768)
        self.assertIsNone(result["error"])

        # Check DB status
        status = get_document_indexing_status(doc_id, db_path=self.db_path)
        self.assertTrue(status["indexed"])
        self.assertEqual(status["chunk_count"], result["chunk_count"])

        # Verify stored chunk records
        chunks = get_document_chunks(doc_id, db_path=self.db_path)
        self.assertEqual(len(chunks), result["chunk_count"])
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("Hello World", chunks[0]["chunk_text"])
        self.assertEqual(chunks[0]["embedding_dim"], 768)

    def test_indexing_blank_pdf_fails_gracefully(self):
        """Verify blank PDF without extractable text produces no misleading index."""
        pdf_path = os.path.join(self.upload_dir, "blank.pdf")
        self._create_blank_pdf(pdf_path)

        doc_id = insert_document(
            filename="blank.pdf",
            original_filename="blank.pdf",
            page_count=1,
            character_count=0,
            status="processed",
            content_hash="hash_blank",
            db_path=self.db_path
        )

        mock_client = self._create_mock_client()
        result = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=mock_client,
            db_path=self.db_path
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["chunk_count"], 0)
        self.assertIn("no extractable text", result["error"].lower())

        # Status must report not indexed
        status = get_document_indexing_status(doc_id, db_path=self.db_path)
        self.assertFalse(status["indexed"])
        self.assertEqual(status["chunk_count"], 0)

    def test_reindexing_replaces_old_chunks_without_duplicates(self):
        """Verify reindexing replaces previous chunks cleanly and enforces atomic consistency."""
        pdf_path = os.path.join(self.upload_dir, "reindex.pdf")
        self._create_pdf_with_text(pdf_path)

        doc_id = insert_document(
            filename="reindex.pdf",
            original_filename="reindex.pdf",
            page_count=1,
            character_count=21,
            status="processed",
            content_hash="hash_reindex",
            db_path=self.db_path
        )

        mock_client = self._create_mock_client(dimension=768)

        # First indexing
        res1 = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=mock_client,
            db_path=self.db_path
        )
        self.assertTrue(res1["success"])
        initial_count = res1["chunk_count"]
        self.assertEqual(count_document_chunks(doc_id, db_path=self.db_path), initial_count)

        # Second indexing
        res2 = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=mock_client,
            db_path=self.db_path
        )
        self.assertTrue(res2["success"])
        self.assertEqual(res2["chunk_count"], initial_count)
        self.assertEqual(count_document_chunks(doc_id, db_path=self.db_path), initial_count)

    def test_failed_embedding_preserves_previous_valid_index(self):
        """Verify that when embedding generation fails, the existing valid index is preserved."""
        pdf_path = os.path.join(self.upload_dir, "safe_reindex.pdf")
        self._create_pdf_with_text(pdf_path)

        doc_id = insert_document(
            filename="safe_reindex.pdf",
            original_filename="safe_reindex.pdf",
            page_count=1,
            character_count=21,
            status="processed",
            content_hash="hash_safe",
            db_path=self.db_path
        )

        mock_client = self._create_mock_client(dimension=768)

        # Successfully index initially
        res1 = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=mock_client,
            db_path=self.db_path
        )
        self.assertTrue(res1["success"])
        initial_chunks = get_document_chunks(doc_id, db_path=self.db_path)
        self.assertGreater(len(initial_chunks), 0)

        # Reindexing fails due to API failure
        failing_client = MagicMock()
        failing_client.models.embed_content.side_effect = RuntimeError("Service unavailable 503")

        res2 = index_document(
            document_id=doc_id,
            file_path=pdf_path,
            client=failing_client,
            db_path=self.db_path
        )
        self.assertFalse(res2["success"])
        self.assertIn("failed", res2["error"].lower())

        # Critical check: initial chunks must remain completely intact!
        current_chunks = get_document_chunks(doc_id, db_path=self.db_path)
        self.assertEqual(len(current_chunks), len(initial_chunks))
        self.assertEqual(current_chunks[0]["chunk_text"], initial_chunks[0]["chunk_text"])

    def test_indexing_nonexistent_document_fails(self):
        """Verify trying to index a non-existent document ID fails cleanly."""
        res = index_document(
            document_id=88888,
            file_path="dummy.pdf",
            db_path=self.db_path
        )
        self.assertFalse(res["success"])
        self.assertIn("does not exist", res["error"].lower())

    def test_indexing_missing_file_on_disk_fails(self):
        """Verify trying to index when PDF file is missing on disk fails gracefully."""
        doc_id = insert_document(
            filename="missing_file.pdf",
            original_filename="missing_file.pdf",
            page_count=1,
            character_count=50,
            status="processed",
            content_hash="hash_missing",
            db_path=self.db_path
        )

        res = index_document(
            document_id=doc_id,
            file_path=os.path.join(self.upload_dir, "nonexistent.pdf"),
            db_path=self.db_path
        )
        self.assertFalse(res["success"])
        self.assertIn("not found on disk", res["error"].lower())
