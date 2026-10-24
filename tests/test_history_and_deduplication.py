"""
Tests for Phase 2, Step 3:
Document History Navigation and Duplicate-Upload Cache Reuse.

Verifies:
1. History links render for existing documents.
2. A valid history link reopens the correct document analysis.
3. Missing files and invalid stored filenames are handled safely.
4. Uploading the same valid PDF twice reuses the existing stored document and cached analysis.
5. Uploading a different PDF creates a distinct document.
6. Existing database records and legacy rows with NULL hashes remain usable.
7. Duplicate uploads do not trigger unnecessary summary or page-explanation LLM calls when valid cached results already exist.
"""

import io
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pypdf

from app import create_app
from config import Config
from database.db import get_db, init_db
from database.models import (
    get_all_documents,
    get_document_by_content_hash,
    get_document_by_filename,
    get_document_by_id,
    insert_document,
    save_document_summary,
    save_page_explanation,
)


class TestStep3Config(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True


class HistoryAndDeduplicationTestCase(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary folders for testing instance and uploads
        self.test_dir = tempfile.mkdtemp()
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(TestStep3Config):
            INSTANCE_FOLDER = self.test_instance_dir
            UPLOAD_FOLDER = self.test_upload_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        """Fetches a valid CSRF token from the application homepage."""
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def _create_minimal_pdf(self, text_content: str = "Test document content") -> bytes:
        """Helper to create minimal valid PDF bytes with selectable text."""
        writer = pypdf.PdfWriter()
        # Create a page and add text
        page = writer.add_blank_page(width=300, height=300)
        # Using raw PDF stream with known text
        escaped_text = text_content.replace("(", "").replace(")", "")
        stream_bytes = f"BT /F1 12 Tf 50 250 Td ({escaped_text}) Tj ET".encode("utf-8")
        
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)
        # Or write a complete well-formed PDF with text object:
        raw_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n"
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"5 0 obj << /Length " + str(len(stream_bytes)).encode("ascii") + b" >> stream\n"
            + stream_bytes + b"\nendstream\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000262 00000 n \n"
            b"0000000341 00000 n \n"
            b"trailer << /Root 1 0 R /Size 6 >>\n"
            b"startxref\n450\n%%EOF"
        )
        return raw_pdf

    # -------------------------------------------------------------------------
    # 1. History links render for existing documents
    # -------------------------------------------------------------------------
    def test_history_links_render_for_existing_documents(self):
        """Test that /history renders 'View Analysis' link for processed documents."""
        with self.app.app_context():
            doc1_id = insert_document(
                filename="doc1.pdf",
                original_filename="Paper One.pdf",
                page_count=5,
                character_count=1200,
                status="processed",
                content_hash="hash111"
            )
            doc2_id = insert_document(
                filename="doc2.pdf",
                original_filename="Failed Paper.pdf",
                page_count=0,
                character_count=0,
                status="failed",
                content_hash="hash222"
            )

        res = self.client.get("/history")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # Check table header
        self.assertIn("Action", html)

        # Check active View Analysis link for processed document
        self.assertIn(f"/analysis/{doc1_id}", html)
        self.assertIn("View Analysis", html)

        # Check failed document has disabled placeholder
        self.assertIn("status-disabled", html)

    # -------------------------------------------------------------------------
    # 2. A valid history link reopens the correct document analysis
    # -------------------------------------------------------------------------
    def test_valid_history_link_reopens_correct_document_analysis(self):
        """Test that clicking /analysis/<doc_id> reopens the analysis with cached summary and explanations."""
        pdf_bytes = self._create_minimal_pdf("Grounded Deep Learning Walkthrough")
        stored_name = "cached_history_walkthrough.pdf"
        file_path = os.path.join(self.test_upload_dir, stored_name)
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        with self.app.app_context():
            doc_id = insert_document(
                filename=stored_name,
                original_filename="Deep_Learning_Paper.pdf",
                page_count=1,
                character_count=50,
                status="processed",
                content_hash="dl_hash_123"
            )
            save_document_summary(
                filename=stored_name,
                summary="### Executive Overview\nNovel deep learning architecture.",
                model_used="gemini-flash-lite-latest"
            )
            save_page_explanation(
                filename=stored_name,
                page_number=1,
                explanation="Page 1 explains convolutional neural networks.",
                model_used="gemini-flash-lite-latest"
            )

        # Access via GET /analysis/<doc_id>
        res = self.client.get(f"/analysis/{doc_id}")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        self.assertIn("Deep_Learning_Paper.pdf", html)
        self.assertIn("Extraction Results", html)
        self.assertIn("Novel deep learning architecture.", html)
        self.assertIn("Page 1 explains convolutional neural networks.", html)
        self.assertIn("Cached in SQLite", html)

        # Also test via route alias /history/<doc_id>
        res_alias = self.client.get(f"/history/{doc_id}")
        self.assertEqual(res_alias.status_code, 200)
        self.assertIn("Deep_Learning_Paper.pdf", res_alias.data.decode("utf-8"))

    # -------------------------------------------------------------------------
    # 3. Missing files and invalid stored filenames are handled safely
    # -------------------------------------------------------------------------
    def test_missing_files_and_invalid_stored_filenames_handled_safely(self):
        """Test that missing DB records, deleted files, and path traversal are rejected safely."""
        # Case A: Document ID does not exist in database
        res_missing_id = self.client.get("/analysis/99999", follow_redirects=True)
        self.assertEqual(res_missing_id.status_code, 200)
        self.assertIn("Document record not found in history.", res_missing_id.data.decode("utf-8"))

        # Case B: Document exists in DB, but physical file is missing from disk
        with self.app.app_context():
            ghost_id = insert_document(
                filename="ghost_file.pdf",
                original_filename="ghost.pdf",
                page_count=2,
                character_count=200,
                status="processed",
                content_hash="ghost_hash"
            )
        # Ensure file does not exist on disk
        ghost_path = os.path.join(self.test_upload_dir, "ghost_file.pdf")
        if os.path.exists(ghost_path):
            os.remove(ghost_path)

        res_missing_file = self.client.get(f"/analysis/{ghost_id}", follow_redirects=True)
        self.assertEqual(res_missing_file.status_code, 200)
        self.assertIn("Document file not found on disk", res_missing_file.data.decode("utf-8"))

        # Case C: Path traversal in stored_filename lookup
        res_traversal = self.client.get("/analysis/..%2F..%2Fsecret.pdf", follow_redirects=True)
        self.assertEqual(res_traversal.status_code, 200)
        # Handled safely and redirected to history
        self.assertIn("Document History", res_traversal.data.decode("utf-8"))

    # -------------------------------------------------------------------------
    # 4. Uploading the same valid PDF twice reuses existing stored document & cached analysis
    # -------------------------------------------------------------------------
    def test_upload_same_valid_pdf_twice_reuses_existing_document(self):
        """Test that uploading the same PDF content twice reuses the stored document and does not duplicate files."""
        pdf_bytes = self._create_minimal_pdf("Duplicate Content Verification Text")

        data1 = {
            "pdf_file": (io.BytesIO(pdf_bytes), "original_upload.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res1 = self.client.post("/upload", data=data1, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 200)

        # Verify 1 document in DB and 1 file in upload directory
        with self.app.app_context():
            docs_initial = get_all_documents()
            self.assertEqual(len(docs_initial), 1)
            stored_name = docs_initial[0]["filename"]

            # Save a summary to cache for this document
            save_document_summary(
                filename=stored_name,
                summary="Executive overview for deduplication test.",
                model_used="gemini-flash-lite-latest"
            )

        saved_files_initial = os.listdir(self.test_upload_dir)
        self.assertEqual(len(saved_files_initial), 1)

        # Upload exact same PDF bytes again under a different upload filename
        data2 = {
            "pdf_file": (io.BytesIO(pdf_bytes), "copy_of_upload.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res2 = self.client.post("/upload", data=data2, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 200)
        html2 = res2.data.decode("utf-8")

        # Verify no second document was created in the database
        with self.app.app_context():
            docs_after = get_all_documents()
            self.assertEqual(len(docs_after), 1)
            self.assertEqual(docs_after[0]["filename"], stored_name)

        # Verify no duplicate file was written to disk
        saved_files_after = os.listdir(self.test_upload_dir)
        self.assertEqual(len(saved_files_after), 1)

        # Verify the cached summary was returned immediately on upload
        self.assertIn("Executive overview for deduplication test.", html2)
        self.assertIn("Cached in SQLite", html2)

    # -------------------------------------------------------------------------
    # 5. Uploading a different PDF creates a distinct document
    # -------------------------------------------------------------------------
    def test_upload_different_pdf_creates_distinct_document(self):
        """Test that uploading distinct PDF contents creates distinct database records and files."""
        pdf1_bytes = self._create_minimal_pdf("Content A for First PDF")
        pdf2_bytes = self._create_minimal_pdf("Content B for Completely Different Second PDF")

        data1 = {
            "pdf_file": (io.BytesIO(pdf1_bytes), "doc1.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res1 = self.client.post("/upload", data=data1, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 200)

        data2 = {
            "pdf_file": (io.BytesIO(pdf2_bytes), "doc2.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res2 = self.client.post("/upload", data=data2, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 200)

        with self.app.app_context():
            docs = get_all_documents()
            self.assertEqual(len(docs), 2)
            self.assertNotEqual(docs[0]["filename"], docs[1]["filename"])
            self.assertNotEqual(docs[0]["content_hash"], docs[1]["content_hash"])

        saved_files = os.listdir(self.test_upload_dir)
        self.assertEqual(len(saved_files), 2)

    # -------------------------------------------------------------------------
    # 6. Existing database records and legacy rows remain usable
    # -------------------------------------------------------------------------
    def test_legacy_database_rows_with_null_hash_remain_usable(self):
        """Test that legacy SQLite rows without content_hash (NULL) remain usable and do not crash."""
        pdf_bytes = self._create_minimal_pdf("Legacy Document Text")
        stored_name = "legacy_sample.pdf"
        file_path = os.path.join(self.test_upload_dir, stored_name)
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        # Directly insert a legacy row with NULL content_hash
        with self.app.app_context():
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, original_filename, page_count, character_count, status, content_hash)
                VALUES (?, ?, ?, ?, ?, NULL)
            """, (stored_name, "legacy_paper.pdf", 1, 20, "processed"))
            conn.commit()
            legacy_id = cursor.lastrowid

            # Verify retrieval
            legacy_doc = get_document_by_id(legacy_id)
            self.assertIsNotNone(legacy_doc)
            self.assertIsNone(legacy_doc["content_hash"])

            # Verify hash lookup for None or empty returns None
            self.assertIsNone(get_document_by_content_hash(None))
            self.assertIsNone(get_document_by_content_hash(""))

        # Verify history navigation works for legacy document
        res = self.client.get(f"/analysis/{legacy_id}")
        self.assertEqual(res.status_code, 200)
        self.assertIn("legacy_paper.pdf", res.data.decode("utf-8"))

        # Verify uploading a new document works alongside legacy rows
        new_pdf_bytes = self._create_minimal_pdf("New Non-legacy PDF")
        data = {
            "pdf_file": (io.BytesIO(new_pdf_bytes), "new_doc.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res_upload = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res_upload.status_code, 200)

        with self.app.app_context():
            docs = get_all_documents()
            self.assertEqual(len(docs), 2)

    # -------------------------------------------------------------------------
    # 7. Duplicate uploads do not trigger unnecessary LLM calls
    # -------------------------------------------------------------------------
    def test_duplicate_uploads_do_not_trigger_unnecessary_llm_calls(self):
        """Test that duplicate uploads reuse cached summary and page walkthrough without calling Gemini."""
        pdf_bytes = self._create_minimal_pdf("LLM Cache Call Verification")

        # Initial upload
        data = {
            "pdf_file": (io.BytesIO(pdf_bytes), "first_run.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        res1 = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 200)

        with self.app.app_context():
            doc = get_all_documents()[0]
            stored_name = doc["filename"]

            # Populate SQLite caches
            save_document_summary(
                filename=stored_name,
                summary="### Executive Summary\nAlready generated and cached in SQLite.",
                model_used="gemini-flash-lite-latest"
            )
            save_page_explanation(
                filename=stored_name,
                page_number=1,
                explanation="Page 1 already explained.",
                model_used="gemini-flash-lite-latest"
            )

        # Duplicate upload attempt with LLM calls mocked
        with patch("routes.pdf_routes.summarize_document") as mock_summarize, \
             patch("routes.pdf_routes.explain_page_content") as mock_explain:

            data_dup = {
                "pdf_file": (io.BytesIO(pdf_bytes), "duplicate_run.pdf"),
                "csrf_token": self.get_csrf_token()
            }
            res2 = self.client.post("/upload", data=data_dup, content_type="multipart/form-data")
            self.assertEqual(res2.status_code, 200)
            html2 = res2.data.decode("utf-8")

            # Assert no Gemini API calls were dispatched
            mock_summarize.assert_not_called()
            mock_explain.assert_not_called()

            # Assert cached content is present in output
            self.assertIn("Already generated and cached in SQLite.", html2)
            self.assertIn("Page 1 already explained.", html2)
            self.assertIn("Cached in SQLite", html2)


if __name__ == "__main__":
    unittest.main()
