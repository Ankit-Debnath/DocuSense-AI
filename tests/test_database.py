import re
import io
import os
import shutil
import sqlite3
import tempfile
import unittest
import pypdf
from app import create_app
from config import Config
from database.db import init_db, get_db
from database.models import (
    insert_document,
    get_all_documents,
    get_document_by_id,
    save_document_summary,
    get_document_summary,
    delete_document_summary,
    save_page_explanation,
    get_page_explanation,
    get_document_page_explanations,
    delete_page_explanation
)


class TestDBConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        # Create dedicated temporary folders for testing instance and uploads
        self.test_dir = tempfile.mkdtemp()
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(TestDBConfig):
            INSTANCE_FOLDER = self.test_instance_dir
            UPLOAD_FOLDER = self.test_upload_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def get_csrf_token(self):
        """Fetches a valid CSRF token from the application homepage."""
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_database_initialization(self):
        """Test that init_db creates the instance directory and documents table with correct schema."""
        self.assertTrue(os.path.exists(self.test_db_path))

        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Query SQLite table schema
        cursor.execute("PRAGMA table_info(documents);")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()

        # Assert all required columns exist
        expected_columns = ["id", "filename", "original_filename", "upload_time", "page_count", "character_count", "status"]
        for col in expected_columns:
            self.assertIn(col, columns, f"Column '{col}' was not found in documents table.")

    def test_insert_and_retrieve_document(self):
        """Test inserting a document record and retrieving it by ID and all documents."""
        with self.app.app_context():
            doc_id = insert_document(
                filename="abc123_annual_report.pdf",
                original_filename="annual_report.pdf",
                page_count=12,
                character_count=4500,
                status="processed"
            )
            self.assertIsInstance(doc_id, int)
            self.assertGreater(doc_id, 0)

            # Retrieve by ID
            doc = get_document_by_id(doc_id)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["filename"], "abc123_annual_report.pdf")
            self.assertEqual(doc["original_filename"], "annual_report.pdf")
            self.assertEqual(doc["page_count"], 12)
            self.assertEqual(doc["character_count"], 4500)
            self.assertEqual(doc["status"], "processed")
            self.assertIsNotNone(doc["upload_time"])

    def test_get_all_documents_ordering(self):
        """Test that documents are returned newest first."""
        with self.app.app_context():
            id1 = insert_document("first.pdf", "first.pdf", 1, 100, "processed")
            id2 = insert_document("second.pdf", "second.pdf", 2, 200, "processed")
            id3 = insert_document("third.pdf", "third.pdf", 3, 300, "processed")

            docs = get_all_documents()
            self.assertEqual(len(docs), 3)
            # Newest inserted should be first
            self.assertEqual(docs[0]["id"], id3)
            self.assertEqual(docs[1]["id"], id2)
            self.assertEqual(docs[2]["id"], id1)

    def test_parameterized_query_handles_special_characters(self):
        """Test that quotes and SQL injection characters are handled safely without syntax errors."""
        malicious_filename = "test'; DROP TABLE documents; --.pdf"
        with self.app.app_context():
            doc_id = insert_document(
                filename="safe.pdf",
                original_filename=malicious_filename,
                page_count=5,
                character_count=250,
                status="processed"
            )
            doc = get_document_by_id(doc_id)
            self.assertEqual(doc["original_filename"], malicious_filename)

            # Verify table still exists
            docs = get_all_documents()
            self.assertEqual(len(docs), 1)

    def test_history_page_renders_empty_state(self):
        """Test visiting /history when no documents have been uploaded."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No Documents Uploaded Yet", response.data)
        self.assertIn(b"Document History", response.data)

    def test_upload_persists_to_database_and_appears_in_history(self):
        """Test that uploading a valid PDF inserts a database record visible in /history."""
        # Create a valid test PDF
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)

        data = {
            "pdf_file": (io.BytesIO(pdf_bytes.getvalue()), "history_test.pdf"),
            "csrf_token": self.get_csrf_token()
        }

        # 1. Upload
        upload_res = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(upload_res.status_code, 200)

        # 2. Check history page
        history_res = self.client.get("/history")
        self.assertEqual(history_res.status_code, 200)
        self.assertIn(b"history_test.pdf", history_res.data)
        self.assertIn(b"processed", history_res.data)

    def test_document_summary_cache_operations(self):
        """Test creating, reading, updating, and deleting cached document summaries."""
        with self.app.app_context():
            # 1. Retrieve non-existent summary
            self.assertIsNone(get_document_summary("missing_doc.pdf"))

            # 2. Save summary
            save_document_summary("cached_doc.pdf", "Executive summary content.", "gemini-flash-lite-latest")
            row = get_document_summary("cached_doc.pdf")
            self.assertIsNotNone(row)
            self.assertEqual(row["filename"], "cached_doc.pdf")
            self.assertEqual(row["summary"], "Executive summary content.")
            self.assertEqual(row["model_used"], "gemini-flash-lite-latest")

            # 3. Update summary (upsert on conflict)
            save_document_summary("cached_doc.pdf", "Updated summary content.", "gemini-flash-lite-latest")
            updated_row = get_document_summary("cached_doc.pdf")
            self.assertEqual(updated_row["summary"], "Updated summary content.")

            # 4. Delete summary
            delete_document_summary("cached_doc.pdf")
            self.assertIsNone(get_document_summary("cached_doc.pdf"))

    def test_page_explanation_cache_operations(self):
        """Test creating, reading, querying all, updating, and deleting page-by-page technical walkthroughs."""
        with self.app.app_context():
            # 1. Retrieve non-existent page explanation
            self.assertIsNone(get_page_explanation("doc.pdf", 1))
            self.assertEqual(get_document_page_explanations("doc.pdf"), {})

            # 2. Save page explanation
            success = save_page_explanation(
                filename="doc.pdf",
                page_number=1,
                explanation="# Page 1 — Abstract\nDetailed walkthrough.",
                model_used="gemini-flash-lite-latest"
            )
            self.assertTrue(success)

            # 3. Retrieve single page
            row = get_page_explanation("doc.pdf", 1)
            self.assertIsNotNone(row)
            self.assertEqual(row["filename"], "doc.pdf")
            self.assertEqual(row["page_number"], 1)
            self.assertIn("Detailed walkthrough", row["explanation"])
            self.assertEqual(row["model_used"], "gemini-flash-lite-latest")

            # 4. Save second page and retrieve all
            save_page_explanation("doc.pdf", 2, "# Page 2 — Introduction", "gemini-flash-lite-latest")
            all_pages = get_document_page_explanations("doc.pdf")
            self.assertEqual(len(all_pages), 2)
            self.assertIn(1, all_pages)
            self.assertIn(2, all_pages)
            self.assertIn("Abstract", all_pages[1]["explanation"])
            self.assertIn("Introduction", all_pages[2]["explanation"])

            # 5. Update page explanation (upsert on conflict)
            save_page_explanation("doc.pdf", 1, "# Page 1 — Updated Abstract", "gemini-flash-lite-latest")
            updated = get_page_explanation("doc.pdf", 1)
            self.assertIn("Updated Abstract", updated["explanation"])

            # 6. Delete page explanation
            delete_page_explanation("doc.pdf", 1)
            self.assertIsNone(get_page_explanation("doc.pdf", 1))
            self.assertIsNotNone(get_page_explanation("doc.pdf", 2))


if __name__ == "__main__":
    unittest.main()
