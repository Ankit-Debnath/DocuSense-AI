import re
import io
import os
import shutil
import sqlite3
import tempfile
import threading
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


class SQLiteConnectionAndTransactionSafetyTestCase(unittest.TestCase):
    """
    Focused regression test suite for Phase 4, Step 3B.1:
    - Connection timeout (15s) and PRAGMA busy_timeout (15000ms)
    - Write transaction commit on success
    - Write transaction rollback on error
    - Request-scoped connection usability after failed write
    - Direct connection closure on success
    - Direct connection closure on exception
    """

    def setUp(self):
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

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_configured_connection_and_busy_timeout(self):
        """Verify that get_db sets PRAGMA busy_timeout = 15000 consistently for request and direct connections."""
        # 1. Direct connection
        conn_direct = get_db(self.test_db_path)
        cur = conn_direct.execute("PRAGMA busy_timeout;")
        timeout_val = cur.fetchone()[0]
        conn_direct.close()
        self.assertEqual(timeout_val, 15000)

        # 2. Request-scoped connection
        with self.app.app_context():
            conn_app = get_db()
            cur = conn_app.execute("PRAGMA busy_timeout;")
            timeout_app_val = cur.fetchone()[0]
            self.assertEqual(timeout_app_val, 15000)

    def test_successful_write_persists_and_commits(self):
        """Verify that successful write transactions commit and are visible from a separate connection."""
        with self.app.app_context():
            doc_id = insert_document(
                filename="persisted_doc.pdf",
                original_filename="original.pdf",
                page_count=5,
                character_count=1200,
                status="processed",
                content_hash="abc123hash"
            )
            self.assertGreater(doc_id, 0)

        # Open completely separate independent connection to verify commit
        separate_conn = sqlite3.connect(self.test_db_path)
        separate_conn.row_factory = sqlite3.Row
        cur = separate_conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        separate_conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["filename"], "persisted_doc.pdf")
        self.assertEqual(row["content_hash"], "abc123hash")

    def test_failed_write_rolls_back_and_leaves_no_partial_changes(self):
        """Verify that an exception during a write operation rolls back changes without persisting partial data."""
        with self.app.app_context():
            initial_docs = get_all_documents()
            initial_count = len(initial_docs)

            # Attempt insert_document with invalid column data (None for NOT NULL page_count)
            with self.assertRaises(sqlite3.IntegrityError):
                insert_document(
                    filename="bad_doc.pdf",
                    original_filename="bad.pdf",
                    page_count=None,  # NOT NULL constraint violation
                    character_count=100,
                    status="processed"
                )

            # Verify no partial or corrupt row was inserted
            current_docs = get_all_documents()
            self.assertEqual(len(current_docs), initial_count)

    def test_request_scoped_connection_remains_usable_after_failed_write(self):
        """Verify that after a failed write, g.db is rolled back cleanly and subsequent writes succeed."""
        with self.app.app_context():
            # 1. Trigger IntegrityError on request-scoped connection
            with self.assertRaises(sqlite3.IntegrityError):
                insert_document(
                    filename="failing_doc.pdf",
                    original_filename="failing.pdf",
                    page_count=None,  # Violates NOT NULL constraint
                    character_count=100,
                    status="processed"
                )

            # 2. Immediately perform a valid write on the SAME request-scoped connection
            doc_id = insert_document(
                filename="subsequent_valid.pdf",
                original_filename="valid.pdf",
                page_count=2,
                character_count=500,
                status="processed"
            )
            self.assertGreater(doc_id, 0)

            # 3. Retrieve and verify the subsequent document was saved
            doc = get_document_by_id(doc_id)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["filename"], "subsequent_valid.pdf")

    def test_direct_connection_is_closed_on_success(self):
        """Verify that functions called with an explicit db_path close the connection upon success."""
        doc_id = insert_document(
            filename="direct_doc.pdf",
            original_filename="direct.pdf",
            page_count=1,
            character_count=200,
            status="processed",
            db_path=self.test_db_path
        )
        self.assertGreater(doc_id, 0)

        # On Windows, open SQLite connections lock the file. If closed, renaming the file succeeds.
        test_copy_path = os.path.join(self.test_instance_dir, "test_rename.db")
        os.rename(self.test_db_path, test_copy_path)
        os.rename(test_copy_path, self.test_db_path)

    def test_direct_connection_is_closed_on_exception(self):
        """Verify that functions called with an explicit db_path close the connection upon exception."""
        with self.assertRaises(sqlite3.IntegrityError):
            insert_document(
                filename="failing_direct.pdf",
                original_filename="failing.pdf",
                page_count=None,  # NOT NULL constraint violation
                character_count=200,
                status="processed",
                db_path=self.test_db_path
            )

        # On Windows, if connection leaked, renaming or removing the file will raise PermissionError
        test_copy_path = os.path.join(self.test_instance_dir, "test_rename_err.db")
        os.rename(self.test_db_path, test_copy_path)
        os.rename(test_copy_path, self.test_db_path)

    def test_cache_helpers_with_direct_connection_close_on_success_and_error(self):
        """Verify summary and page-explanation cache helpers close direct connections on success and errors."""
        # 1. Summary save and get
        self.assertTrue(save_document_summary("cached.pdf", "Summary text", "test-model", db_path=self.test_db_path))
        row = get_document_summary("cached.pdf", db_path=self.test_db_path)
        self.assertIsNotNone(row)
        self.assertEqual(row["summary"], "Summary text")

        # 2. Page explanation save, get, and delete
        self.assertTrue(save_page_explanation("cached.pdf", 1, "Page 1 exp", "test-model", db_path=self.test_db_path))
        exp_row = get_page_explanation("cached.pdf", 1, db_path=self.test_db_path)
        self.assertIsNotNone(exp_row)
        self.assertTrue(delete_page_explanation("cached.pdf", 1, db_path=self.test_db_path))
        self.assertTrue(delete_document_summary("cached.pdf", db_path=self.test_db_path))

        # 3. Verify file is not locked on disk
        test_copy_path = os.path.join(self.test_instance_dir, "test_cache_rename.db")
        os.rename(self.test_db_path, test_copy_path)
        os.rename(test_copy_path, self.test_db_path)


class SQLiteWALModeTestCase(unittest.TestCase):
    """
    Focused regression test suite for Phase 4, Step 3B.2B:
    SQLite Write-Ahead Logging (WAL) Mode:
    - Test A: WAL mode persists across independent connections
    - Test B: Reader concurrency while an uncommitted writer transaction is active
    - Test C: Windows temporary-directory cleanup safety
    """

    def setUp(self):
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

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_wal_mode_persists_across_connections(self):
        """Test A: Verify that WAL mode is enabled by init_db and persists across separate connections."""
        fresh_dir = tempfile.mkdtemp()
        fresh_db_path = os.path.join(fresh_dir, "wal_persist.db")
        try:
            init_db(fresh_db_path)

            # Open an independent connection without setting any PRAGMA
            conn = sqlite3.connect(fresh_db_path, timeout=15.0)
            try:
                cur = conn.execute("PRAGMA journal_mode;")
                mode = cur.fetchone()[0]
                self.assertEqual(mode.lower(), "wal")
            finally:
                conn.close()
        finally:
            if os.path.exists(fresh_dir):
                shutil.rmtree(fresh_dir)

    def test_reader_can_read_while_writer_transaction_is_open(self):
        """Test B: Verify that in WAL mode, a concurrent reader can read committed data while a writer transaction is open."""
        # 1. Seed committed test document
        with self.app.app_context():
            doc_id = insert_document(
                filename="wal_seed_doc.pdf",
                original_filename="seed.pdf",
                page_count=2,
                character_count=400,
                status="processed"
            )
            self.assertGreater(doc_id, 0)

        # Synchronization events
        writer_started = threading.Event()
        writer_can_commit = threading.Event()
        writer_errors = []

        # 2. Worker thread performs a write and keeps transaction open
        def writer_worker():
            writer_conn = None
            try:
                writer_conn = sqlite3.connect(self.test_db_path, timeout=15.0)
                writer_conn.execute("PRAGMA busy_timeout = 15000;")
                writer_conn.execute("BEGIN IMMEDIATE;")
                writer_conn.execute(
                    "INSERT INTO documents (filename, original_filename, page_count, character_count, status) "
                    "VALUES (?, ?, ?, ?, ?);",
                    ("writer_in_flight.pdf", "in_flight.pdf", 1, 10, "pending")
                )
                # Signal main thread that writer has uncommitted write in progress
                writer_started.set()

                # Wait for reader to perform its query
                if not writer_can_commit.wait(timeout=5.0):
                    raise TimeoutError("Timed out waiting for reader verification signal")
                writer_conn.commit()
            except Exception as exc:
                writer_errors.append(exc)
            finally:
                if writer_conn:
                    writer_conn.close()

        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()

        reader_conn = None
        try:
            # Wait for writer to establish its active write transaction
            self.assertTrue(writer_started.wait(timeout=5.0), "Writer thread did not start in time")

            # 3. Reader query on separate connection runs while writer transaction is open
            reader_conn = sqlite3.connect(self.test_db_path, timeout=15.0)
            reader_conn.row_factory = sqlite3.Row
            reader_conn.execute("PRAGMA busy_timeout = 15000;")

            # Reader reads committed record without blocking or encountering database is locked
            cur = reader_conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["filename"], "wal_seed_doc.pdf")

            # Reader cannot see uncommitted row from the active writer transaction
            cur_uncommitted = reader_conn.execute(
                "SELECT * FROM documents WHERE filename = ?", ("writer_in_flight.pdf",)
            )
            self.assertIsNone(cur_uncommitted.fetchone())

        finally:
            # 4. Release writer and ensure both connections close cleanly
            writer_can_commit.set()
            writer_thread.join(timeout=5.0)
            if reader_conn:
                reader_conn.close()

        self.assertEqual(writer_errors, [], f"Writer thread failed: {writer_errors}")

    def test_windows_temporary_directory_cleanup(self):
        """Test C: Verify that WAL database and sidecar files clean up without PermissionError on Windows."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "wal_cleanup.db")
        try:
            # Initialize WAL database and write records
            init_db(db_path)
            doc_id = insert_document(
                filename="cleanup_test.pdf",
                original_filename="cleanup.pdf",
                page_count=1,
                character_count=100,
                status="processed",
                db_path=db_path
            )
            doc = get_document_by_id(doc_id, db_path=db_path)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["filename"], "cleanup_test.pdf")

            # Save and retrieve cache entries
            self.assertTrue(save_document_summary("cleanup_test.pdf", "Summary", "model", db_path=db_path))
            summary_row = get_document_summary("cleanup_test.pdf", db_path=db_path)
            self.assertIsNotNone(summary_row)
        finally:
            # All connections are closed by models finally blocks; rmtree must succeed cleanly
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        self.assertFalse(os.path.exists(temp_dir))


if __name__ == "__main__":
    unittest.main()
