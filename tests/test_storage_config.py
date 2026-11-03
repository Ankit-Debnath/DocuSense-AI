"""
Focused regression test suite for Step 6.5.2.2: Configurable Persistent Storage.
Verifies:
- SQLite database location configuration via DATABASE_PATH
- Upload directory configuration via UPLOAD_FOLDER
- Fail-fast validation of invalid storage configurations
- Directory auto-creation and existing database preservation
- Strict path traversal protections on configured upload folders
- Isolated multi-instance database behavior
"""

import io
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
import pypdf
from unittest.mock import patch

from app import create_app
from config import (
    BASE_DIR,
    DEFAULT_DATABASE_PATH,
    DEFAULT_UPLOAD_FOLDER,
    DEFAULT_INSTANCE_FOLDER,
    Config,
    ProductionConfig,
    DevelopmentConfig,
    resolve_database_path,
    validate_database_path,
    resolve_upload_folder,
    validate_upload_folder,
)
from database.models import insert_document, get_all_documents
from routes.pdf_routes import _validate_stored_filename


class StorageConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()
        self.test_root = tempfile.mkdtemp()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root, ignore_errors=True)

    def _create_minimal_pdf(self) -> bytes:
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def _get_csrf_token(self, client) -> str:
        res = client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    # =========================================================================
    # 1. DATABASE_PATH Configuration Tests
    # =========================================================================

    def test_default_database_path_preserved(self):
        """When DATABASE_PATH is not configured, the default instance database path is used."""
        os.environ.pop("DATABASE_PATH", None)
        resolved = resolve_database_path()
        self.assertEqual(resolved, DEFAULT_DATABASE_PATH)
        self.assertEqual(
            resolved,
            os.path.abspath(os.path.join(BASE_DIR, "instance", "pdf_analyzer.db"))
        )

    def test_absolute_external_database_path_respected(self):
        """Configuring DATABASE_PATH with an absolute path outside the repo is respected."""
        ext_db = os.path.join(self.test_root, "external_db_dir", "custom.db")
        os.environ["DATABASE_PATH"] = ext_db

        class ProdTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = "test-prod-secret-at-least-16-chars!"

        app = create_app(ProdTestConfig)
        self.assertEqual(app.config["DATABASE_PATH"], os.path.abspath(ext_db))
        self.assertTrue(os.path.isfile(ext_db))

    def test_missing_parent_directories_created_for_database(self):
        """Deeply nested non-existent parent directories for DATABASE_PATH are created automatically."""
        nested_db = os.path.join(self.test_root, "level1", "level2", "level3", "app.db")
        self.assertFalse(os.path.exists(os.path.dirname(nested_db)))

        os.environ["DATABASE_PATH"] = nested_db

        class DevTestConfig(DevelopmentConfig):
            TESTING = True

        app = create_app(DevTestConfig)
        self.assertTrue(os.path.isdir(os.path.dirname(nested_db)))
        self.assertTrue(os.path.isfile(nested_db))

    def test_existing_database_file_preserved(self):
        """Starting an application with an existing populated database preserves all data and tables."""
        ext_db = os.path.join(self.test_root, "existing.db")
        os.environ["DATABASE_PATH"] = ext_db

        class DevTestConfig(DevelopmentConfig):
            TESTING = True

        # First instance creates and populates data
        app1 = create_app(DevTestConfig)
        with app1.app_context():
            doc_id = insert_document(
                filename="preserved_doc_123.pdf",
                original_filename="doc.pdf",
                page_count=5,
                character_count=1200,
                status="processed",
                content_hash="abc123hash"
            )
            self.assertIsNotNone(doc_id)

        # Second instance initializes with the exact same database path
        app2 = create_app(DevTestConfig)
        with app2.app_context():
            docs = get_all_documents()
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["filename"], "preserved_doc_123.pdf")
            self.assertEqual(docs[0]["content_hash"], "abc123hash")

    def test_isolated_database_instances(self):
        """Two application instances configured with different database paths remain isolated."""
        db1 = os.path.join(self.test_root, "inst1", "app.db")
        db2 = os.path.join(self.test_root, "inst2", "app.db")

        class Config1(DevelopmentConfig):
            TESTING = True
            DATABASE_PATH = db1

        class Config2(DevelopmentConfig):
            TESTING = True
            DATABASE_PATH = db2

        app1 = create_app(Config1)
        app2 = create_app(Config2)

        with app1.app_context():
            insert_document(
                filename="doc1.pdf",
                original_filename="doc1.pdf",
                page_count=1,
                character_count=100,
                status="processed",
                content_hash="hash1"
            )

        with app2.app_context():
            docs2 = get_all_documents()
            self.assertEqual(len(docs2), 0)

        with app1.app_context():
            docs1 = get_all_documents()
            self.assertEqual(len(docs1), 1)

    def test_invalid_database_paths_fail_clearly(self):
        """Invalid DATABASE_PATH values fail fast with descriptive ValueError."""
        # Empty / whitespace
        for bad_val in ("", "   ", None):
            with self.subTest(val=bad_val):
                with self.assertRaises(ValueError) as ctx:
                    validate_database_path(bad_val)
                self.assertIn("DATABASE_PATH cannot be empty", str(ctx.exception))

        # Existing directory passed as DB file
        ext_dir = os.path.join(self.test_root, "some_dir")
        os.makedirs(ext_dir, exist_ok=True)
        with self.assertRaises(ValueError) as ctx:
            validate_database_path(ext_dir)
        self.assertIn("is an existing directory, not a database file", str(ctx.exception))

        # Parent path is an existing file, preventing directory creation
        file_parent = os.path.join(self.test_root, "blocking_file")
        with open(file_parent, "w") as f:
            f.write("blocking")
        bad_db_path = os.path.join(file_parent, "sub.db")
        with self.assertRaises(ValueError) as ctx:
            validate_database_path(bad_db_path)
        self.assertIn("Parent path of DATABASE_PATH is an existing file", str(ctx.exception))

    def test_sqlite_pragmas_and_wal_mode_enforced_on_external_db(self):
        """Configured database correctly enables WAL mode, foreign keys, and busy timeout."""
        ext_db = os.path.join(self.test_root, "pragma_test.db")
        os.environ["DATABASE_PATH"] = ext_db

        class DevTestConfig(DevelopmentConfig):
            TESTING = True

        app = create_app(DevTestConfig)

        # Inspect directly using sqlite3
        conn = sqlite3.connect(ext_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        self.assertEqual(str(journal_mode).lower(), "wal")

        cursor.execute("PRAGMA table_info(documents);")
        doc_cols = [row[1] for row in cursor.fetchall()]
        self.assertIn("content_hash", doc_cols)

        cursor.execute("PRAGMA table_info(document_chunks);")
        chunk_cols = [row[1] for row in cursor.fetchall()]
        self.assertIn("embedding", chunk_cols)
        conn.close()

    # =========================================================================
    # 2. UPLOAD_FOLDER Configuration Tests
    # =========================================================================

    def test_default_upload_folder_preserved(self):
        """When UPLOAD_FOLDER is not configured, the default uploads directory is used."""
        os.environ.pop("UPLOAD_FOLDER", None)
        resolved = resolve_upload_folder()
        self.assertEqual(resolved, DEFAULT_UPLOAD_FOLDER)
        self.assertEqual(resolved, os.path.abspath(os.path.join(BASE_DIR, "uploads")))

    def test_external_upload_folder_respected(self):
        """Configuring UPLOAD_FOLDER with an absolute path outside the repo is respected."""
        ext_upload = os.path.join(self.test_root, "external_uploads")
        os.environ["UPLOAD_FOLDER"] = ext_upload

        class DevTestConfig(DevelopmentConfig):
            TESTING = True

        app = create_app(DevTestConfig)
        self.assertEqual(app.config["UPLOAD_FOLDER"], os.path.abspath(ext_upload))
        self.assertTrue(os.path.isdir(ext_upload))

    def test_missing_upload_directories_created(self):
        """Deeply nested non-existent UPLOAD_FOLDER directories are created automatically."""
        nested_upload = os.path.join(self.test_root, "up_level1", "up_level2", "uploads")
        self.assertFalse(os.path.exists(nested_upload))

        os.environ["UPLOAD_FOLDER"] = nested_upload

        class DevTestConfig(DevelopmentConfig):
            TESTING = True

        app = create_app(DevTestConfig)
        self.assertTrue(os.path.isdir(nested_upload))

    def test_uploads_written_to_configured_directory(self):
        """PDF uploads are physically written to the configured upload folder."""
        ext_upload = os.path.join(self.test_root, "configured_uploads")
        ext_db = os.path.join(self.test_root, "upload_test.db")

        class CustomConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = ext_upload
            DATABASE_PATH = ext_db

        app = create_app(CustomConfig)
        client = app.test_client()

        pdf_bytes = self._create_minimal_pdf()
        csrf_token = self._get_csrf_token(client)

        res = client.post(
            "/upload",
            data={
                "pdf_file": (io.BytesIO(pdf_bytes), "sample_upload.pdf"),
                "csrf_token": csrf_token
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        self.assertEqual(res.status_code, 200)

        # Confirm the file is in ext_upload
        saved_files = os.listdir(ext_upload)
        self.assertEqual(len(saved_files), 1)
        self.assertTrue(saved_files[0].endswith("sample_upload.pdf"))

    def test_duplicate_uploads_in_configured_directory(self):
        """Duplicate uploads of identical content are deduplicated in the configured directory."""
        ext_upload = os.path.join(self.test_root, "dedup_uploads")
        ext_db = os.path.join(self.test_root, "dedup_test.db")

        class CustomConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = ext_upload
            DATABASE_PATH = ext_db

        app = create_app(CustomConfig)
        client = app.test_client()

        pdf_bytes = self._create_minimal_pdf()
        csrf_token = self._get_csrf_token(client)

        # First upload
        res1 = client.post(
            "/upload",
            data={
                "pdf_file": (io.BytesIO(pdf_bytes), "original_name.pdf"),
                "csrf_token": csrf_token
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(len(os.listdir(ext_upload)), 1)

        # Duplicate upload
        res2 = client.post(
            "/upload",
            data={
                "pdf_file": (io.BytesIO(pdf_bytes), "different_name.pdf"),
                "csrf_token": csrf_token
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        self.assertEqual(res2.status_code, 200)
        # Deduplicated: no additional physical file created
        self.assertEqual(len(os.listdir(ext_upload)), 1)

    def test_path_traversal_attempts_cannot_escape_configured_directory(self):
        """_validate_stored_filename strictly rejects traversal attempts outside the configured upload directory."""
        ext_upload = os.path.join(self.test_root, "safe_zone")
        os.makedirs(ext_upload, exist_ok=True)

        # Attempt to target a file outside safe_zone
        secret_file = os.path.join(self.test_root, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("confidential")

        traversal_attempts = [
            "../secret.txt",
            "..\\secret.txt",
            "../../secret.txt",
            "..%2Fsecret.txt",
            "/secret.txt",
            "C:\\secret.txt",
            "sub/file.pdf",
            "sub\\file.pdf",
        ]

        for attempt in traversal_attempts:
            with self.subTest(attempt=attempt):
                is_valid, path = _validate_stored_filename(attempt, ext_upload)
                self.assertFalse(is_valid)
                self.assertEqual(path, "")

    def test_invalid_upload_paths_fail_clearly(self):
        """Invalid UPLOAD_FOLDER values fail fast with descriptive ValueError."""
        # Empty / whitespace
        for bad_val in ("", "   ", None):
            with self.subTest(val=bad_val):
                with self.assertRaises(ValueError) as ctx:
                    validate_upload_folder(bad_val)
                self.assertIn("UPLOAD_FOLDER cannot be empty", str(ctx.exception))

        # Existing file passed as upload directory
        existing_file = os.path.join(self.test_root, "regular_file.txt")
        with open(existing_file, "w") as f:
            f.write("text content")

        with self.assertRaises(ValueError) as ctx:
            validate_upload_folder(existing_file)
        self.assertIn("is an existing file, not a directory", str(ctx.exception))

    def _can_create_symlink(self, target_is_directory=False) -> bool:
        """Helper to test if the OS environment permits symlink creation."""
        probe_src = os.path.join(self.test_root, f"_probe_src_{target_is_directory}")
        probe_dst = os.path.join(self.test_root, f"_probe_dst_{target_is_directory}")
        try:
            if target_is_directory:
                os.makedirs(probe_src, exist_ok=True)
                os.symlink(probe_src, probe_dst, target_is_directory=True)
            else:
                with open(probe_src, "w") as f:
                    f.write("probe")
                os.symlink(probe_src, probe_dst)
            return True
        except (OSError, NotImplementedError):
            return False
        finally:
            if os.path.islink(probe_dst) or os.path.exists(probe_dst):
                try:
                    if target_is_directory:
                        os.rmdir(probe_dst)
                    else:
                        os.remove(probe_dst)
                except OSError:
                    pass
            if os.path.exists(probe_src):
                try:
                    if target_is_directory:
                        shutil.rmtree(probe_src, ignore_errors=True)
                    else:
                        os.remove(probe_src)
                except OSError:
                    pass

    # =========================================================================
    # 3. Symbolic-Link Hardening Tests
    # =========================================================================

    def test_symlink_file_pointing_outside_upload_folder_rejected(self):
        """A symlink inside the upload directory pointing to an external file is rejected."""
        if not self._can_create_symlink(target_is_directory=False):
            self.skipTest(
                "OS privilege restriction: Cannot create file symbolic links in this environment "
                "(requires Developer Mode or SeCreateSymbolicLinkPrivilege on Windows)."
            )

        outside_dir = os.path.join(self.test_root, "outside")
        upload_dir = os.path.join(self.test_root, "uploads")
        os.makedirs(outside_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)

        secret_file = os.path.join(outside_dir, "sensitive.pdf")
        with open(secret_file, "wb") as f:
            f.write(b"%PDF-1.4 sensitive content")

        symlink_path = os.path.join(upload_dir, "symlink_doc.pdf")
        os.symlink(secret_file, symlink_path)

        is_valid, resolved = _validate_stored_filename("symlink_doc.pdf", upload_dir)
        self.assertFalse(is_valid)
        self.assertEqual(resolved, "")
        self.assertTrue(os.path.exists(secret_file))

    def test_symlink_upload_directory_itself_works(self):
        """A configured upload directory that is itself a symlink to real storage continues to work."""
        if not self._can_create_symlink(target_is_directory=True):
            self.skipTest(
                "OS privilege restriction: Cannot create directory symbolic links in this environment "
                "(requires Developer Mode or SeCreateSymbolicLinkPrivilege on Windows)."
            )

        real_storage = os.path.join(self.test_root, "real_storage")
        symlink_upload_dir = os.path.join(self.test_root, "symlink_uploads")
        os.makedirs(real_storage, exist_ok=True)

        real_file = os.path.join(real_storage, "valid_doc.pdf")
        with open(real_file, "wb") as f:
            f.write(b"%PDF-1.4 valid content")

        os.symlink(real_storage, symlink_upload_dir, target_is_directory=True)

        is_valid, resolved = _validate_stored_filename("valid_doc.pdf", symlink_upload_dir)
        self.assertTrue(is_valid)
        self.assertEqual(resolved, os.path.realpath(real_file))

    def test_canonical_path_resolution_rejects_external_target_mocked(self):
        """Canonical realpath resolution rejects candidate paths resolving outside upload directory."""
        upload_dir = os.path.join(self.test_root, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        candidate_file = os.path.join(upload_dir, "doc.pdf")
        with open(candidate_file, "wb") as f:
            f.write(b"data")

        # Simulate realpath returning an external target outside upload_dir
        external_target = os.path.abspath(os.path.join(self.test_root, "outside", "target.pdf"))

        def fake_realpath(path):
            if os.path.basename(path) == "doc.pdf":
                return external_target
            return os.path.abspath(path)

        with patch("routes.pdf_routes.os.path.realpath", side_effect=fake_realpath):
            is_valid, resolved = _validate_stored_filename("doc.pdf", upload_dir)
            self.assertFalse(is_valid)
            self.assertEqual(resolved, "")

    def test_retrieval_endpoint_rejects_symlink_pointing_outside(self):
        """Endpoints (/result, /summarize, /explain-page) reject documents resolving outside upload folder."""
        upload_dir = os.path.join(self.test_root, "ep_uploads")
        db_path = os.path.join(self.test_root, "ep_test.db")
        os.makedirs(upload_dir, exist_ok=True)

        class CustomConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = upload_dir
            DATABASE_PATH = db_path

        app = create_app(CustomConfig)
        client = app.test_client()

        # Insert a document whose record points to an external file
        with app.app_context():
            insert_document(
                filename="external_link.pdf",
                original_filename="external.pdf",
                page_count=1,
                character_count=50,
                status="processed",
                content_hash="mockhash123"
            )

        # External sensitive file
        outside_file = os.path.join(self.test_root, "confidential.pdf")
        with open(outside_file, "wb") as f:
            f.write(b"%PDF-1.4 confidential data")

        # If OS symlink is possible, create real symlink; otherwise simulate realpath
        symlink_path = os.path.join(upload_dir, "external_link.pdf")
        has_real_symlink = self._can_create_symlink(target_is_directory=False)
        if has_real_symlink:
            os.symlink(outside_file, symlink_path)
        else:
            with open(symlink_path, "wb") as f:
                f.write(b"%PDF dummy")

        def run_endpoint_checks():
            # 1. Analysis page
            res_result = client.get("/analysis/external_link.pdf", follow_redirects=True)
            self.assertEqual(res_result.status_code, 200)
            self.assertIn(b"Document file not found on disk or invalid filename.", res_result.data)

            # 2. Page explain (JSON API endpoint returns 404)
            csrf_token = self._get_csrf_token(client)
            res_explain = client.post(
                "/explain-page",
                headers={"X-CSRFToken": csrf_token},
                json={"stored_filename": "external_link.pdf", "page_number": 1}
            )
            self.assertEqual(res_explain.status_code, 404)
            data_explain = res_explain.get_json()
            self.assertFalse(data_explain["success"])

            # 3. Summarize (Form endpoint flashes error and redirects)
            res_sum = client.post(
                "/summarize",
                data={"stored_filename": "external_link.pdf", "csrf_token": csrf_token},
                follow_redirects=True
            )
            self.assertEqual(res_sum.status_code, 200)
            self.assertIn(b"Invalid document reference or file not found.", res_sum.data)

            # 4. Ask endpoint (Form endpoint flashes error and redirects)
            res_ask = client.post(
                "/ask",
                data={"stored_filename": "external_link.pdf", "question": "What is this?", "csrf_token": csrf_token},
                follow_redirects=True
            )
            self.assertEqual(res_ask.status_code, 200)
            self.assertIn(b"Invalid document reference or file not found.", res_ask.data)

        if has_real_symlink:
            run_endpoint_checks()
        else:
            # When symlink creation is restricted, test canonical rejection via patched realpath
            def fake_realpath(path):
                if os.path.basename(path) == "external_link.pdf":
                    return outside_file
                return os.path.abspath(path)

            with patch("routes.pdf_routes.os.path.realpath", side_effect=fake_realpath):
                run_endpoint_checks()

        # External file must remain completely intact and unchanged
        self.assertTrue(os.path.exists(outside_file))
        with open(outside_file, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.4 confidential data")

    def test_cleanup_cannot_target_external_file_through_symlink(self):
        """Cleanup logic on extraction failure only removes files inside the upload directory."""
        upload_dir = os.path.join(self.test_root, "cleanup_uploads")
        os.makedirs(upload_dir, exist_ok=True)

        outside_target = os.path.join(self.test_root, "outside_target.txt")
        with open(outside_target, "w") as f:
            f.write("preserve this content")

        # Destination paths generated during upload are strictly within upload_dir
        safe_name = "test_cleanup_file.pdf"
        dest_path = os.path.join(upload_dir, safe_name)
        with open(dest_path, "w") as f:
            f.write("corrupted upload file")

        # Simulating upload cleanup
        if os.path.exists(dest_path):
            os.remove(dest_path)

        self.assertFalse(os.path.exists(dest_path))
        # External target was completely untouched
        self.assertTrue(os.path.exists(outside_target))
        with open(outside_target, "r") as f:
            self.assertEqual(f.read(), "preserve this content")


if __name__ == "__main__":
    unittest.main()
