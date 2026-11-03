"""
Focused regression test suite for Phase 4, Step 2B:
Session-Cookie and PDF Upload Reliability:
- Task 1: Q&A session-cookie bounds, large answer mitigation, and cross-document pruning
- Task 2: Maximum PDF page limit (MAX_PDF_PAGES) and extracted character ceiling (MAX_EXTRACTED_CHARACTERS)
- Task 3: Cleanup of newly uploaded PDF on database insertion failure while preserving duplicate files
"""

import io
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pypdf
from app import create_app
from config import Config
from pdf_processing.extractor import extract_text_from_pdf, DEFAULT_MAX_PDF_PAGES, DEFAULT_MAX_EXTRACTED_CHARACTERS


class Step2BTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    MAX_PDF_PAGES = 5
    MAX_EXTRACTED_CHARACTERS = 1000
    MAX_QA_HISTORY_TURNS = 5
    MAX_QA_QUESTION_CHARS = 120
    MAX_QA_ANSWER_CHARS = 140


def create_minimal_pdf_bytes(num_pages: int = 1, page_text_prefix: str = "Test page") -> bytes:
    """Helper creating valid in-memory PDF binary with specified page count."""
    writer = pypdf.PdfWriter()
    for i in range(1, num_pages + 1):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class SessionCookieReliabilityTestCase(unittest.TestCase):
    """Tests for Task 1: Q&A session-cookie bounds and overflow prevention."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(Step2BTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_large_model_answers_do_not_overflow_session_cookie(self):
        """Even if Gemini returns 2000-char answers over 5 turns, session cookie remains strictly bounded."""
        pdf_bytes = create_minimal_pdf_bytes(1)
        pdf_path = os.path.join(self.test_upload_dir, "cookie_test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        token = self.get_csrf_token()
        huge_answer = "This is a detailed analysis of neural mechanism with equations and tables. " * 30  # ~2,200 chars

        with patch("routes.pdf_routes.answer_pdf_question") as mock_ask:
            mock_ask.return_value = {
                "success": True,
                "answer": huge_answer,
                "page_citations": [1],
                "model_used": "gemini-flash-lite-latest"
            }

            for turn in range(1, 6):
                res = self.client.post("/ask", data={
                    "stored_filename": "cookie_test.pdf",
                    "original_filename": "cookie_test.pdf",
                    "question": f"Question {turn}: Please explain section {turn} in thorough detail?",
                    "csrf_token": token
                })
                self.assertEqual(res.status_code, 200)

                # Assert that Set-Cookie header remains strictly bounded below 1024 bytes on every turn
                for header in res.headers.getlist("Set-Cookie"):
                    if header.startswith("session="):
                        self.assertLess(
                            len(header),
                            1024,
                            f"Turn {turn} Set-Cookie header length {len(header)} exceeded 1024 bytes"
                        )

            # Inspect session payload directly
            with self.client.session_transaction() as sess:
                history = sess.get("qa_history_cookie_test.pdf", [])
                self.assertEqual(len(history), 5)
                for item in history:
                    # Stored answer is compact and does not store the full 2,200-character blob
                    self.assertLessEqual(len(item["answer"]), 140)
                    self.assertLessEqual(len(item["question"]), 120)

    def test_failed_requests_do_not_persist_failed_turns_in_session(self):
        """If Gemini inference fails, the failed turn must not be stored in qa_history."""
        pdf_bytes = create_minimal_pdf_bytes(1)
        pdf_path = os.path.join(self.test_upload_dir, "fail_qa.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        token = self.get_csrf_token()

        with patch("routes.pdf_routes.answer_pdf_question") as mock_ask:
            mock_ask.return_value = {
                "success": False,
                "answer": "",
                "error": "Google Gemini service is temporarily unavailable after retrying.",
                "page_citations": [],
                "model_used": "gemini-flash-lite-latest"
            }

            res = self.client.post("/ask", data={
                "stored_filename": "fail_qa.pdf",
                "original_filename": "fail_qa.pdf",
                "question": "What is the conclusion?",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"temporarily unavailable", res.data)

            with self.client.session_transaction() as sess:
                history = sess.get("qa_history_fail_qa.pdf", [])
                self.assertEqual(len(history), 0)

    def test_cross_document_history_pruning_prevents_cookie_accumulation(self):
        """Interacting with a new document prunes previous document history keys from the session."""
        pdf_bytes = create_minimal_pdf_bytes(1)
        doc1_path = os.path.join(self.test_upload_dir, "doc1.pdf")
        doc2_path = os.path.join(self.test_upload_dir, "doc2.pdf")
        with open(doc1_path, "wb") as f:
            f.write(pdf_bytes)
        with open(doc2_path, "wb") as f:
            f.write(pdf_bytes)

        token = self.get_csrf_token()

        with patch("routes.pdf_routes.answer_pdf_question") as mock_ask:
            mock_ask.return_value = {
                "success": True,
                "answer": "Answer for doc1",
                "page_citations": [1],
                "model_used": "gemini-flash-lite-latest"
            }
            # Ask on doc1
            self.client.post("/ask", data={
                "stored_filename": "doc1.pdf",
                "original_filename": "doc1.pdf",
                "question": "Question about doc1?",
                "csrf_token": token
            })

            with self.client.session_transaction() as sess:
                self.assertIn("qa_history_doc1.pdf", sess)

            # Now ask on doc2
            mock_ask.return_value = {
                "success": True,
                "answer": "Answer for doc2",
                "page_citations": [1],
                "model_used": "gemini-flash-lite-latest"
            }
            self.client.post("/ask", data={
                "stored_filename": "doc2.pdf",
                "original_filename": "doc2.pdf",
                "question": "Question about doc2?",
                "csrf_token": token
            })

            # Verify doc2 history is active and old doc1 history was pruned from the session cookie
            with self.client.session_transaction() as sess:
                self.assertIn("qa_history_doc2.pdf", sess)
                self.assertNotIn("qa_history_doc1.pdf", sess)


class PDFPageAndCharacterLimitsTestCase(unittest.TestCase):
    """Tests for Task 2: Bounded PDF page count and text accumulation limits."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(Step2BTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_pdf_within_page_limit_extracts_successfully(self):
        """A PDF with pages <= MAX_PDF_PAGES extracts cleanly."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=3)
        pdf_path = os.path.join(self.test_dir, "valid_3pages.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        result = extract_text_from_pdf(pdf_path, max_pages=5)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_pages"], 3)
        self.assertEqual(len(result["pages"]), 3)
        self.assertIsNone(result["error"])

    def test_pdf_exceeding_page_limit_fails_fast_without_per_page_extraction(self):
        """A PDF exceeding MAX_PDF_PAGES fails immediately with an informative error."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=6)
        pdf_path = os.path.join(self.test_dir, "oversized_6pages.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Configured max_pages is 5
        result = extract_text_from_pdf(pdf_path, max_pages=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["total_pages"], 6)
        self.assertEqual(len(result["pages"]), 0)
        self.assertIn("exceeds the maximum allowed limit of 5 pages", result["error"])
        self.assertIn("found 6 pages", result["error"])

    def test_pdf_exceeding_extracted_character_limit_aborts_cleanly(self):
        """A PDF that generates text beyond MAX_EXTRACTED_CHARACTERS aborts extraction safely."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=2)
        pdf_path = os.path.join(self.test_dir, "dense_text.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Mock page.extract_text to return 600 characters per page (total 1200 > limit 1000)
        with patch("pypdf.PageObject.extract_text", return_value="A" * 600):
            result = extract_text_from_pdf(pdf_path, max_pages=10, max_characters=1000)

        self.assertFalse(result["success"])
        self.assertEqual(len(result["pages"]), 0)
        self.assertIn("exceeds the maximum allowed extractable text limit", result["error"])

    def test_upload_route_rejects_oversized_pdf_and_removes_uploaded_file(self):
        """Uploading a PDF that exceeds MAX_PDF_PAGES flashes error and cleans up newly saved file."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=10)  # Config has MAX_PDF_PAGES = 5
        token = self.get_csrf_token()

        res = self.client.post("/upload", data={
            "pdf_file": (io.BytesIO(pdf_bytes), "oversized.pdf"),
            "csrf_token": token
        }, follow_redirects=True)

        self.assertEqual(res.status_code, 200)
        self.assertIn(b"exceeds the maximum allowed limit of 5 pages", res.data)

        # Verify no orphaned file remained in uploads/
        uploaded_files = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
        self.assertEqual(len(uploaded_files), 0)


class DatabaseInsertionCleanupTestCase(unittest.TestCase):
    """Tests for Task 3: File cleanup on database insertion failure while preserving deduplicated files."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(Step2BTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_database_insertion_failure_removes_newly_saved_file(self):
        """If insert_document() fails, the newly created upload file is deleted from disk."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        with patch("routes.pdf_routes.insert_document", side_effect=Exception("Database lock timeout")):
            res = self.client.post("/upload", data={
                "pdf_file": (io.BytesIO(pdf_bytes), "failed_insert.pdf"),
                "csrf_token": token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            self.assertIn(b"database error occurred while registering the document", res.data)

            # Assert disk file was cleaned up and not orphaned
            uploaded_files = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
            self.assertEqual(len(uploaded_files), 0)

    def test_cleanup_failure_does_not_mask_original_database_error(self):
        """If removing the file raises OSError, the original error flash/redirect is still returned."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        with patch("routes.pdf_routes.insert_document", side_effect=Exception("Database lock timeout")), \
             patch("os.remove", side_effect=OSError("Permission denied on delete")):
            res = self.client.post("/upload", data={
                "pdf_file": (io.BytesIO(pdf_bytes), "failed_cleanup.pdf"),
                "csrf_token": token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            self.assertIn(b"database error occurred while registering the document", res.data)

    def test_duplicate_upload_detection_never_deletes_existing_file(self):
        """Uploading an exact duplicate reuses the existing file and never deletes it."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        # Upload 1: Successful initial upload
        res1 = self.client.post("/upload", data={
            "pdf_file": (io.BytesIO(pdf_bytes), "original_file.pdf"),
            "csrf_token": token
        })
        self.assertEqual(res1.status_code, 200)
        uploaded_files_before = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
        self.assertEqual(len(uploaded_files_before), 1)
        saved_filename = uploaded_files_before[0]

        # Upload 2: Exact duplicate
        res2 = self.client.post("/upload", data={
            "pdf_file": (io.BytesIO(pdf_bytes), "duplicate_file.pdf"),
            "csrf_token": token
        })
        self.assertEqual(res2.status_code, 200)
        self.assertIn(saved_filename.encode("utf-8"), res2.data)

        # Verify the original file still exists and was not deleted or duplicated
        uploaded_files_after = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
        self.assertEqual(len(uploaded_files_after), 1)
        self.assertEqual(uploaded_files_after[0], saved_filename)


class UploadFailureCleanupHardeningTestCase(unittest.TestCase):
    """
    Focused regression test suite for Phase 4, Step 2B.2:
    Upload failure cleanup hardening (F-13 and F-14):
    - File-save failure handling, partial file cleanup, and structured error logging
    - Partial-file cleanup failure dual logging and controlled response
    - Extraction failure cleanup logging on OSError (replacing silent pass)
    - Regression protection for successful uploads
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(Step2BTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_file_save_failure_cleans_up_partial_file_and_logs_error(self):
        """Simulate file.save() raising OSError; verify 200/redirect, partial file cleaned, error logged, and safe user flash."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        def mock_save_with_partial(dst, *args, **kwargs):
            with open(dst, "wb") as f:
                f.write(b"partial-data")
            raise OSError("No space left on device")

        with patch("werkzeug.datastructures.FileStorage.save", side_effect=mock_save_with_partial), \
             self.assertLogs("pdf_analyzer.routes", level="ERROR") as cm:
            res = self.client.post("/upload", data={
                "pdf_file": (io.BytesIO(pdf_bytes), "save_fail.pdf"),
                "csrf_token": token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            self.assertIn(b"An error occurred while saving the uploaded file", res.data)
            # Ensure internal filesystem path or raw exception is not exposed in HTML
            self.assertNotIn(b"No space left on device", res.data)
            self.assertNotIn(b"save_fail.pdf", res.data)

            # Assert partial file was removed
            uploaded_files = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
            self.assertEqual(len(uploaded_files), 0)

            # Assert original save failure was logged
            log_output = "\n".join(cm.output)
            self.assertIn("[UPLOAD ERROR] Failed to save uploaded file", log_output)
            self.assertIn("No space left on device", log_output)

    def test_file_save_partial_cleanup_failure_logs_both_errors(self):
        """Simulate file.save() raising OSError AND partial file deletion raising OSError; verify controlled response and both logs."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        def mock_save_with_partial(dst, *args, **kwargs):
            with open(dst, "wb") as f:
                f.write(b"partial-data")
            raise OSError("I/O error during save")

        with patch("werkzeug.datastructures.FileStorage.save", side_effect=mock_save_with_partial), \
             patch("os.remove", side_effect=OSError("Permission denied on unlink")), \
             self.assertLogs("pdf_analyzer.routes", level="WARNING") as cm:
            res = self.client.post("/upload", data={
                "pdf_file": (io.BytesIO(pdf_bytes), "partial_cleanup_fail.pdf"),
                "csrf_token": token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            self.assertIn(b"An error occurred while saving the uploaded file", res.data)

            log_output = "\n".join(cm.output)
            # Save error logged at ERROR level
            self.assertIn("[UPLOAD ERROR] Failed to save uploaded file", log_output)
            self.assertIn("I/O error during save", log_output)
            # Cleanup failure logged at WARNING level
            self.assertIn("[UPLOAD ERROR] Failed to clean up partial file", log_output)
            self.assertIn("Permission denied on unlink", log_output)

    def test_extraction_failure_cleanup_logs_warning_on_os_error(self):
        """Simulate extraction failure after save, with os.remove raising OSError; verify warning logged and user receives extraction error."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        extraction_failure_dict = {
            "success": False,
            "total_pages": 0,
            "total_characters": 0,
            "has_extractable_text": False,
            "pages": [],
            "error": "This PDF is password-protected and cannot be read."
        }

        with patch("routes.pdf_routes.extract_text_from_pdf", return_value=extraction_failure_dict), \
             patch("os.remove", side_effect=OSError("Windows lock error")), \
             self.assertLogs("pdf_analyzer.routes", level="WARNING") as cm:
            res = self.client.post("/upload", data={
                "pdf_file": (io.BytesIO(pdf_bytes), "encrypted_fail.pdf"),
                "csrf_token": token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            # User receives extraction error
            self.assertIn(b"This PDF is password-protected and cannot be read.", res.data)
            # Internal lock error is not in HTML
            self.assertNotIn(b"Windows lock error", res.data)

            # Cleanup failure is logged with context
            log_output = "\n".join(cm.output)
            self.assertIn("[UPLOAD ERROR] Failed to clean up invalid PDF file", log_output)
            self.assertIn("Windows lock error", log_output)

    def test_successful_upload_remains_unaffected(self):
        """Verify successful upload completes normally, file remains saved, and status is 200."""
        pdf_bytes = create_minimal_pdf_bytes(num_pages=1)
        token = self.get_csrf_token()

        res = self.client.post("/upload", data={
            "pdf_file": (io.BytesIO(pdf_bytes), "success_flow.pdf"),
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"success_flow.pdf", res.data)

        # File is preserved in upload folder
        uploaded_files = [f for f in os.listdir(self.test_upload_dir) if f.endswith(".pdf")]
        self.assertEqual(len(uploaded_files), 1)


if __name__ == "__main__":
    unittest.main()
