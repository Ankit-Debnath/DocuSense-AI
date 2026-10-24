import io
import os
import re
import shutil
import tempfile
import unittest
import pypdf
from app import create_app
from config import Config
from pdf_processing.validator import generate_safe_filename, validate_pdf_file

class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True


class PDFUploadTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test uploads
        self.test_upload_dir = tempfile.mkdtemp()
        
        # Configure test app to use the temporary upload directory
        class CustomTestConfig(TestConfig):
            UPLOAD_FOLDER = self.test_upload_dir

        self.app = create_app(CustomTestConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        # Remove temporary test directory and any uploaded files
        if os.path.exists(self.test_upload_dir):
            shutil.rmtree(self.test_upload_dir)

    def get_csrf_token(self, client=None):
        """Fetches a valid CSRF token from the application homepage."""
        active_client = client or self.client
        res = active_client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_homepage_renders_upload_form(self):
        """Test that homepage loads successfully and includes the multipart upload form."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PDF Analyzer", response.data)
        self.assertIn(b'enctype="multipart/form-data"', response.data)
        self.assertIn(b'type="file"', response.data)

    def test_valid_pdf_upload_renders_results(self):
        """Test uploading a valid PDF document renders the results page."""
        # Create a valid minimal PDF using pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)

        data = {
            "pdf_file": (io.BytesIO(pdf_bytes.getvalue()), "test_doc.pdf"),
            "csrf_token": self.get_csrf_token()
        }

        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Extraction Results", response.data)
        self.assertIn(b"test_doc.pdf", response.data)

        # Verify that a file was actually saved in the upload directory
        saved_files = os.listdir(self.test_upload_dir)
        self.assertEqual(len(saved_files), 1)
        self.assertTrue(saved_files[0].endswith("test_doc.pdf"))

    def test_upload_missing_file_part(self):
        """Test POST request without the file field."""
        data = {"csrf_token": self.get_csrf_token()}
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No file part found in request.", response.data)

    def test_upload_empty_filename(self):
        """Test submitting the form with an empty file selection."""
        data = {
            "pdf_file": (io.BytesIO(b""), ""),
            "csrf_token": self.get_csrf_token()
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No file selected", response.data)

    def test_upload_invalid_extension(self):
        """Test uploading a non-PDF file like a .txt or .png file."""
        data = {
            "pdf_file": (io.BytesIO(b"Hello world text"), "notes.txt"),
            "csrf_token": self.get_csrf_token()
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Only PDF files (.pdf) are allowed.", response.data)

    def test_upload_fake_pdf_signature(self):
        """Test uploading a file with .pdf extension but invalid header bytes."""
        fake_pdf_data = b"This is plain text pretending to be a pdf."
        data = {
            "pdf_file": (io.BytesIO(fake_pdf_data), "corrupted.pdf"),
            "csrf_token": self.get_csrf_token()
        }
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The uploaded file is not a valid PDF document.", response.data)

    def test_safe_filename_generation(self):
        """Test that directory traversal characters are sanitized."""
        unsafe_name = "../../../secret_system_file.pdf"
        safe_name = generate_safe_filename(unsafe_name)
        
        # Verify no directory separators remain
        self.assertNotIn("..", safe_name)
        self.assertNotIn("/", safe_name)
        self.assertNotIn("\\", safe_name)
        self.assertTrue(safe_name.endswith("secret_system_file.pdf"))

    def test_upload_file_exceeds_max_size(self):
        """Test that file exceeding MAX_CONTENT_LENGTH is handled gracefully."""
        # Create an app instance with a tiny 50-byte max limit
        class LowLimitConfig(TestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            MAX_CONTENT_LENGTH = 50

        tiny_app = create_app(LowLimitConfig)
        tiny_client = tiny_app.test_client()

        large_payload = b"%PDF-" + b"A" * 200
        data = {
            "pdf_file": (io.BytesIO(large_payload), "large.pdf"),
            "csrf_token": self.get_csrf_token(client=tiny_client)
        }
        response = tiny_client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"File is too large", response.data)


if __name__ == "__main__":
    unittest.main()
