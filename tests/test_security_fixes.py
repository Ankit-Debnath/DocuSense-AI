"""
Regression Tests for Confirmed Security Fixes
1. Open Redirect Vulnerability in routes/pdf_routes.py
2. CSRF Protection via Flask-WTF across state-changing endpoints
3. Local Vendor Assets and Fail-Closed Frontend Sanitization
"""

import io
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app import create_app
from config import Config
from routes.pdf_routes import is_safe_redirect_url


class SecurityTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    GEMINI_API_KEY = "mock_test_key"


class SecurityFixesTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(SecurityTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        """Extracts the CSRF token rendered in the homepage form."""
        res = self.client.get("/")
        match = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        self.assertIsNotNone(match, "CSRF token must be present in homepage form")
        return match.group(1)

    # =========================================================================
    # 1. Open Redirect Vulnerability Tests
    # =========================================================================

    def test_is_safe_redirect_url_helper_unit_tests(self):
        """Direct unit tests for the centralized is_safe_redirect_url validator."""
        # Safe relative URLs
        self.assertTrue(is_safe_redirect_url("/"))
        self.assertTrue(is_safe_redirect_url("/history"))
        self.assertTrue(is_safe_redirect_url("/result?stored_filename=doc.pdf"))
        self.assertTrue(is_safe_redirect_url("/dashboard/settings"))

        # Unsafe protocol-relative and backslash URLs
        self.assertFalse(is_safe_redirect_url("//evil.com"))
        self.assertFalse(is_safe_redirect_url(r"/\evil.com"))
        self.assertFalse(is_safe_redirect_url(r"\\evil.com"))
        self.assertFalse(is_safe_redirect_url(r"/path\with\backslash"))

        # Unsafe absolute external URLs
        self.assertFalse(is_safe_redirect_url("https://evil.com"))
        self.assertFalse(is_safe_redirect_url("http://evil.com/phishing"))
        self.assertFalse(is_safe_redirect_url("ftp://malicious.org"))
        self.assertFalse(is_safe_redirect_url("javascript:alert(1)"))
        self.assertFalse(is_safe_redirect_url("data:text/html,<script>alert(1)</script>"))

        # Unsafe relative URLs missing leading slash
        self.assertFalse(is_safe_redirect_url("evil.com"))
        self.assertFalse(is_safe_redirect_url("history"))

        # Null, empty, or non-string inputs
        self.assertFalse(is_safe_redirect_url(""))
        self.assertFalse(is_safe_redirect_url("   "))
        self.assertFalse(is_safe_redirect_url(None))

    def test_save_key_accepts_valid_internal_redirect(self):
        """Verifies legitimate internal navigation via return_url."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidDummyKeyForTest12345",
                "return_url": "/history",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get("Location"), "/history")

    def test_save_key_rejects_protocol_relative_open_redirect(self):
        """Confirms //evil.com is rejected and falls back safely to index."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidDummyKeyForTest12345",
                "return_url": "//evil.com",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get("Location"), "/")

    def test_save_key_rejects_backslash_redirect_trick(self):
        """Confirms /\\evil.com is rejected and falls back safely to index."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidDummyKeyForTest12345",
                "return_url": r"/\evil.com",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get("Location"), "/")

    def test_save_key_rejects_absolute_external_url(self):
        """Confirms https://evil.com is rejected and falls back safely to index."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidDummyKeyForTest12345",
                "return_url": "https://evil.com",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get("Location"), "/")

    def test_save_key_rejects_relative_path_without_leading_slash(self):
        """Confirms evil.com or login without leading slash is rejected."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidDummyKeyForTest12345",
                "return_url": "evil.com",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)
            self.assertEqual(res.headers.get("Location"), "/")

    # =========================================================================
    # 2. CSRF Protection Tests
    # =========================================================================

    def test_csrf_token_present_in_html_meta_and_upload_form(self):
        """Confirms CSRF token is injected into <meta> and form <input> tags."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'<meta name="csrf-token"', res.data)
        self.assertIn(b'name="csrf_token"', res.data)

    def test_post_without_csrf_token_is_rejected_with_400(self):
        """Proves missing CSRF token results in an HTTP 400 rejection."""
        res = self.client.post("/save-key", data={
            "gemini_api_key": "test_key",
            "return_url": "/"
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Security validation failed", res.data)

    def test_post_with_invalid_csrf_token_is_rejected_with_400(self):
        """Proves tampered or forged CSRF tokens are rejected with HTTP 400."""
        res = self.client.post("/save-key", data={
            "gemini_api_key": "test_key",
            "return_url": "/",
            "csrf_token": "tampered_invalid_token_xyz"
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn(b"Security validation failed", res.data)

    def test_post_with_valid_csrf_token_succeeds(self):
        """Proves valid CSRF tokens permit state-changing form submissions."""
        token = self.get_csrf_token()
        with patch("routes.pdf_routes.save_api_key"):
            res = self.client.post("/save-key", data={
                "gemini_api_key": "valid_key",
                "return_url": "/",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 302)

    def test_json_post_without_csrf_header_is_rejected_with_400(self):
        """Confirms JSON POST requests missing X-CSRFToken header are rejected."""
        res = self.client.post("/summarize", json={"stored_filename": "test.pdf"})
        self.assertEqual(res.status_code, 400)
        json_body = res.get_json()
        self.assertFalse(json_body["success"])
        self.assertIn("CSRF validation failed", json_body["error"])

    def test_json_post_with_valid_csrf_header_accepted(self):
        """Confirms JSON POST requests with valid X-CSRFToken header pass CSRF check."""
        token = self.get_csrf_token()
        res = self.client.post(
            "/summarize",
            json={"stored_filename": "nonexistent.pdf"},
            headers={"X-CSRFToken": token}
        )
        # Should pass CSRF check and proceed to filename check (which redirects or flashes)
        self.assertNotEqual(res.status_code, 400)

    # =========================================================================
    # 3. Local Vendor Assets & Fail-Closed Sanitizer Tests
    # =========================================================================

    def test_local_vendor_assets_exist_on_disk(self):
        """Confirms all vendor libraries are bundled locally in static/vendor/."""
        vendor_dir = os.path.join(self.app.root_path, "static", "vendor")
        required_files = [
            "marked.min.js",
            "purify.min.js",
            "katex.min.js",
            "auto-render.min.js",
            "katex.min.css"
        ]
        for f in required_files:
            file_path = os.path.join(vendor_dir, f)
            self.assertTrue(os.path.isfile(file_path), f"Missing bundled vendor file: {f}")
            self.assertGreater(os.path.getsize(file_path), 0, f"Vendor file must not be empty: {f}")

        fonts_dir = os.path.join(vendor_dir, "fonts")
        self.assertTrue(os.path.isdir(fonts_dir), "static/vendor/fonts directory must exist")
        self.assertGreater(len(os.listdir(fonts_dir)), 0, "Vendor fonts must be present")

    def test_local_vendor_assets_served_by_flask(self):
        """Verifies Flask static handler serves local vendor assets with HTTP 200."""
        assets = [
            "/static/vendor/marked.min.js",
            "/static/vendor/purify.min.js",
            "/static/vendor/katex.min.js",
            "/static/vendor/auto-render.min.js",
            "/static/vendor/katex.min.css"
        ]
        for asset in assets:
            res = self.client.get(asset)
            self.assertEqual(res.status_code, 200, f"Failed to serve {asset}")

    def test_result_template_does_not_use_external_cdn_for_vendor_scripts(self):
        """Confirms templates load local vendor assets rather than external CDN URLs."""
        # Create a valid test PDF to render result page
        import pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)

        token = self.get_csrf_token()
        upload_res = self.client.post("/upload", data={
            "pdf_file": (io.BytesIO(pdf_bytes.getvalue()), "cdn_check.pdf"),
            "csrf_token": token
        }, content_type="multipart/form-data", follow_redirects=True)

        self.assertEqual(upload_res.status_code, 200)
        html = upload_res.data.decode("utf-8")

        # Verify local vendor links are used
        self.assertIn("/static/vendor/marked.min.js", html)
        self.assertIn("/static/vendor/purify.min.js", html)
        self.assertIn("/static/vendor/katex.min.js", html)
        self.assertIn("/static/vendor/auto-render.min.js", html)
        self.assertIn("/static/vendor/katex.min.css", html)

        # Confirm external jsdelivr CDN is NOT used
        self.assertNotIn("cdn.jsdelivr.net", html)


if __name__ == "__main__":
    unittest.main()
