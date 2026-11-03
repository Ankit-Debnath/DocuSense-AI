"""
Focused regression tests for HTTP security response headers and HSTS configuration:
- Required standard headers on standard and informational routes (GET /, GET /history)
- Content Security Policy (CSP) directive validation and unsafe-script exclusion
- HSTS absent by default and strictly restricted to secure HTTPS requests when enabled
- Security headers present on 404, CSRF rejection, and oversized upload error responses
"""

import io
import os
import shutil
import tempfile
import unittest

from app import create_app
from config import Config


class SecurityHeadersTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    GEMINI_API_KEY = "mock_test_key"
    ENABLE_HSTS = False


class SecurityHeadersTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(SecurityHeadersTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.CustomConfig = CustomConfig
        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def assert_standard_security_headers(self, response):
        """Helper to assert that standard security headers are set correctly."""
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertEqual(
            response.headers.get("Permissions-Policy"),
            "camera=(), microphone=(), geolocation=()"
        )
        self.assertIn("Content-Security-Policy", response.headers)

    # =========================================================================
    # 1. Standard Routes Header Verification
    # =========================================================================

    def test_required_security_headers_on_index_route(self):
        """GET / must return all required security response headers."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assert_standard_security_headers(res)

    def test_required_security_headers_on_history_route(self):
        """GET /history must return all required security response headers."""
        res = self.client.get("/history")
        self.assertEqual(res.status_code, 200)
        self.assert_standard_security_headers(res)

    # =========================================================================
    # 2. Content Security Policy (CSP) Policy Directives
    # =========================================================================

    def test_csp_header_contains_required_directives_and_excludes_unsafe_scripts(self):
        """CSP must contain required directives and strictly disallow unsafe script execution."""
        res = self.client.get("/")
        csp = res.headers.get("Content-Security-Policy", "")

        # Verify exact required directives
        self.assertIn("default-src 'self'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertIn("style-src 'self' 'unsafe-inline'", csp)
        self.assertIn("font-src 'self'", csp)
        self.assertIn("img-src 'self' data:", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("form-action 'self'", csp)
        self.assertIn("base-uri 'self'", csp)
        self.assertIn("object-src 'none'", csp)

        # Prohibit unsafe scripts
        self.assertNotIn("'unsafe-eval'", csp)
        # Verify 'unsafe-inline' is only in style-src, not script-src
        directives = [d.strip() for d in csp.split(";") if d.strip()]
        script_src_directive = next((d for d in directives if d.startswith("script-src")), None)
        self.assertIsNotNone(script_src_directive)
        self.assertNotIn("'unsafe-inline'", script_src_directive)
        self.assertNotIn("'unsafe-eval'", script_src_directive)

    # =========================================================================
    # 3. HSTS Configuration and HTTPS Conditionality
    # =========================================================================

    def test_hsts_is_absent_by_default(self):
        """HSTS header must be absent by default in development and test configurations."""
        res = self.client.get("/")
        self.assertNotIn("Strict-Transport-Security", res.headers)

    def test_hsts_is_absent_on_insecure_http_even_when_enabled(self):
        """HSTS header must NOT be sent over plain HTTP even when ENABLE_HSTS is True."""
        class HstsEnabledConfig(self.CustomConfig):
            ENABLE_HSTS = True

        app = create_app(HstsEnabledConfig)
        client = app.test_client()

        # Regular HTTP request (request.is_secure is False)
        res = client.get("/", base_url="http://localhost")
        self.assertNotIn("Strict-Transport-Security", res.headers)

    def test_hsts_is_present_on_secure_request_when_enabled(self):
        """HSTS header must be sent on secure HTTPS requests when ENABLE_HSTS is True."""
        class HstsEnabledConfig(self.CustomConfig):
            ENABLE_HSTS = True

        app = create_app(HstsEnabledConfig)
        client = app.test_client()

        # Secure HTTPS request (request.is_secure is True)
        res = client.get("/", base_url="https://localhost")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Strict-Transport-Security", res.headers)
        hsts_value = res.headers.get("Strict-Transport-Security")
        self.assertEqual(hsts_value, "max-age=31536000; includeSubDomains")
        # Ensure 'preload' is not present
        self.assertNotIn("preload", hsts_value.lower())

    # =========================================================================
    # 4. Error Responses Security Headers Verification
    # =========================================================================

    def test_security_headers_present_on_404_response(self):
        """Security headers must be present on 404 Not Found error responses."""
        res = self.client.get("/non-existent-endpoint-404")
        self.assertEqual(res.status_code, 404)
        self.assert_standard_security_headers(res)

    def test_security_headers_present_on_csrf_rejection_response(self):
        """Security headers must be present on 400 CSRF rejection responses."""
        # Form submission lacking CSRF token
        res_form = self.client.post("/save-key", data={"gemini_api_key": "test_key"})
        self.assertEqual(res_form.status_code, 400)
        self.assert_standard_security_headers(res_form)

        # JSON submission lacking CSRF header
        res_json = self.client.post("/summarize", json={"stored_filename": "doc.pdf"})
        self.assertEqual(res_json.status_code, 400)
        self.assert_standard_security_headers(res_json)

    def test_security_headers_present_on_oversized_upload_response(self):
        """Security headers must be present when an upload exceeds MAX_CONTENT_LENGTH."""
        # Configure a small MAX_CONTENT_LENGTH to trigger RequestEntityTooLarge cleanly
        class TinyUploadConfig(self.CustomConfig):
            MAX_CONTENT_LENGTH = 100  # 100 bytes limit

        app = create_app(TinyUploadConfig)
        client = app.test_client()

        oversized_data = b"%PDF-1.4 " + (b"A" * 200)
        res = client.post(
            "/upload",
            data={"pdf_file": (io.BytesIO(oversized_data), "too_large.pdf")},
            content_type="multipart/form-data"
        )
        # The app error handler redirects to pdf.index with a flash message
        self.assertEqual(res.status_code, 302)
        self.assert_standard_security_headers(res)


if __name__ == "__main__":
    unittest.main()
