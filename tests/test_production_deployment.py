"""
Focused regression tests for Phase 4 Step 5: Production Deployment Readiness.
Verifies:
- Proxy support (ProxyFix) is disabled by default.
- Forwarded headers are not trusted when proxy support is disabled (spoofing protection).
- Invalid PROXY_COUNT values raise a clear ValueError configuration error.
- Enabled ProxyFix safely recognizes trusted HTTPS forwarded scheme.
- HSTS interacts correctly with ProxyFix and only activates on confirmed HTTPS.
- Standard HTTP security headers remain present under proxy configurations.
- Production secret and debug safeguards remain intact.
"""

import os
import shutil
import tempfile
import unittest

from app import create_app, is_debug_enabled
from config import Config, ProductionConfig


class BaseDeploymentTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    GEMINI_API_KEY = "mock_test_key"


class ProductionDeploymentTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(BaseDeploymentTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.CustomConfig = CustomConfig

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # =========================================================================
    # 1. Default Proxy Behavior (Disabled by default)
    # =========================================================================

    def test_proxy_support_is_disabled_by_default(self):
        """ProxyFix must not be active by default."""
        app = create_app(self.CustomConfig)
        # werkzeug.middleware.proxy_fix.ProxyFix wraps wsgi_app
        # When disabled, wsgi_app is Flask's default wsgi_app method
        from werkzeug.middleware.proxy_fix import ProxyFix
        self.assertNotIsInstance(app.wsgi_app, ProxyFix)

    def test_forwarded_headers_not_trusted_when_proxy_support_disabled(self):
        """Spoofed X-Forwarded-Proto must NOT be trusted when USE_PROXY_FIX is False."""
        class HstsConfig(self.CustomConfig):
            ENABLE_HSTS = True
            USE_PROXY_FIX = False

        app = create_app(HstsConfig)
        client = app.test_client()

        # Send spoofed X-Forwarded-Proto over HTTP
        res = client.get("/", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(res.status_code, 200)
        # Since proxy support is disabled, request is not recognized as secure,
        # and Strict-Transport-Security must NOT be emitted.
        self.assertNotIn("Strict-Transport-Security", res.headers)

    # =========================================================================
    # 2. PROXY_COUNT Validation
    # =========================================================================

    def test_invalid_proxy_count_fails_clearly_when_proxy_enabled(self):
        """PROXY_COUNT must be validated as a positive integer when USE_PROXY_FIX is True."""
        invalid_counts = ["0", 0, "-1", -5, "not_a_number", ""]
        for invalid_val in invalid_counts:
            with self.subTest(proxy_count=invalid_val):
                class InvalidProxyConfig(self.CustomConfig):
                    USE_PROXY_FIX = True
                    PROXY_COUNT = invalid_val

                with self.assertRaises(ValueError) as ctx:
                    create_app(InvalidProxyConfig)
                self.assertIn("Invalid PROXY_COUNT", str(ctx.exception))
                self.assertIn("positive integer", str(ctx.exception))

    # =========================================================================
    # 3. Enabled ProxyFix Functionality
    # =========================================================================

    def test_enabled_proxy_fix_recognizes_trusted_https_scheme(self):
        """When USE_PROXY_FIX is True and PROXY_COUNT is valid, ProxyFix wraps app."""
        class ProxiedConfig(self.CustomConfig):
            USE_PROXY_FIX = True
            PROXY_COUNT = 1

        app = create_app(ProxiedConfig)
        from werkzeug.middleware.proxy_fix import ProxyFix
        self.assertIsInstance(app.wsgi_app, ProxyFix)

    def test_hsts_emitted_when_proxy_fix_enabled_and_forwarded_https(self):
        """HSTS must be emitted when ENABLE_HSTS is True and trusted proxy forwards HTTPS."""
        class ProxiedHstsConfig(self.CustomConfig):
            USE_PROXY_FIX = True
            PROXY_COUNT = 1
            ENABLE_HSTS = True

        app = create_app(ProxiedHstsConfig)
        client = app.test_client()

        # Request with trusted X-Forwarded-Proto: https
        res = client.get("/", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("Strict-Transport-Security", res.headers)
        self.assertEqual(
            res.headers.get("Strict-Transport-Security"),
            "max-age=31536000; includeSubDomains"
        )

    def test_hsts_not_emitted_when_proxy_fix_enabled_but_forwarded_http(self):
        """HSTS must NOT be emitted if the trusted proxy forwarded plain HTTP."""
        class ProxiedHstsConfig(self.CustomConfig):
            USE_PROXY_FIX = True
            PROXY_COUNT = 1
            ENABLE_HSTS = True

        app = create_app(ProxiedHstsConfig)
        client = app.test_client()

        # Request with trusted X-Forwarded-Proto: http
        res = client.get("/", headers={"X-Forwarded-Proto": "http"})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("Strict-Transport-Security", res.headers)

    # =========================================================================
    # 4. Standard Security Headers Under Proxy
    # =========================================================================

    def test_standard_security_headers_remain_present_under_proxy(self):
        """Standard security headers must remain present when ProxyFix is active."""
        class ProxiedConfig(self.CustomConfig):
            USE_PROXY_FIX = True
            PROXY_COUNT = 1

        app = create_app(ProxiedConfig)
        client = app.test_client()

        res = client.get("/", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(res.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertEqual(
            res.headers.get("Permissions-Policy"),
            "camera=(), microphone=(), geolocation=()"
        )
        self.assertIn("Content-Security-Policy", res.headers)

    # =========================================================================
    # 5. Production Safeguards Integrity
    # =========================================================================

    def test_production_secret_safeguard_remains_active(self):
        """ProductionConfig without explicit secret key must fail fast on startup."""
        class InsecureProdConfig(ProductionConfig):
            SECRET_KEY = None
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        with self.assertRaises(RuntimeError) as ctx:
            create_app(InsecureProdConfig)
        self.assertIn("SECRET_KEY must be explicitly configured", str(ctx.exception))

    def test_debug_mode_defaults_to_false(self):
        """Debug mode helper must default to False when environment variables are unset."""
        orig_flask_debug = os.environ.pop("FLASK_DEBUG", None)
        orig_debug = os.environ.pop("DEBUG", None)
        try:
            self.assertFalse(is_debug_enabled())
        finally:
            if orig_flask_debug is not None:
                os.environ["FLASK_DEBUG"] = orig_flask_debug
            if orig_debug is not None:
                os.environ["DEBUG"] = orig_debug

    # =========================================================================
    # 6. WSGI Entrypoint & Request ID Trust Configuration
    # =========================================================================

    def test_app_wsgi_callable_exported_and_valid(self):
        """The main app module must export a callable Flask WSGI application object 'app'."""
        import app as app_module
        self.assertTrue(hasattr(app_module, "app"))
        self.assertTrue(callable(app_module.app))
        self.assertEqual(app_module.app.name, "app")

    def test_trust_incoming_request_id_config_resolution(self):
        """TRUST_INCOMING_REQUEST_ID must default to False and resolve properly from environment."""
        orig_val = os.environ.pop("TRUST_INCOMING_REQUEST_ID", None)
        try:
            # Default resolution
            app_default = create_app(self.CustomConfig)
            self.assertFalse(app_default.config.get("TRUST_INCOMING_REQUEST_ID", False))

            # Explicit truthy resolution via environment
            os.environ["TRUST_INCOMING_REQUEST_ID"] = "true"
            app_trust = create_app(self.CustomConfig)
            self.assertTrue(app_trust.config.get("TRUST_INCOMING_REQUEST_ID"))

            # Explicit falsy resolution via environment
            os.environ["TRUST_INCOMING_REQUEST_ID"] = "false"
            app_falsy = create_app(self.CustomConfig)
            self.assertFalse(app_falsy.config.get("TRUST_INCOMING_REQUEST_ID"))
        finally:
            if orig_val is not None:
                os.environ["TRUST_INCOMING_REQUEST_ID"] = orig_val
            else:
                os.environ.pop("TRUST_INCOMING_REQUEST_ID", None)


if __name__ == "__main__":
    unittest.main()
