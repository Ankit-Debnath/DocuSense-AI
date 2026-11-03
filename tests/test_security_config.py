"""
Focused regression test suite for Phase 4, Step 2A:
Secure Configuration and API-Key Management:
- Task 1: Strict Production SECRET_KEY enforcement & safe dev fallback
- Task 2: Environment-driven debug mode defaulting to False & WSGI export safety
- Task 3: Unauthenticated /save-key route security in production and safe error responses
"""

import os
import re
import tempfile
import unittest
from unittest.mock import patch, mock_open

from flask import Flask, session
from app import create_app, is_debug_enabled, app as wsgi_app
from config import (
    Config,
    ProductionConfig,
    DevelopmentConfig,
    DEV_SECRET_KEY,
    is_production_mode,
    validate_secret_key,
    validate_cookie_security,
    resolve_session_cookie_secure,
)
from ai.llm_client import save_api_key, get_api_key


class SecurityConfigTestCase(unittest.TestCase):
    """Tests for Task 1: SECRET_KEY validation and environment detection."""

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_production_init_fails_when_secret_key_missing(self):
        """Production initialization must fail fast when SECRET_KEY is missing."""
        os.environ["PRODUCTION"] = "true"
        os.environ.pop("SECRET_KEY", None)

        class ProdNoSecret(ProductionConfig):
            SECRET_KEY = None

        with self.assertRaises(RuntimeError) as ctx:
            create_app(ProdNoSecret)
        self.assertIn("SECRET_KEY must be explicitly configured", str(ctx.exception))

    def test_production_init_fails_when_secret_key_is_empty_or_whitespace(self):
        """Production initialization must reject empty or whitespace-only SECRET_KEY."""
        os.environ["PRODUCTION"] = "true"

        class ProdEmptySecret(ProductionConfig):
            SECRET_KEY = "   "

        with self.assertRaises(RuntimeError) as ctx:
            create_app(ProdEmptySecret)
        self.assertIn("SECRET_KEY must be explicitly configured", str(ctx.exception))

    def test_production_init_fails_when_secret_key_is_insecure_default(self):
        """Production initialization must reject known development fallback secret."""
        os.environ["PRODUCTION"] = "true"

        class ProdDevSecret(ProductionConfig):
            SECRET_KEY = DEV_SECRET_KEY

        with self.assertRaises(RuntimeError) as ctx:
            create_app(ProdDevSecret)
        self.assertIn("known insecure default", str(ctx.exception))

    def test_production_init_fails_when_secret_key_too_short(self):
        """Production initialization must reject secrets shorter than 16 characters."""
        os.environ["PRODUCTION"] = "true"

        class ProdShortSecret(ProductionConfig):
            SECRET_KEY = "short-secret"

        with self.assertRaises(RuntimeError) as ctx:
            create_app(ProdShortSecret)
        self.assertIn("too short for production", str(ctx.exception))

    def test_production_init_accepts_valid_strong_secret(self):
        """Production initialization succeeds when a sufficiently strong secret is provided."""
        os.environ["PRODUCTION"] = "true"
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProdValidConfig(ProductionConfig):
            SECRET_KEY = strong_secret
            TESTING = True

        app = create_app(ProdValidConfig)
        self.assertEqual(app.config["SECRET_KEY"], strong_secret)

    def test_development_init_remains_usable_with_dev_fallback(self):
        """Development initialization remains convenient and functional without explicit secret."""
        os.environ.pop("SECRET_KEY", None)
        os.environ["FLASK_ENV"] = "development"
        os.environ.pop("PRODUCTION", None)

        class DevConfig(DevelopmentConfig):
            TESTING = True

        app = create_app(DevConfig)
        self.assertEqual(app.config["SECRET_KEY"], DEV_SECRET_KEY)
        client = app.test_client()
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

    def test_secret_values_never_leaked_in_error_messages(self):
        """Secret values must never be exposed or printed inside exception messages."""
        os.environ["PRODUCTION"] = "true"
        secret_attempt = "leaky-bad-key"

        class ProdTestSecret(ProductionConfig):
            SECRET_KEY = secret_attempt

        with self.assertRaises(RuntimeError) as ctx:
            create_app(ProdTestSecret)

        error_message = str(ctx.exception)
        self.assertNotIn(secret_attempt, error_message)


class DebugModeTestCase(unittest.TestCase):
    """Tests for Task 2: Environment-driven debug mode and WSGI export."""

    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_debug_mode_defaults_to_false_when_absent(self):
        """Debug mode must default to False when neither FLASK_DEBUG nor DEBUG is set."""
        os.environ.pop("FLASK_DEBUG", None)
        os.environ.pop("DEBUG", None)
        self.assertFalse(is_debug_enabled())

    def test_debug_mode_explicit_false_values_remain_false(self):
        """Explicit falsy values ('0', 'false', 'no', 'off') must keep debug mode disabled."""
        for falsy in ("0", "false", "False", "no", "off", "disabled"):
            with self.subTest(val=falsy):
                os.environ["FLASK_DEBUG"] = falsy
                self.assertFalse(is_debug_enabled())

    def test_debug_mode_explicit_truthy_values_enable_debug(self):
        """Only explicitly accepted truthy values ('1', 'true', 'yes', 'on') enable debug mode."""
        for truthy in ("1", "true", "True", "yes", "on"):
            with self.subTest(val=truthy):
                os.environ["FLASK_DEBUG"] = truthy
                self.assertTrue(is_debug_enabled())

    def test_wsgi_application_export_remains_intact(self):
        """WSGI entrypoints must have an accessible 'app' instance with debug=False by default."""
        self.assertIsInstance(wsgi_app, Flask)
        self.assertFalse(wsgi_app.debug)
        self.assertTrue(callable(create_app))


class SaveKeySecurityTestCase(unittest.TestCase):
    """Tests for Task 3: Secure /save-key route behavior in production and development."""

    def setUp(self):
        self.orig_env = os.environ.copy()
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self, client):
        res = client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_unauthenticated_request_cannot_modify_env_in_production(self):
        """In production, /save-key must reject requests with HTTP 403 without modifying credentials."""
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProductionTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProductionTestConfig)
        client = app.test_client()
        csrf_token = self.get_csrf_token(client)

        with patch("ai.llm_client.save_api_key") as mock_save:
            res = client.post("/save-key", data={
                "gemini_api_key": "AIzaSyMaliciousAttackerKey12345",
                "return_url": "/",
                "csrf_token": csrf_token
            })
            self.assertEqual(res.status_code, 403)
            self.assertIn(b"disabled in production", res.data)
            mock_save.assert_not_called()

    def test_rejected_request_does_not_change_existing_credentials(self):
        """When /save-key is rejected in production, existing environment key is preserved."""
        os.environ["GEMINI_API_KEY"] = "initial_legitimate_gemini_key_777"
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProductionTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProductionTestConfig)
        client = app.test_client()
        csrf_token = self.get_csrf_token(client)

        res = client.post("/save-key", data={
            "gemini_api_key": "AIzaSyNewUntrustedKey999",
            "csrf_token": csrf_token
        })
        self.assertEqual(res.status_code, 403)
        self.assertEqual(os.environ.get("GEMINI_API_KEY"), "initial_legitimate_gemini_key_777")

    def test_api_key_values_never_returned_in_response(self):
        """Submitted API keys must never be echoed back in HTML or JSON responses."""
        strong_secret = "production-super-strong-secret-key-32chars!"
        secret_token = "AIzaSySuperSecretPrivateTokenXYZ123456"

        class ProductionTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProductionTestConfig)
        client = app.test_client()
        csrf_token = self.get_csrf_token(client)

        res = client.post("/save-key", data={
            "gemini_api_key": secret_token,
            "csrf_token": csrf_token
        })
        self.assertEqual(res.status_code, 403)
        self.assertNotIn(secret_token.encode("utf-8"), res.data)

    def test_failed_save_returns_safe_error_response_in_dev(self):
        """In development, a failed filesystem write must return a safe error message."""
        class DevTestConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(DevTestConfig)
        client = app.test_client()
        csrf_token = self.get_csrf_token(client)

        with patch("routes.pdf_routes.save_api_key", return_value=False):
            res = client.post("/save-key", data={
                "gemini_api_key": "AIzaSyValidFormatKey12345",
                "return_url": "/",
                "csrf_token": csrf_token
            }, follow_redirects=True)

            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Failed to save API key", res.data)
            self.assertNotIn(b"AIzaSyValidFormatKey12345", res.data)

    def test_save_api_key_handles_unwritable_env_gracefully(self):
        """save_api_key() must return False and log a warning on unwritable disk without crashing."""
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", side_effect=OSError("Read-only filesystem")):
            result = save_api_key("some_key_value")
            self.assertFalse(result)

    def test_environment_based_api_key_configuration_still_works(self):
        """Verifies that environment-based API key loading functions reliably."""
        os.environ["GEMINI_API_KEY"] = "AIzaSyValidEnvironmentKey123456"
        with patch("os.path.exists", return_value=False):
            resolved = get_api_key()
            self.assertEqual(resolved, "AIzaSyValidEnvironmentKey123456")

    def test_csrf_protection_remains_effective_on_save_key(self):
        """Missing or forged CSRF tokens must be rejected with HTTP 400."""
        class DevTestConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(DevTestConfig)
        client = app.test_client()

        # Missing token
        res_missing = client.post("/save-key", data={"gemini_api_key": "dummy"})
        self.assertEqual(res_missing.status_code, 400)
        self.assertIn(b"Security validation failed", res_missing.data)

        # Forged token
        res_forged = client.post("/save-key", data={
            "gemini_api_key": "dummy",
            "csrf_token": "tampered_bad_token"
        })
        self.assertEqual(res_forged.status_code, 400)
        self.assertIn(b"Security validation failed", res_forged.data)

    def test_json_post_to_save_key_in_production_returns_403_json(self):
        """JSON requests to /save-key in production receive a clean JSON 403 error."""
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProductionTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProductionTestConfig)
        client = app.test_client()
        csrf_token = self.get_csrf_token(client)

        res = client.post(
            "/save-key",
            headers={"X-CSRFToken": csrf_token},
            json={"gemini_api_key": "AIzaSyKey123"}
        )
        self.assertEqual(res.status_code, 403)
        body = res.get_json()
        self.assertFalse(body["success"])
        self.assertIn("disabled in production", body["error"])



class SessionCookieSecurityTestCase(unittest.TestCase):
    """Tests for Step 6.5.2.1: Production Session-Cookie Security Hardening."""

    def setUp(self):
        self.orig_env = os.environ.copy()
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_development_session_cookie_defaults(self):
        """In development, session cookies must default to HttpOnly=True, SameSite='Lax', Secure=False."""
        os.environ["FLASK_ENV"] = "development"
        os.environ.pop("PRODUCTION", None)
        os.environ.pop("SESSION_COOKIE_SECURE", None)

        class DevTestConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(DevTestConfig)
        self.assertTrue(app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(app.config["SESSION_COOKIE_SAMESITE"], "Lax")
        self.assertFalse(app.config["SESSION_COOKIE_SECURE"])

    def test_production_session_cookie_defaults(self):
        """In production, session cookies must default to HttpOnly=True, SameSite='Lax', Secure=True."""
        os.environ["PRODUCTION"] = "true"
        os.environ.pop("SESSION_COOKIE_SECURE", None)
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProdTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProdTestConfig)
        self.assertTrue(app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(app.config["SESSION_COOKIE_SAMESITE"], "Lax")
        self.assertTrue(app.config["SESSION_COOKIE_SECURE"])

    def test_resolve_session_cookie_secure_mode_resolution(self):
        """resolve_session_cookie_secure defaults to False in dev and True in prod."""
        os.environ.pop("SESSION_COOKIE_SECURE", None)
        self.assertFalse(resolve_session_cookie_secure({"ENV": "development"}))
        self.assertTrue(resolve_session_cookie_secure({"ENV": "production"}))

    def test_resolve_session_cookie_secure_environment_override(self):
        """SESSION_COOKIE_SECURE env var overrides deployment mode defaults."""
        for truthy in ("1", "true", "True", "yes", "on"):
            with self.subTest(val=truthy):
                os.environ["SESSION_COOKIE_SECURE"] = truthy
                self.assertTrue(resolve_session_cookie_secure({"ENV": "development"}))

        for falsy in ("0", "false", "False", "no", "off"):
            with self.subTest(val=falsy):
                os.environ["SESSION_COOKIE_SECURE"] = falsy
                self.assertFalse(resolve_session_cookie_secure({"ENV": "production"}))

    def test_resolve_session_cookie_secure_invalid_value_raises_value_error(self):
        """Invalid SESSION_COOKIE_SECURE env var values raise ValueError."""
        for invalid in ("invalid", "2", "maybe", "enable"):
            with self.subTest(val=invalid):
                os.environ["SESSION_COOKIE_SECURE"] = invalid
                with self.assertRaises(ValueError) as ctx:
                    resolve_session_cookie_secure()
                self.assertIn("Invalid SESSION_COOKIE_SECURE", str(ctx.exception))

    def test_create_app_rejects_invalid_session_cookie_secure_env(self):
        """create_app fails fast with ValueError on invalid SESSION_COOKIE_SECURE env var."""
        os.environ["SESSION_COOKIE_SECURE"] = "not-a-boolean"

        class DevTestConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        with self.assertRaises(ValueError) as ctx:
            create_app(DevTestConfig)
        self.assertIn("Invalid SESSION_COOKIE_SECURE", str(ctx.exception))

    def test_production_fails_fast_when_session_cookie_secure_is_false(self):
        """In production, SESSION_COOKIE_SECURE=False fails fast with RuntimeError."""
        os.environ["PRODUCTION"] = "true"
        strong_secret = "production-super-strong-secret-key-32chars!"

        class InsecureProdConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            SESSION_COOKIE_SECURE = False
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        with self.assertRaises(RuntimeError) as ctx:
            create_app(InsecureProdConfig)
        self.assertIn("SESSION_COOKIE_SECURE must be set to True in production", str(ctx.exception))

    def test_production_fails_fast_when_session_cookie_httponly_is_false(self):
        """In production, SESSION_COOKIE_HTTPONLY=False fails fast with RuntimeError."""
        os.environ["PRODUCTION"] = "true"
        strong_secret = "production-super-strong-secret-key-32chars!"

        class InsecureHttpOnlyProdConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            SESSION_COOKIE_HTTPONLY = False
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        with self.assertRaises(RuntimeError) as ctx:
            create_app(InsecureHttpOnlyProdConfig)
        self.assertIn("SESSION_COOKIE_HTTPONLY must be set to True in production", str(ctx.exception))

    def test_production_fails_fast_when_session_cookie_samesite_is_invalid(self):
        """In production, SESSION_COOKIE_SAMESITE not in ('Lax', 'Strict') fails fast with RuntimeError."""
        os.environ["PRODUCTION"] = "true"
        strong_secret = "production-super-strong-secret-key-32chars!"

        for bad_samesite in ("None", "none", "invalid", False, None):
            with self.subTest(val=bad_samesite):
                class InsecureSameSiteProdConfig(ProductionConfig):
                    TESTING = True
                    SECRET_KEY = strong_secret
                    SESSION_COOKIE_SAMESITE = bad_samesite
                    UPLOAD_FOLDER = self.test_upload_dir
                    INSTANCE_FOLDER = self.test_instance_dir

                with self.assertRaises(RuntimeError) as ctx:
                    create_app(InsecureSameSiteProdConfig)
                self.assertIn("SESSION_COOKIE_SAMESITE must be 'Lax' or 'Strict'", str(ctx.exception))

    def test_development_validation_passes_silently_with_insecure_cookie(self):
        """In development mode, validate_cookie_security passes silently even if cookie is insecure."""
        dev_config = {
            "ENV": "development",
            "SESSION_COOKIE_SECURE": False,
            "SESSION_COOKIE_HTTPONLY": True,
            "SESSION_COOKIE_SAMESITE": "Lax",
        }
        # Must not raise
        validate_cookie_security(dev_config)

    def test_development_session_cookie_in_http_response(self):
        """In development, HTTP response Set-Cookie contains HttpOnly and SameSite=Lax but not Secure."""
        os.environ["FLASK_ENV"] = "development"
        os.environ.pop("PRODUCTION", None)
        os.environ.pop("SESSION_COOKIE_SECURE", None)

        class DevTestConfig(DevelopmentConfig):
            TESTING = True
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(DevTestConfig)

        @app.route("/test-set-dev-session")
        def set_dev_sess():
            session["test_key"] = "test_val"
            return "session_saved"

        client = app.test_client()
        res = client.get("/test-set-dev-session")
        cookie_header = res.headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("SameSite=Lax", cookie_header)
        self.assertNotIn("Secure", cookie_header)

    def test_production_session_cookie_in_http_response(self):
        """In production, HTTP response Set-Cookie contains HttpOnly, SameSite=Lax, and Secure."""
        os.environ["PRODUCTION"] = "true"
        os.environ.pop("SESSION_COOKIE_SECURE", None)
        strong_secret = "production-super-strong-secret-key-32chars!"

        class ProdTestConfig(ProductionConfig):
            TESTING = True
            SECRET_KEY = strong_secret
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir

        app = create_app(ProdTestConfig)

        @app.route("/test-set-prod-session")
        def set_prod_sess():
            session["test_key"] = "test_val"
            return "session_saved"

        client = app.test_client()
        res = client.get("/test-set-prod-session")
        cookie_header = res.headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("SameSite=Lax", cookie_header)
        self.assertIn("Secure", cookie_header)


if __name__ == "__main__":
    unittest.main()
