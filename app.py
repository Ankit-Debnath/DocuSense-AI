import os
from flask import Flask, flash, redirect, url_for, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from flask_wtf.csrf import CSRFProtect, CSRFError
from config import (
    Config,
    ProductionConfig,
    DevelopmentConfig,
    DEFAULT_INSTANCE_FOLDER,
    validate_secret_key,
    validate_cookie_security,
    resolve_session_cookie_secure,
    resolve_database_path,
    validate_database_path,
    resolve_upload_folder,
    validate_upload_folder,
)
from routes.pdf_routes import pdf_bp
from routes.health_routes import health_bp
from database.db import init_app as init_db_app
from observability import init_observability

csrf = CSRFProtect()

def create_app(config_class=Config):
    """
    Application factory that initializes and configures the Flask application.
    Validates secret key and security configuration before startup.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Dynamic session cookie security resolution:
    # If SESSION_COOKIE_SECURE environment variable is explicitly provided, it overrides config.
    # If base Config is used, dynamically evaluate secure cookie flag against current runtime environment.
    env_cookie_secure = os.environ.get("SESSION_COOKIE_SECURE")
    if env_cookie_secure is not None and env_cookie_secure.strip() != "":
        clean_secure = env_cookie_secure.strip().lower()
        if clean_secure in ("1", "true", "yes", "on"):
            app.config["SESSION_COOKIE_SECURE"] = True
        elif clean_secure in ("0", "false", "no", "off"):
            app.config["SESSION_COOKIE_SECURE"] = False
        else:
            raise ValueError(
                f"Invalid SESSION_COOKIE_SECURE environment variable value: '{env_cookie_secure}'. "
                "Expected 'true', 'false', '1', or '0'."
            )
    elif config_class is Config:
        app.config["SESSION_COOKIE_SECURE"] = resolve_session_cookie_secure(app.config)

    # Dynamic request ID trust resolution from environment variable
    env_trust_req_id = os.environ.get("TRUST_INCOMING_REQUEST_ID")
    if env_trust_req_id is not None and env_trust_req_id.strip() != "":
        app.config["TRUST_INCOMING_REQUEST_ID"] = env_trust_req_id.strip().lower() in ("1", "true", "yes", "on")

    # Dynamic storage path resolution and validation:
    is_standard_config = config_class in (Config, ProductionConfig, DevelopmentConfig)

    # 1. DATABASE_PATH
    if not is_standard_config and hasattr(config_class, "DATABASE_PATH") and "DATABASE_PATH" in config_class.__dict__:
        raw_db = getattr(config_class, "DATABASE_PATH")
        app.config["DATABASE_PATH"] = validate_database_path(raw_db)
    else:
        app.config["DATABASE_PATH"] = resolve_database_path(app.config)

    # 2. UPLOAD_FOLDER
    if not is_standard_config and hasattr(config_class, "UPLOAD_FOLDER") and "UPLOAD_FOLDER" in config_class.__dict__:
        raw_upload = getattr(config_class, "UPLOAD_FOLDER")
        app.config["UPLOAD_FOLDER"] = validate_upload_folder(raw_upload)
    else:
        app.config["UPLOAD_FOLDER"] = resolve_upload_folder(app.config)

    # Validate SECRET_KEY and session-cookie security for production deployments
    validate_secret_key(app.config)
    validate_cookie_security(app.config)

    # Ensure required runtime directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.dirname(app.config["DATABASE_PATH"]), exist_ok=True)
    os.makedirs(app.config.get("INSTANCE_FOLDER", DEFAULT_INSTANCE_FOLDER), exist_ok=True)

    # Initialize SQLite database and teardown handlers
    init_db_app(app)

    # Initialize request correlation, structured access logging, and error correlation
    init_observability(app)

    # Initialize CSRF protection globally
    csrf.init_app(app)

    # Register blueprints
    app.register_blueprint(pdf_bp)
    csrf.exempt(health_bp)
    app.register_blueprint(health_bp)

    # Handle file uploads that exceed MAX_CONTENT_LENGTH (10 MB)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(error):
        flash("File is too large! Maximum allowed file size is 10 MB.", "error")
        return redirect(url_for("pdf.index"))

    # User-friendly error handler for expired or missing CSRF security tokens
    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        is_ajax_or_api = (
            request.is_json
            or request.path == "/explain-page"
            or request.path.startswith("/api/")
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        )
        if is_ajax_or_api:
            return jsonify({
                "success": False,
                "error": f"CSRF validation failed: {error.description}. Please refresh the page.",
                "csrf_expired": True
            }), 400
        flash(f"Security validation failed: {error.description}. Please refresh and submit the form again.", "error")
        return render_template("index.html"), 400

    # Set standard HTTP security headers on all responses
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self'; "
            "base-uri 'self'; "
            "object-src 'none'"
        )

        # Conditionally enable HSTS only when explicitly configured and served over HTTPS
        if app.config.get("ENABLE_HSTS", False) and request.is_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    # Configure optional trusted reverse proxy support (ProxyFix)
    use_proxy_fix = app.config.get("USE_PROXY_FIX")
    if use_proxy_fix is None:
        use_proxy_fix = os.environ.get("USE_PROXY_FIX", "").strip().lower() in ("1", "true", "yes", "on")

    if use_proxy_fix:
        raw_proxy_count = app.config.get("PROXY_COUNT")
        if raw_proxy_count is None:
            raw_proxy_count = os.environ.get("PROXY_COUNT", "1")
        try:
            proxy_count = int(str(raw_proxy_count).strip())
            if proxy_count <= 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid PROXY_COUNT: {raw_proxy_count!r}. "
                "PROXY_COUNT must be a positive integer representing the number of trusted upstream proxies."
            )
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxy_count, x_proto=proxy_count, x_host=proxy_count)

    return app


def is_debug_enabled() -> bool:
    """
    Determines whether debug mode should be enabled for the development server.
    Defaults to False if FLASK_DEBUG or DEBUG is unset.
    Only explicitly accepted truthy values ('1', 'true', 'yes', 'on') enable debug mode.
    """
    for var in ("FLASK_DEBUG", "DEBUG"):
        val = os.environ.get(var)
        if val is not None:
            return val.strip().lower() in ("1", "true", "yes", "on")
    return False


# Main app instance for running directly or via test clients / WSGI servers
app = create_app()

if __name__ == "__main__":
    debug = is_debug_enabled()
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=debug)
