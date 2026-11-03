import os
from typing import Any, Dict, Optional

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Attempt to load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
except ImportError:
    pass

# Known insecure or default secrets that must never be used in production
DEV_SECRET_KEY = "dev-secret-key-change-in-production"
INSECURE_SECRET_KEYS = {
    DEV_SECRET_KEY.lower(),
    "secret",
    "secret_key",
    "secretkey",
    "password",
    "change-me",
    "change_me",
    "change_in_production",
    "default",
    "test-secret",
    "12345678",
}


def is_production_mode(config_dict: Optional[Any] = None) -> bool:
    """
    Determines whether the application is running in production mode.
    Inspects config object/dict and standard environment variables:
      - Config.ENV / Config.ENVIRONMENT in ('production', 'prod')
      - FLASK_ENV in ('production', 'prod')
      - ENVIRONMENT in ('production', 'prod')
      - APP_ENV in ('production', 'prod')
      - PRODUCTION in ('1', 'true', 'yes')
    """
    candidates = []
    if config_dict is not None:
        if isinstance(config_dict, dict):
            candidates.append(config_dict.get("ENV"))
            candidates.append(config_dict.get("ENVIRONMENT"))
        else:
            candidates.append(getattr(config_dict, "ENV", None))
            candidates.append(getattr(config_dict, "ENVIRONMENT", None))

    candidates.extend([
        os.environ.get("FLASK_ENV"),
        os.environ.get("ENVIRONMENT"),
        os.environ.get("APP_ENV"),
    ])

    for c in candidates:
        if c and str(c).strip().lower() in ("production", "prod"):
            return True

    prod_flag = os.environ.get("PRODUCTION", "").strip().lower()
    if prod_flag in ("1", "true", "yes"):
        return True

    return False


def validate_secret_key(config_dict: Any) -> None:
    """
    Validates that the SECRET_KEY meets production security requirements.
    Fails fast with RuntimeError if running in production without an explicitly
    configured, sufficiently strong secret key.
    Never exposes the key value or parts of it in exception messages or logs.
    """
    if not is_production_mode(config_dict):
        return

    secret = None
    if isinstance(config_dict, dict):
        secret = config_dict.get("SECRET_KEY")
    elif config_dict is not None:
        secret = getattr(config_dict, "SECRET_KEY", None)

    if not secret or not isinstance(secret, (str, bytes)):
        raise RuntimeError("SECRET_KEY must be explicitly configured in production environment.")

    clean = secret.decode("utf-8", errors="ignore") if isinstance(secret, bytes) else str(secret)
    clean = clean.strip()

    if not clean:
        raise RuntimeError("SECRET_KEY must be explicitly configured in production environment.")

    if clean.lower() in INSECURE_SECRET_KEYS:
        raise RuntimeError(
            "Configured SECRET_KEY is a known insecure default. "
            "A strong random secret is required for production."
        )

    if len(clean) < 16:
        raise RuntimeError(
            "Configured SECRET_KEY is too short for production. "
            "A secret of at least 16 characters is required."
        )


def resolve_session_cookie_secure(config_dict: Optional[Any] = None) -> bool:
    """
    Resolves the SESSION_COOKIE_SECURE setting based on environment variable
    override or active deployment mode.

    If the SESSION_COOKIE_SECURE environment variable is set, it takes precedence
    ('1', 'true', 'yes', 'on' -> True; '0', 'false', 'no', 'off' -> False).
    Otherwise, defaults to True in production mode and False in development mode.
    """
    env_override = os.environ.get("SESSION_COOKIE_SECURE")
    if env_override is not None and env_override.strip() != "":
        clean = env_override.strip().lower()
        if clean in ("1", "true", "yes", "on"):
            return True
        if clean in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"Invalid SESSION_COOKIE_SECURE environment variable value: '{env_override}'. "
            "Expected 'true', 'false', '1', or '0'."
        )
    return is_production_mode(config_dict)


def validate_cookie_security(config_dict: Any) -> None:
    """
    Validates that session cookie security settings meet production requirements.
    Fails fast with RuntimeError if running in production without secure session cookie attributes.
    In development mode, validation passes silently.
    """
    if not is_production_mode(config_dict):
        return

    def _get_val(key: str, default: Any = None) -> Any:
        if isinstance(config_dict, dict):
            return config_dict.get(key, default)
        return getattr(config_dict, key, default)

    cookie_secure = _get_val("SESSION_COOKIE_SECURE")
    if cookie_secure is not True:
        raise RuntimeError(
            "SESSION_COOKIE_SECURE must be set to True in production environment."
        )

    cookie_httponly = _get_val("SESSION_COOKIE_HTTPONLY")
    if cookie_httponly is not True:
        raise RuntimeError(
            "SESSION_COOKIE_HTTPONLY must be set to True in production environment."
        )

    cookie_samesite = _get_val("SESSION_COOKIE_SAMESITE")
    samesite_normalized = None
    if isinstance(cookie_samesite, str):
        samesite_normalized = cookie_samesite.strip().capitalize()

    if samesite_normalized not in ("Lax", "Strict"):
        raise RuntimeError(
            f"SESSION_COOKIE_SAMESITE must be 'Lax' or 'Strict' in production environment (got {cookie_samesite!r})."
        )


# Default persistent storage locations
DEFAULT_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DEFAULT_INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")
DEFAULT_DATABASE_PATH = os.path.join(DEFAULT_INSTANCE_FOLDER, "pdf_analyzer.db")


def validate_database_path(db_path: Any) -> str:
    """
    Validates that db_path is a usable SQLite database file path.
    Ensures parent directory exists or can be created.
    Fails fast with ValueError if the path is invalid, a directory, or has an unwritable/file parent.
    Returns the resolved absolute path.
    """
    if db_path is None or not str(db_path).strip():
        raise ValueError("DATABASE_PATH cannot be empty or whitespace.")

    clean_path = str(db_path).strip()
    abs_path = os.path.abspath(clean_path if os.path.isabs(clean_path) else os.path.join(BASE_DIR, clean_path))

    if os.path.isdir(abs_path):
        raise ValueError(f"Configured DATABASE_PATH is an existing directory, not a database file: '{abs_path}'")

    parent_dir = os.path.dirname(abs_path)
    if os.path.isfile(parent_dir):
        raise ValueError(f"Parent path of DATABASE_PATH is an existing file, not a directory: '{parent_dir}'")

    try:
        os.makedirs(parent_dir, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create or access parent directory for DATABASE_PATH '{abs_path}': {exc}") from exc

    return abs_path


def resolve_database_path(config_dict: Optional[Any] = None) -> str:
    """
    Resolves and validates the SQLite database path.
    Priority order:
    1. Explicit value in config_dict / config object (if provided and different from default).
    2. Explicit value in DATABASE_PATH environment variable.
    3. Default path (instance/pdf_analyzer.db).
    """
    raw = None
    if config_dict is not None:
        if isinstance(config_dict, dict):
            raw = config_dict.get("DATABASE_PATH")
        else:
            raw = getattr(config_dict, "DATABASE_PATH", None)

    # Check environment variable if not explicitly set in config_dict or matches default
    if raw is None or raw == DEFAULT_DATABASE_PATH or str(raw).strip() == "":
        env_val = os.environ.get("DATABASE_PATH")
        if env_val is not None:
            if not env_val.strip():
                raise ValueError("Configured DATABASE_PATH environment variable cannot be empty or whitespace.")
            raw = env_val

    if raw is not None and str(raw).strip() != "":
        return validate_database_path(str(raw))

    return validate_database_path(DEFAULT_DATABASE_PATH)


def validate_upload_folder(upload_folder: Any) -> str:
    """
    Validates that upload_folder is a usable directory path.
    Ensures the directory exists or can be created.
    Fails fast with ValueError if the path is invalid, an existing file, or cannot be created.
    Returns the resolved absolute path.
    """
    if upload_folder is None or not str(upload_folder).strip():
        raise ValueError("UPLOAD_FOLDER cannot be empty or whitespace.")

    clean_path = str(upload_folder).strip()
    abs_path = os.path.abspath(clean_path if os.path.isabs(clean_path) else os.path.join(BASE_DIR, clean_path))

    if os.path.isfile(abs_path):
        raise ValueError(f"Configured UPLOAD_FOLDER is an existing file, not a directory: '{abs_path}'")

    try:
        os.makedirs(abs_path, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create or access UPLOAD_FOLDER '{abs_path}': {exc}") from exc

    return abs_path


def resolve_upload_folder(config_dict: Optional[Any] = None) -> str:
    """
    Resolves and validates the uploads directory path.
    Priority order:
    1. Explicit value in config_dict / config object (if provided and different from default).
    2. Explicit value in UPLOAD_FOLDER environment variable.
    3. Default path (uploads/).
    """
    raw = None
    if config_dict is not None:
        if isinstance(config_dict, dict):
            raw = config_dict.get("UPLOAD_FOLDER")
        else:
            raw = getattr(config_dict, "UPLOAD_FOLDER", None)

    if raw is None or raw == DEFAULT_UPLOAD_FOLDER or str(raw).strip() == "":
        env_val = os.environ.get("UPLOAD_FOLDER")
        if env_val is not None:
            if not env_val.strip():
                raise ValueError("Configured UPLOAD_FOLDER environment variable cannot be empty or whitespace.")
            raw = env_val

    if raw is not None and str(raw).strip() != "":
        return validate_upload_folder(str(raw))

    return validate_upload_folder(DEFAULT_UPLOAD_FOLDER)


class Config:
    # Environment mode: 'development', 'production', or 'testing'
    ENV = os.environ.get("FLASK_ENV") or os.environ.get("ENVIRONMENT") or os.environ.get("APP_ENV") or "development"

    # Secret key for signing session cookies and flash messages.
    # In development, falls back to DEV_SECRET_KEY to preserve stable sessions across restarts.
    # In production, validate_secret_key() ensures this fallback cannot be used.
    SECRET_KEY = os.environ.get("SECRET_KEY") or (
        None if is_production_mode() else DEV_SECRET_KEY
    )

    # Session cookie security configuration.
    # HTTPOnly prevents JavaScript access to session cookies (mitigates XSS).
    # SameSite=Lax prevents CSRF on cross-site requests while allowing top-level navigations.
    # Secure ensures cookies are only transmitted over HTTPS (default True in production, False in dev).
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = resolve_session_cookie_secure()

    # Web-based API key modification is strictly disabled in production
    ENABLE_WEB_KEY_CONFIG = not is_production_mode()

    # CSRF protection enabled across all state-changing endpoints
    WTF_CSRF_ENABLED = True
    # Default to 86400 (24 hours) token validity to prevent token expiry while reading long documents
    WTF_CSRF_TIME_LIMIT = int(os.environ.get("WTF_CSRF_TIME_LIMIT", "86400"))

    # HTTP Strict Transport Security (HSTS) configuration.
    # Defaults to False to prevent breaking local HTTP development.
    # Enable explicitly in production deployments served over HTTPS.
    ENABLE_HSTS = os.environ.get("ENABLE_HSTS", "").strip().lower() in ("1", "true", "yes", "on")

    # Trusted reverse proxy configuration (ProxyFix).
    # Disabled by default. Enable only when deployed behind a trusted upstream reverse proxy.
    USE_PROXY_FIX = os.environ.get("USE_PROXY_FIX", "").strip().lower() in ("1", "true", "yes", "on")
    PROXY_COUNT = os.environ.get("PROXY_COUNT", "1")

    # Request ID trust configuration.
    # Disabled by default. When True, accepts incoming X-Request-ID headers from a trusted upstream proxy.
    TRUST_INCOMING_REQUEST_ID = os.environ.get("TRUST_INCOMING_REQUEST_ID", "").strip().lower() in ("1", "true", "yes", "on")

    # Upload configuration
    UPLOAD_FOLDER = resolve_upload_folder()
    
    # Maximum file upload size: 10 Megabytes (10 * 1024 * 1024 bytes)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    # Permitted file extensions
    ALLOWED_EXTENSIONS = {"pdf"}

    # Database configuration (stored in dedicated instance folder by default)
    INSTANCE_FOLDER = DEFAULT_INSTANCE_FOLDER
    DATABASE_PATH = resolve_database_path()

    # Google Gemini API configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    # gemini-flash-lite-latest provides ~3-5s low latency responses compared to heavy tiers
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    # Text embedding configuration for semantic RAG (gemini-embedding-001)
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

    # Safe batch size for embedding API calls to respect provider instance ceilings
    try:
        EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "50"))
        if EMBEDDING_BATCH_SIZE <= 0:
            EMBEDDING_BATCH_SIZE = 50
    except (ValueError, TypeError):
        EMBEDDING_BATCH_SIZE = 50

    # Semantic chunking configuration (character counts)
    try:
        CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
        if CHUNK_SIZE <= 0:
            CHUNK_SIZE = 800
    except (ValueError, TypeError):
        CHUNK_SIZE = 800

    try:
        CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))
        if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
            CHUNK_OVERLAP = 150
    except (ValueError, TypeError):
        CHUNK_OVERLAP = 150

    # Timeout in seconds for Gemini API calls to prevent indefinite hangs
    try:
        GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
    except (ValueError, TypeError):
        GEMINI_TIMEOUT_SECONDS = 30

    # Maximum characters sent to Gemini for summarization (bounded condensation)
    try:
        MAX_SUMMARY_CHARACTERS = int(os.environ.get("MAX_SUMMARY_CHARACTERS", "16000"))
    except (ValueError, TypeError):
        MAX_SUMMARY_CHARACTERS = 16000

    # Maximum allowed pages in a PDF document (bounded resource limit)
    try:
        MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "100"))
        if MAX_PDF_PAGES <= 0:
            MAX_PDF_PAGES = 100
    except (ValueError, TypeError):
        MAX_PDF_PAGES = 100

    # Maximum total extracted characters across all pages (decompression bomb guard)
    try:
        MAX_EXTRACTED_CHARACTERS = int(os.environ.get("MAX_EXTRACTED_CHARACTERS", "250000"))
        if MAX_EXTRACTED_CHARACTERS <= 0:
            MAX_EXTRACTED_CHARACTERS = 250000
    except (ValueError, TypeError):
        MAX_EXTRACTED_CHARACTERS = 250000

    # Q&A session-cookie bounds to prevent client cookie overflow
    try:
        MAX_QA_HISTORY_TURNS = int(os.environ.get("MAX_QA_HISTORY_TURNS", "5"))
        if MAX_QA_HISTORY_TURNS <= 0:
            MAX_QA_HISTORY_TURNS = 5
    except (ValueError, TypeError):
        MAX_QA_HISTORY_TURNS = 5

    try:
        MAX_QA_QUESTION_CHARS = int(os.environ.get("MAX_QA_QUESTION_CHARS", "120"))
        if MAX_QA_QUESTION_CHARS <= 0:
            MAX_QA_QUESTION_CHARS = 120
    except (ValueError, TypeError):
        MAX_QA_QUESTION_CHARS = 120

    try:
        MAX_QA_ANSWER_CHARS = int(os.environ.get("MAX_QA_ANSWER_CHARS", "140"))
        if MAX_QA_ANSWER_CHARS <= 0:
            MAX_QA_ANSWER_CHARS = 140
    except (ValueError, TypeError):
        MAX_QA_ANSWER_CHARS = 140


class ProductionConfig(Config):
    """Explicit production configuration requiring secure secrets."""
    ENV = "production"
    SECRET_KEY = os.environ.get("SECRET_KEY")
    ENABLE_WEB_KEY_CONFIG = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = resolve_session_cookie_secure({"ENV": "production"})


class DevelopmentConfig(Config):
    """Explicit development configuration with convenient local defaults."""
    ENV = "development"
    SECRET_KEY = os.environ.get("SECRET_KEY", DEV_SECRET_KEY)
    ENABLE_WEB_KEY_CONFIG = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = resolve_session_cookie_secure({"ENV": "development"})
