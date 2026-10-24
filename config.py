import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Attempt to load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

class Config:
    # Secret key for signing session cookies and flash messages
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # CSRF protection enabled across all state-changing endpoints
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour token expiration

    # Upload configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    
    # Maximum file upload size: 10 Megabytes (10 * 1024 * 1024 bytes)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    # Permitted file extensions
    ALLOWED_EXTENSIONS = {"pdf"}

    # Database configuration (stored in dedicated instance folder)
    INSTANCE_FOLDER = os.path.join(BASE_DIR, "instance")
    DATABASE_PATH = os.path.join(INSTANCE_FOLDER, "pdf_analyzer.db")

    # Google Gemini API configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    # gemini-flash-lite-latest provides ~3-5s low latency responses compared to heavy tiers
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
    # Timeout in seconds for Gemini API calls to prevent indefinite hangs
    GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
    # Maximum characters sent to Gemini for summarization (bounded condensation)
    MAX_SUMMARY_CHARACTERS = int(os.environ.get("MAX_SUMMARY_CHARACTERS", "16000"))
