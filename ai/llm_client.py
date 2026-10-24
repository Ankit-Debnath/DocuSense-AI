"""
Gemini Client Module
Initializes the official google-genai client using configured environment variables.
Configures HTTP timeouts and avoids indefinite hangs.
Never hardcodes, logs, or exposes API keys.
"""

import os
from typing import Optional
from flask import current_app

try:
    from google import genai
except ImportError:
    genai = None  # Handled gracefully if SDK is still installing

# Default model offering lowest latency (~3s) for document summarization & QA
DEFAULT_FAST_MODEL = "gemini-flash-lite-latest"
DEFAULT_TIMEOUT_SECONDS = 30


def _is_valid_key_value(val: Optional[str]) -> bool:
    """
    Checks if a string is a non-empty, non-placeholder API key.
    """
    if not val or not val.strip():
        return False
    clean = val.strip().strip("'\"")
    if clean.lower() in ("your_gemini_api_key_here", "your_key", "your_api_key_here", "your_api_key", "none", "null"):
        return False
    return True


def get_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key primarily from the local .env file,
    falling back to active environment or Flask config.
    Dynamically checks .env so manual edits take effect immediately without server restart.
    """
    # 1. Inspect .env file on disk first as the primary source of truth
    env_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("GEMINI_API_KEY="):
                        _, _, val = stripped.partition("=")
                        val = val.strip().strip("'\"")
                        if _is_valid_key_value(val):
                            os.environ["GEMINI_API_KEY"] = val
                            return val
        except Exception:
            pass

    # 2. Check active os.environ
    key = os.environ.get("GEMINI_API_KEY")
    if _is_valid_key_value(key):
        return key.strip().strip("'\"")

    # 3. Check Flask current_app config
    try:
        if current_app:
            key = current_app.config.get("GEMINI_API_KEY")
            if _is_valid_key_value(key):
                return key.strip().strip("'\"")
    except RuntimeError:
        pass

    return None


def get_model_name() -> str:
    """
    Retrieves the configured Gemini model name with fallback to DEFAULT_FAST_MODEL.
    """
    env_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("GEMINI_MODEL="):
                        _, _, val = stripped.partition("=")
                        val = val.strip().strip("'\"")
                        if val:
                            return val
        except Exception:
            pass

    model = os.environ.get("GEMINI_MODEL")
    if model and model.strip():
        return model.strip()

    try:
        if current_app:
            model = current_app.config.get("GEMINI_MODEL")
            if model and model.strip():
                return model.strip()
    except RuntimeError:
        pass

    return DEFAULT_FAST_MODEL


def get_timeout_seconds() -> int:
    """
    Retrieves the configured client timeout in seconds.
    """
    val = os.environ.get("GEMINI_TIMEOUT_SECONDS")
    if val and val.strip().isdigit():
        return int(val.strip())

    try:
        if current_app:
            val = current_app.config.get("GEMINI_TIMEOUT_SECONDS")
            if val is not None:
                return int(val)
    except RuntimeError:
        pass

    return DEFAULT_TIMEOUT_SECONDS


def save_api_key(api_key: str) -> bool:
    """
    Saves the provided API key to the local .env file and active environment.
    Never exposes or logs the key.
    """
    clean_key = (api_key or "").strip()
    if not clean_key:
        return False

    os.environ["GEMINI_API_KEY"] = clean_key
    try:
        if current_app:
            current_app.config["GEMINI_API_KEY"] = clean_key
    except RuntimeError:
        pass

    env_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), ".env")
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith("GEMINI_API_KEY="):
            new_lines.append(f"GEMINI_API_KEY={clean_key}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"\nGEMINI_API_KEY={clean_key}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True


def create_gemini_client(api_key: Optional[str] = None, timeout_seconds: Optional[int] = None):
    """
    Creates and returns an official Google GenAI Client instance configured with a bounded HTTP timeout.

    Args:
        api_key: Optional API key override.
        timeout_seconds: Timeout in seconds for API calls (default 30s).

    Raises:
        ValueError: If the API key is not configured.
        RuntimeError: If the google-genai package is not installed.
    """
    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Please install dependencies.")

    resolved_key = api_key or get_api_key()
    if not resolved_key:
        raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file.")

    timeout = timeout_seconds if timeout_seconds is not None else get_timeout_seconds()
    # google-genai HttpOptions.timeout is in milliseconds
    timeout_ms = int(timeout * 1000)

    try:
        return genai.Client(api_key=resolved_key, http_options={"timeout": timeout_ms})
    except TypeError:
        # Fallback for environments or mocks not supporting http_options kwarg
        return genai.Client(api_key=resolved_key)
