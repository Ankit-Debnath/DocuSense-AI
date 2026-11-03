"""
Google GenAI Embedding Service Module
Provides document and query embedding generation using the official Google GenAI SDK.
Enforces stable task types (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY), safe batching,
rigorous vector validation (finiteness, dimension consistency), exponential backoff retries,
and safe error reporting.
"""

import logging
import math
import sys
from typing import Any, Dict, List, Optional, Union

try:
    from google.genai import types
except ImportError:
    types = None

from ai.llm_client import create_gemini_client, get_embedding_model_name, DEFAULT_EMBEDDING_MODEL
from ai.analyzer import _execute_with_retry, _format_error_message
from database.models import serialize_embedding, deserialize_embedding

# Structured logger for embedding service
logger = logging.getLogger("pdf_analyzer.embedding")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Standard Google GenAI Task Types
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Default batch size for document chunk embeddings (respects provider ceilings)
DEFAULT_EMBEDDING_BATCH_SIZE = 50


class EmbeddingError(Exception):
    """Raised when embedding generation or vector validation fails."""
    pass


def get_embedding_batch_size() -> int:
    """
    Retrieves the configured embedding batch size from application config or environment.
    Falls back safely to DEFAULT_EMBEDDING_BATCH_SIZE (50).
    """
    import os
    env_val = os.environ.get("EMBEDDING_BATCH_SIZE")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    try:
        from flask import current_app
        if current_app:
            val = current_app.config.get("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    try:
        from config import Config
        val = getattr(Config, "EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
        if isinstance(val, int) and val > 0:
            return val
    except Exception:
        pass

    return DEFAULT_EMBEDDING_BATCH_SIZE


def validate_embedding_vector(vector: Any, expected_dim: Optional[int] = None) -> List[float]:
    """
    Validates that a vector is a non-empty, finite, numeric sequence.
    Rejects booleans, non-numbers, NaN, infinity, and dimensional mismatches.

    Parameters:
        vector: The raw vector sequence from the API or mock response.
        expected_dim: Optional integer dimension to enforce consistency across a batch.

    Returns:
        List[float]: The validated vector as clean Python floats.

    Raises:
        ValueError: If the vector is empty, contains non-numeric/non-finite values,
                    or fails the dimensional check.
    """
    if not isinstance(vector, (list, tuple)) or len(vector) == 0:
        raise ValueError("Embedding vector must be a non-empty sequence of numbers.")

    clean_vector: List[float] = []
    for idx, val in enumerate(vector):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Embedding value at index {idx} is non-numeric: {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val):
            raise ValueError(f"Embedding value at index {idx} is non-finite (NaN or Inf): {val!r}")
        clean_vector.append(f_val)

    if expected_dim is not None and len(clean_vector) != expected_dim:
        raise ValueError(
            f"Embedding vector dimension mismatch: expected {expected_dim}, got {len(clean_vector)}."
        )

    return clean_vector


def _extract_vector_values(raw_item: Any) -> Any:
    """
    Extracts numerical vector values from an SDK ContentEmbedding object, dict, or sequence.
    """
    if hasattr(raw_item, "values"):
        return raw_item.values
    if isinstance(raw_item, dict) and "values" in raw_item:
        return raw_item["values"]
    if isinstance(raw_item, (list, tuple)):
        return raw_item
    raise ValueError(f"Unrecognized embedding response format: {type(raw_item).__name__}")


def embed_documents(
    texts: Union[List[str], List[Dict[str, Any]]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None
) -> List[List[float]]:
    """
    Generates embedding vectors for document chunks using RETRIEVAL_DOCUMENT.
    Executes in batches of `batch_size` (or configured default) to avoid payload limit exhaustion.
    Preserves exact 1:1 input ordering and ensures uniform embedding dimensions across all batches.

    Parameters:
        texts: A list of text strings or list of chunk dicts (each containing a 'text' key).
        client: Optional pre-configured Google GenAI client instance.
        model_name: Optional embedding model name override (defaults to config or 'gemini-embedding-001').
        batch_size: Optional batch size override (must be a positive int).

    Returns:
        List[List[float]]: A list of validated embedding vectors in 1-to-1 correspondence with input texts.

    Raises:
        EmbeddingError: If the API call fails or the returned vectors are invalid.
        TypeError: If input items are neither str nor dict with 'text'.
        ValueError: If batch_size is not a positive integer.
    """
    if texts is None or len(texts) == 0:
        return []

    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        effective_batch_size = batch_size
    else:
        effective_batch_size = get_embedding_batch_size()

    # Extract clean string inputs preserving correspondence
    extracted_texts: List[str] = []
    for idx, item in enumerate(texts):
        if isinstance(item, str):
            extracted_texts.append(item)
        elif isinstance(item, dict) and "text" in item:
            extracted_texts.append(str(item["text"]))
        else:
            raise TypeError(
                f"Invalid input item at index {idx}: expected str or dict with 'text', got {type(item).__name__}."
            )

    if not extracted_texts:
        return []

    resolved_model = model_name or get_embedding_model_name()
    active_client = client or create_gemini_client()

    config = (
        types.EmbedContentConfig(task_type=TASK_TYPE_DOCUMENT)
        if types is not None
        else {"task_type": TASK_TYPE_DOCUMENT}
    )

    validated_vectors: List[List[float]] = []
    expected_dim: Optional[int] = None
    total_texts = len(extracted_texts)

    for start_idx in range(0, total_texts, effective_batch_size):
        end_idx = min(start_idx + effective_batch_size, total_texts)
        batch_slice = extracted_texts[start_idx:end_idx]
        batch_num = (start_idx // effective_batch_size) + 1

        try:
            def _call_api():
                return active_client.models.embed_content(
                    model=resolved_model,
                    contents=batch_slice,
                    config=config
                )

            response = _execute_with_retry(
                _call_api,
                operation_name=f"Gemini Embed Documents ({resolved_model}, batch={batch_num}, count={len(batch_slice)})"
            )
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError, EmbeddingError)):
                raise
            clean_msg = _format_error_message(exc)
            logger.warning("Document embedding generation failed for batch %d: %s", batch_num, clean_msg)
            raise EmbeddingError(f"Document embedding generation failed: {clean_msg}") from exc

        raw_embeddings = getattr(response, "embeddings", None)
        if raw_embeddings is None:
            raise EmbeddingError(f"Gemini embedding response for batch {batch_num} contained no 'embeddings' field.")

        if len(raw_embeddings) != len(batch_slice):
            raise EmbeddingError(
                f"Embedding count mismatch in batch {batch_num}: sent {len(batch_slice)} texts but received {len(raw_embeddings)} vectors."
            )

        for item_offset, raw_item in enumerate(raw_embeddings):
            global_idx = start_idx + item_offset
            try:
                raw_vals = _extract_vector_values(raw_item)
                clean_vec = validate_embedding_vector(raw_vals, expected_dim=expected_dim)
                if expected_dim is None:
                    expected_dim = len(clean_vec)
                validated_vectors.append(clean_vec)
            except ValueError as val_err:
                raise EmbeddingError(f"Invalid embedding returned at index {global_idx}: {val_err}") from val_err

    return validated_vectors


def embed_query(
    query: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None
) -> List[float]:
    """
    Generates an embedding vector for a search query or user question using RETRIEVAL_QUERY.

    Parameters:
        query: The user query string to embed.
        client: Optional pre-configured Google GenAI client instance.
        model_name: Optional embedding model name override (defaults to config or 'gemini-embedding-001').

    Returns:
        List[float]: A validated embedding vector compatible with document vectors.

    Raises:
        ValueError: If query is empty or whitespace.
        EmbeddingError: If API call fails or returned vector is invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace.")

    cleaned_query = query.strip()
    resolved_model = model_name or get_embedding_model_name()
    active_client = client or create_gemini_client()

    config = (
        types.EmbedContentConfig(task_type=TASK_TYPE_QUERY)
        if types is not None
        else {"task_type": TASK_TYPE_QUERY}
    )

    try:
        def _call_api():
            return active_client.models.embed_content(
                model=resolved_model,
                contents=cleaned_query,
                config=config
            )

        response = _execute_with_retry(
            _call_api,
            operation_name=f"Gemini Embed Query ({resolved_model})"
        )
    except Exception as exc:
        if isinstance(exc, (ValueError, TypeError, EmbeddingError)):
            raise
        clean_msg = _format_error_message(exc)
        logger.warning("Query embedding generation failed: %s", clean_msg)
        raise EmbeddingError(f"Query embedding generation failed: {clean_msg}") from exc

    raw_embeddings = getattr(response, "embeddings", None)
    if not raw_embeddings:
        singular = getattr(response, "embedding", None)
        if singular is not None:
            raw_embeddings = [singular]
        else:
            raise EmbeddingError("Gemini query embedding response contained no embeddings.")

    try:
        raw_vals = _extract_vector_values(raw_embeddings[0])
        return validate_embedding_vector(raw_vals)
    except ValueError as val_err:
        raise EmbeddingError(f"Invalid query embedding returned: {val_err}") from val_err


def embed_single_document(
    text: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None
) -> List[float]:
    """
    Convenience wrapper to embed a single document chunk using RETRIEVAL_DOCUMENT.

    Parameters:
        text: A single document chunk text string.
        client: Optional Google GenAI client instance.
        model_name: Optional model override.

    Returns:
        List[float]: A validated embedding vector.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text cannot be empty or whitespace.")

    vectors = embed_documents([text.strip()], client=client, model_name=model_name)
    if not vectors:
        raise EmbeddingError("No embedding vector was produced for the document.")
    return vectors[0]
