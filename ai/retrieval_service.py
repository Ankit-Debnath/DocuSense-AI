"""
Retrieval Service Module
Implements semantic vector retrieval, Reciprocal Rank Fusion (RRF),
and hybrid RAG retrieval combining keyword (lexical) and semantic ranking.

Core Guarantees:
  - Document-scoped retrieval (never leaks or searches cross-document chunks).
  - Cosine similarity with strict validation (finite values, matching dimensions, non-zero magnitudes).
  - Model name and dimension compatibility checks before vector math.
  - Safe skipping of malformed stored vectors without failing the request.
  - Reciprocal Rank Fusion (k=60) combining lexical and semantic signals without score contamination.
  - Resilient fallbacks: falls back gracefully to keyword retrieval if embeddings or indexing fail.
"""

import logging
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

from ai.embedding_client import embed_query, get_embedding_model_name, EmbeddingError
from database.models import get_document_chunks, deserialize_embedding
from pdf_processing.chunker import (
    chunk_extracted_pages,
    retrieve_relevant_chunks,
    is_broad_query,
)

logger = logging.getLogger("pdf_analyzer.retrieval")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Standard Reciprocal Rank Fusion constant
DEFAULT_RRF_K = 60


def normalize_model_name(name: Optional[str]) -> str:
    """
    Normalizes embedding model names by stripping whitespace, converting to lowercase,
    and removing any leading 'models/' prefix so 'models/gemini-embedding-001'
    and 'gemini-embedding-001' compare identically.
    """
    if not name:
        return ""
    clean = str(name).strip().lower()
    if clean.startswith("models/"):
        clean = clean[len("models/"):]
    return clean.strip()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates cosine similarity between two numeric vectors:
        cosine(A, B) = (A . B) / (||A|| * ||B||)

    Parameters:
        vec_a: Non-empty sequence of finite numeric float values.
        vec_b: Non-empty sequence of finite numeric float values with identical dimension.

    Returns:
        float: Cosine similarity score bounded in [-1.0, 1.0].

    Raises:
        TypeError: If either vector is not a list or tuple, or contains non-numeric values.
        ValueError: If vectors are empty, dimensions mismatch, values are non-finite (NaN/Inf),
                    or either vector has zero magnitude.
    """
    if not isinstance(vec_a, (list, tuple)) or not isinstance(vec_b, (list, tuple)):
        raise TypeError("Vectors must be sequences (list or tuple).")

    dim_a = len(vec_a)
    dim_b = len(vec_b)

    if dim_a == 0 or dim_b == 0:
        raise ValueError("Vectors cannot be empty.")

    if dim_a != dim_b:
        raise ValueError(f"Vector dimension mismatch: vector A has {dim_a} dimensions, vector B has {dim_b}.")

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for i in range(dim_a):
        val_a = vec_a[i]
        val_b = vec_b[i]

        if isinstance(val_a, bool) or not isinstance(val_a, (int, float)):
            raise TypeError(f"Vector A contains non-numeric value at index {i}: {val_a!r}")
        if isinstance(val_b, bool) or not isinstance(val_b, (int, float)):
            raise TypeError(f"Vector B contains non-numeric value at index {i}: {val_b!r}")

        f_a = float(val_a)
        f_b = float(val_b)

        if not math.isfinite(f_a):
            raise ValueError(f"Vector A contains non-finite value at index {i}: {val_a!r}")
        if not math.isfinite(f_b):
            raise ValueError(f"Vector B contains non-finite value at index {i}: {val_b!r}")

        dot_product += f_a * f_b
        norm_a_sq += f_a * f_a
        norm_b_sq += f_b * f_b

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)

    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("Cannot calculate cosine similarity for zero-magnitude vector.")

    similarity = dot_product / (norm_a * norm_b)
    # Clamp bounds to guard against floating-point rounding errors (e.g. 1.0000000000000002)
    return max(-1.0, min(1.0, similarity))


def retrieve_semantic_chunks(
    document_id: int,
    query: str,
    top_k: int = 4,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most semantically similar chunks for a query from SQLite vector storage.
    Enforces document-scoping, model compatibility, dimension alignment, and corrupt BLOB resilience.

    Parameters:
        document_id: Integer document ID to scope retrieval to.
        query: User search query or question.
        top_k: Maximum number of relevant semantic chunks to return.
        client: Optional Gemini client instance.
        model_name: Optional model override.
        db_path: Optional SQLite database path.

    Returns:
        List[Dict[str, Any]]: Ranked chunk dictionaries with similarity scores, sorted descending.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        return []

    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return []

    # 1. Generate query embedding using existing embed_query service
    resolved_model = model_name or get_embedding_model_name()
    try:
        query_vec = embed_query(cleaned_query, client=client, model_name=resolved_model)
    except Exception as exc:
        logger.warning(
            "Query embedding generation failed for document_id=%d: %s. Semantic retrieval unavailable.",
            document_id,
            exc
        )
        return []

    query_dim = len(query_vec)
    normalized_active_model = normalize_model_name(resolved_model)

    # 2. Load stored document chunks for the requested document
    try:
        rows = get_document_chunks(document_id, db_path=db_path)
    except Exception as exc:
        logger.error("Failed to load chunks from database for document_id=%d: %s", document_id, exc)
        return []

    if not rows:
        logger.info("No stored chunks found for document_id=%d. Semantic retrieval unavailable.", document_id)
        return []

    # 3. Score chunks with dimension and model compatibility safeguards
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        raw_blob = row["embedding"]
        if raw_blob is None or len(raw_blob) == 0:
            continue

        stored_model = row["embedding_model"]
        if stored_model:
            normalized_stored_model = normalize_model_name(stored_model)
            if normalized_stored_model != normalized_active_model:
                logger.warning(
                    "Model mismatch for document_id=%d: stored '%s' != active '%s'. Skipping semantic candidate.",
                    document_id,
                    stored_model,
                    resolved_model
                )
                return []

        stored_dim = row["embedding_dim"]
        if stored_dim is not None and stored_dim != query_dim:
            logger.warning(
                "Dimension mismatch for document_id=%d: stored dim %d != query dim %d. Skipping semantic retrieval.",
                document_id,
                stored_dim,
                query_dim
            )
            return []

        # Safe deserialization with expected dimension validation
        try:
            chunk_vec = deserialize_embedding(raw_blob, expected_dimension=stored_dim or query_dim)
        except Exception as exc:
            logger.warning(
                "Skipping malformed or corrupted embedding in chunk id=%d for document_id=%d: %s",
                row["id"],
                document_id,
                exc
            )
            continue

        # Cosine similarity computation
        try:
            sim_score = cosine_similarity(query_vec, chunk_vec)
        except Exception as exc:
            logger.warning(
                "Skipping chunk id=%d due to similarity calculation failure: %s",
                row["id"],
                exc
            )
            continue

        candidates.append({
            "document_id": document_id,
            "chunk_id": row["id"],
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "text": row["chunk_text"],
            "similarity_score": sim_score,
            "source": "semantic"
        })

    if not candidates:
        return []

    # Sort candidates descending by similarity score
    candidates.sort(key=lambda item: item["similarity_score"], reverse=True)
    return candidates[:top_k]


def _get_chunk_identity(chunk: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Computes a stable identity key to identify matching chunks across lexical and semantic lists.
    Prefers (page_number, chunk_index); falls back to (page_number, text_snippet).
    """
    page_num = chunk.get("page_number", 1)
    chunk_idx = chunk.get("chunk_index")
    if chunk_idx is not None:
        return (page_num, chunk_idx)

    txt = str(chunk.get("text") or chunk.get("chunk_text") or "").strip()
    return (page_num, txt[:120])


def reciprocal_rank_fusion(
    lexical_chunks: List[Dict[str, Any]],
    semantic_chunks: List[Dict[str, Any]],
    top_k: int = 4,
    k: int = DEFAULT_RRF_K
) -> List[Dict[str, Any]]:
    """
    Combines ranked candidate lists from lexical and semantic retrieval using
    Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum_{r in R} (1 / (k + rank_r(d)))
    where rank_r(d) is 1-based (rank 1 -> 1 / (k + 1)).

    Parameters:
        lexical_chunks: Top candidates from keyword-based retriever.
        semantic_chunks: Top candidates from vector cosine similarity.
        top_k: Number of final combined chunks to return.
        k: Smoothing constant (default: 60).

    Returns:
        List[Dict[str, Any]]: Deduplicated, fused candidate chunks ordered by descending RRF score.
    """
    if not lexical_chunks and not semantic_chunks:
        return []

    if not semantic_chunks:
        result = []
        for rank, chunk in enumerate(lexical_chunks[:top_k], start=1):
            c = dict(chunk)
            c.setdefault("text", c.get("chunk_text", ""))
            c["rrf_score"] = 1.0 / (k + rank)
            c["sources"] = ["lexical"]
            result.append(c)
        return result

    if not lexical_chunks:
        result = []
        for rank, chunk in enumerate(semantic_chunks[:top_k], start=1):
            c = dict(chunk)
            c.setdefault("text", c.get("chunk_text", ""))
            c["rrf_score"] = 1.0 / (k + rank)
            c["sources"] = ["semantic"]
            result.append(c)
        return result

    rrf_scores: Dict[Tuple[int, Any], float] = {}
    chunk_store: Dict[Tuple[int, Any], Dict[str, Any]] = {}
    sources: Dict[Tuple[int, Any], List[str]] = {}

    # Score lexical candidates (1-based rank)
    for rank, chunk in enumerate(lexical_chunks, start=1):
        ident = _get_chunk_identity(chunk)
        contribution = 1.0 / (k + rank)
        rrf_scores[ident] = rrf_scores.get(ident, 0.0) + contribution
        sources.setdefault(ident, []).append("lexical")

        if ident not in chunk_store:
            c = dict(chunk)
            c.setdefault("text", c.get("chunk_text", ""))
            chunk_store[ident] = c

    # Score semantic candidates (1-based rank)
    for rank, chunk in enumerate(semantic_chunks, start=1):
        ident = _get_chunk_identity(chunk)
        contribution = 1.0 / (k + rank)
        rrf_scores[ident] = rrf_scores.get(ident, 0.0) + contribution
        if "semantic" not in sources.setdefault(ident, []):
            sources[ident].append("semantic")

        if ident not in chunk_store:
            c = dict(chunk)
            c.setdefault("text", c.get("chunk_text", ""))
            chunk_store[ident] = c
        else:
            # Preserve similarity score if available
            if "similarity_score" in chunk:
                chunk_store[ident]["similarity_score"] = chunk["similarity_score"]

    # Stable sort: descending by RRF score, secondary tie-break by chunk_index
    def _sort_key(ident: Tuple[int, Any]) -> Tuple[float, int]:
        score = rrf_scores[ident]
        chunk_obj = chunk_store[ident]
        idx = chunk_obj.get("chunk_index", 0)
        return (score, -idx if isinstance(idx, int) else 0)

    sorted_idents = sorted(rrf_scores.keys(), key=_sort_key, reverse=True)

    final_chunks: List[Dict[str, Any]] = []
    for ident in sorted_idents[:top_k]:
        item = chunk_store[ident]
        item["rrf_score"] = rrf_scores[ident]
        item["sources"] = sources[ident]
        final_chunks.append(item)

    return final_chunks


def retrieve_hybrid_chunks(
    document_id: Optional[int],
    query: str,
    pages: List[Dict[str, Any]],
    top_k: int = 4,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    file_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Coordinates hybrid retrieval for question answering.
    Runs lexical retrieval and, when document_id is present, attempts semantic retrieval
    (with on-demand indexing if needed), then fuses candidates via RRF.

    Parameters:
        document_id: Optional database document ID for vector lookup.
        query: User question.
        pages: Extracted PDF pages for lexical chunking.
        top_k: Number of relevant chunks to retrieve.
        client: Optional Gemini client.
        model_name: Optional embedding model name override.
        file_path: Optional path to PDF file for on-demand indexing.
        db_path: Optional SQLite database path.

    Returns:
        Dict[str, Any]: {
            "chunks": List[Dict[str, Any]],
            "retrieval_method": "hybrid" | "keyword_fallback" | "keyword_only" | "semantic_only",
            "lexical_count": int,
            "semantic_count": int
        }
    """
    cleaned_query = (query or "").strip()
    non_empty_pages = [p for p in pages if p.get("text", "").strip()]

    if not cleaned_query or not non_empty_pages:
        return {
            "chunks": [],
            "retrieval_method": "keyword_only",
            "lexical_count": 0,
            "semantic_count": 0
        }

    # Expand candidate pool size for RRF fusion to prevent premature truncation of dual-ranked chunks
    candidate_pool_size = max(top_k * 3, 15)

    # 1. Lexical Retrieval (Retriever A)
    lexical_chunks: List[Dict[str, Any]] = []
    try:
        raw_chunks = chunk_extracted_pages(non_empty_pages, chunk_size=1000, chunk_overlap=150)
        lexical_chunks = retrieve_relevant_chunks(raw_chunks, cleaned_query, top_k=candidate_pool_size)
    except Exception as exc:
        logger.warning("Lexical chunk retrieval encountered error: %s", exc)

    # If no document ID is available, return lexical retrieval directly
    if document_id is None or document_id <= 0:
        return {
            "chunks": lexical_chunks[:top_k],
            "retrieval_method": "keyword_only",
            "lexical_count": len(lexical_chunks),
            "semantic_count": 0
        }

    # 2. Ensure document is indexed on-demand if needed
    try:
        from ai.indexing_service import ensure_document_indexed
        index_status = ensure_document_indexed(
            document_id=document_id,
            file_path=file_path,
            client=client,
            model_name=model_name,
            db_path=db_path
        )
        has_index = index_status.get("indexed", False) and index_status.get("chunk_count", 0) > 0
    except Exception as exc:
        logger.warning("On-demand indexing check failed for document_id=%d: %s", document_id, exc)
        has_index = False

    # 3. Semantic Retrieval (Retriever B)
    semantic_chunks: List[Dict[str, Any]] = []
    if has_index:
        try:
            semantic_chunks = retrieve_semantic_chunks(
                document_id=document_id,
                query=cleaned_query,
                top_k=candidate_pool_size,
                client=client,
                model_name=model_name,
                db_path=db_path
            )
        except Exception as exc:
            logger.warning("Semantic retrieval failed for document_id=%d: %s", document_id, exc)

    # 4. Fusion and Fallback Decision
    result_chunks: List[Dict[str, Any]] = []
    if semantic_chunks and lexical_chunks:
        result_chunks = reciprocal_rank_fusion(lexical_chunks, semantic_chunks, top_k=top_k, k=DEFAULT_RRF_K)
        method = "hybrid"
    elif semantic_chunks and not lexical_chunks:
        result_chunks = semantic_chunks[:top_k]
        method = "semantic_only"
    else:
        # Keyword fallback when semantic chunks are unavailable or failed
        method = "keyword_fallback" if has_index else "keyword_only"
        result_chunks = lexical_chunks[:top_k]

    if document_id is not None and document_id > 0:
        for c in result_chunks:
            c.setdefault("document_id", document_id)

    return {
        "chunks": result_chunks,
        "retrieval_method": method,
        "lexical_count": len(lexical_chunks),
        "semantic_count": len(semantic_chunks)
    }
