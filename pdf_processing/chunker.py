"""
PDF Text Chunker & Passage Retrieval Module
Splits page-by-page extracted text into overlapping chunks while preserving page number citations,
and provides passage retrieval for grounded question answering.
Features semantic sentence- and paragraph-preserving text splitting.
"""

import os
import re
from typing import Any, Dict, List, Optional

# Basic English stopwords for keyword relevance scoring
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is",
    "are", "was", "were", "it", "this", "that", "of", "from", "by", "as", "be",
    "what", "when", "where", "who", "which", "why", "how", "can", "could", "do",
    "does", "did", "have", "has", "had", "not", "but"
}

# Conservative default chunking parameters
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def get_chunk_size() -> int:
    """
    Retrieves the configured target chunk size in characters from application config or env.
    """
    try:
        from flask import current_app
        if current_app:
            val = current_app.config.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
            if isinstance(val, int) and val > 0:
                return val
    except (RuntimeError, ImportError):
        pass

    val = os.environ.get("CHUNK_SIZE")
    if val and val.strip().isdigit() and int(val.strip()) > 0:
        return int(val.strip())

    return DEFAULT_CHUNK_SIZE


def get_chunk_overlap() -> int:
    """
    Retrieves the configured chunk overlap in characters from application config or env.
    """
    try:
        from flask import current_app
        if current_app:
            val = current_app.config.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
            if isinstance(val, int) and val >= 0:
                return val
    except (RuntimeError, ImportError):
        pass

    val = os.environ.get("CHUNK_OVERLAP")
    if val and val.strip().isdigit() and int(val.strip()) >= 0:
        return int(val.strip())

    return DEFAULT_CHUNK_OVERLAP


def _split_into_atomic_units(text: str, max_len: int, overlap: int) -> List[str]:
    r"""
    Recursively decomposes text into atomic units each <= max_len.
    Preserves structural boundaries in order of priority:
      1. Paragraph breaks (\n\s*\n+)
      2. Sentence boundaries ((?<=[.!?])\s+)
      3. Word boundaries (\s+)
      4. Hard character slices for monolithic tokens exceeding max_len
    """
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= max_len:
        return [clean]

    # Level 1: Split by paragraph breaks (\n\s*\n+)
    paragraphs = re.split(r'(\n\s*\n+)', clean)
    if len(paragraphs) > 1:
        units = []
        for p in paragraphs:
            if not p:
                continue
            if re.match(r'^\n\s*\n+$', p):
                if units:
                    units[-1] += p
                continue
            units.extend(_split_into_atomic_units(p, max_len, overlap))
        return units

    # Level 2: Split by sentence boundaries ([.!?] followed by whitespace)
    sentences = re.split(r'((?<=[.!?])\s+)', clean)
    if len(sentences) > 1:
        units = []
        for s in sentences:
            if not s:
                continue
            if re.match(r'^\s+$', s):
                if units:
                    units[-1] += s
                continue
            units.extend(_split_into_atomic_units(s, max_len, overlap))
        return units

    # Level 3: Split by word boundaries (whitespace)
    words = re.split(r'(\s+)', clean)
    if len(words) > 1:
        units = []
        for w in words:
            if not w:
                continue
            if re.match(r'^\s+$', w):
                if units:
                    units[-1] += w
                continue
            units.extend(_split_into_atomic_units(w, max_len, overlap))
        return units

    # Level 4: Arbitrary character slices for an oversized continuous token
    step = max(1, max_len - overlap)
    slices = []
    start = 0
    while start < len(clean):
        slices.append(clean[start:start + max_len])
        if start + max_len >= len(clean):
            break
        start += step
    return slices


def semantic_chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[str]:
    """
    Splits arbitrary text into overlapping chunks, preferring paragraph and sentence boundaries.

    Parameters:
        text: The raw text string to chunk.
        chunk_size: Target maximum characters per chunk (defaults to config or 800).
        chunk_overlap: Overlap character count between consecutive chunks (defaults to config or 150).

    Returns:
        List[str]: List of non-empty chunk strings.
    """
    effective_size = chunk_size if chunk_size is not None else get_chunk_size()
    effective_overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()

    if effective_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if effective_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if effective_size <= effective_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= effective_size:
        return [clean]

    units = _split_into_atomic_units(clean, effective_size, effective_overlap)
    if not units:
        return []

    chunks: List[str] = []
    i = 0
    n = len(units)

    while i < n:
        curr_units = []
        curr_len = 0
        j = i

        while j < n:
            u = units[j]
            u_len = len(u)
            if curr_units and (curr_len + u_len > effective_size):
                break
            curr_units.append(u)
            curr_len += u_len
            j += 1

        chunk_str = "".join(curr_units).strip()
        if chunk_str:
            chunks.append(chunk_str)

        if j >= n:
            break

        # Calculate overlap starting unit for next chunk
        overlap_len = 0
        next_i = j
        for k in range(j - 1, i, -1):
            cand_len = overlap_len + len(units[k])
            if cand_len <= effective_overlap:
                overlap_len = cand_len
                next_i = k
            else:
                break

        # Guarantee strict forward progress to prevent infinite loops
        if next_i <= i:
            next_i = i + 1

        i = next_i

    return chunks


def chunk_extracted_pages(
    pages: List[Dict[str, Any]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Splits page-extracted text into overlapping chunks, preserving source page citations
    and adhering strictly to paragraph and sentence boundaries where possible.

    Parameters:
        pages: List of dicts per page [{'page_number': int, 'text': str, ...}]
        chunk_size: Target maximum characters per chunk (defaults to config or 800).
        chunk_overlap: Overlap character count between consecutive chunks (defaults to config or 150).

    Returns:
        List[Dict[str, Any]]: Chunks with 'chunk_id', 'chunk_index', 'page_number', 'text', and 'char_count'.
    """
    effective_size = chunk_size if chunk_size is not None else get_chunk_size()
    effective_overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()

    if effective_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if effective_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if effective_size <= effective_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page_info in pages:
        page_num = page_info.get("page_number", 1)
        raw_text = str(page_info.get("text") or "").strip()

        if not raw_text:
            continue

        page_chunks = semantic_chunk_text(
            raw_text,
            chunk_size=effective_size,
            chunk_overlap=effective_overlap
        )

        for chunk_text in page_chunks:
            chunks.append({
                "chunk_id": chunk_counter,
                "chunk_index": chunk_counter - 1,
                "page_number": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text)
            })
            chunk_counter += 1

    return chunks


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return [word.lower() for word in re.findall(r"\b\w+\b", text) if len(word) > 1]


def is_broad_query(query: str) -> bool:
    """
    Detects whether a user query is asking for a comprehensive, section-by-section,
    or broad overview of the document (e.g. 'Explain this PDF in detail').
    """
    q = (query or "").lower().strip()
    broad_indicators = (
        "explain this pdf", "explain the pdf", "explain this document", "explain the document",
        "explain in detail", "explain this paper", "in detail", "section by section",
        "detailed explanation", "walk me through", "overview of", "comprehensive summary",
        "summarize this", "break down this", "what is this document about", "what is this pdf about",
        "what is this paper about", "explain everything", "full breakdown", "summary of the pdf"
    )
    return any(ind in q for ind in broad_indicators)


def retrieve_relevant_chunks(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant passages for a query using keyword frequency scoring
    or balanced document distribution for broad overview inquiries.
    Grounds question answering without needing an external vector database.

    Parameters:
        chunks: List of document chunks.
        query: The user's question or search query.
        top_k: Maximum number of passages to return.

    Returns:
        List of the top matching chunk dictionaries, sorted by relevance score.
    """
    if not chunks or not query.strip():
        return chunks[:top_k]

    # For broad queries asking for comprehensive or section-by-section explanation,
    # sample representative chunks distributed across the document
    if is_broad_query(query) and len(chunks) > top_k:
        step = len(chunks) / float(top_k)
        selected_indices = [int(round(i * step)) for i in range(top_k)]
        seen = set()
        distributed = []
        for idx in selected_indices:
            clamped = min(max(0, idx), len(chunks) - 1)
            if clamped not in seen:
                seen.add(clamped)
                distributed.append(chunks[clamped])
        if len(distributed) == top_k:
            return distributed

    query_tokens = [t for t in _tokenize(query) if t not in STOPWORDS]
    if not query_tokens:
        query_tokens = _tokenize(query)

    scored_chunks = []
    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        score = 0
        for token in query_tokens:
            count = chunk_text_lower.count(token)
            if count > 0:
                score += 1 + count * 0.5  # Weight term occurrences

        scored_chunks.append((score, chunk))

    # Sort descending by score
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Return top_k with positive score, or fallback to first chunks if no terms matched
    matched = [chunk for score, chunk in scored_chunks if score > 0]
    if matched:
        return matched[:top_k]

    return [chunk for _, chunk in scored_chunks[:top_k]]
