"""
PDF Text Chunker & Passage Retrieval Module
Splits page-by-page extracted text into overlapping chunks while preserving page number citations,
and provides passage retrieval for grounded question answering.
"""

import re
from typing import Any, Dict, List

# Basic English stopwords for keyword relevance scoring
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "is",
    "are", "was", "were", "it", "this", "that", "of", "from", "by", "as", "be",
    "what", "when", "where", "who", "which", "why", "how", "can", "could", "do",
    "does", "did", "have", "has", "had", "not", "but"
}


def chunk_extracted_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Splits page-extracted text into overlapping chunks, preserving source page citations.

    Parameters:
        pages: List of dicts per page [{'page_number': int, 'text': str, ...}]
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap character count between consecutive chunks.

    Returns:
        List[Dict[str, Any]]: Chunks with 'chunk_id', 'page_number', 'text', and 'char_count'.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page_info in pages:
        page_num = page_info.get("page_number", 1)
        raw_text = page_info.get("text", "").strip()

        if not raw_text:
            continue

        # If page text is within chunk_size, keep as a single chunk
        if len(raw_text) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_counter,
                "page_number": page_num,
                "text": raw_text,
                "char_count": len(raw_text)
            })
            chunk_counter += 1
            continue

        # Sliding window chunking with overlap
        start = 0
        step = chunk_size - chunk_overlap
        while start < len(raw_text):
            end = start + chunk_size
            chunk_text = raw_text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_counter,
                    "page_number": page_num,
                    "text": chunk_text,
                    "char_count": len(chunk_text)
                })
                chunk_counter += 1

            start += step

    return chunks


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return [word.lower() for word in re.findall(r"\b\w+\b", text) if len(word) > 1]


def retrieve_relevant_chunks(
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant passages for a query using keyword frequency scoring.
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
