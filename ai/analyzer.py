"""
AI Analyzer Module
Provides PDF document summarization and grounded question answering using the Google GenAI SDK.
Grounds answers in retrieved passages with source page citations.
Features bounded text condensation, structured latency logging, and robust error handling.
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from flask import current_app

from ai.llm_client import create_gemini_client, get_model_name
from pdf_processing.chunker import chunk_extracted_pages, retrieve_relevant_chunks

# Structured performance logger
logger = logging.getLogger("pdf_analyzer.ai")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default character budget for summarization prompts (bounded condensation)
DEFAULT_MAX_SUMMARY_CHARACTERS = 16000


def get_max_summary_characters() -> int:
    """
    Retrieves the maximum character budget for document summarization prompts.
    """
    try:
        if current_app:
            return current_app.config.get("MAX_SUMMARY_CHARACTERS", DEFAULT_MAX_SUMMARY_CHARACTERS)
    except RuntimeError:
        pass
    return DEFAULT_MAX_SUMMARY_CHARACTERS


def _format_error_message(exception: Exception) -> str:
    """
    Translates low-level or SDK exceptions into user-friendly messages
    without leaking internal stack traces or credentials.
    """
    err_str = str(exception).lower()

    if "api_key" in err_str or "unauthenticated" in err_str or "invalid_argument" in err_str:
        return "Invalid or missing Google Gemini API key. Please check your GEMINI_API_KEY environment variable."
    if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
        return "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
    if "deadline" in err_str or "timeout" in err_str or "connection" in err_str:
        return "Network timeout while communicating with Google Gemini. Please verify your connection."
    if "not_found" in err_str or "model" in err_str:
        return "The configured Gemini model was not found or is unavailable. Please check GEMINI_MODEL."

    return "An error occurred while processing the request with Gemini. Please try again later."


def prepare_condensed_text(
    pages: List[Dict[str, Any]],
    max_chars: Optional[int] = None
) -> Tuple[str, bool, float]:
    """
    Intelligently prepares text for summarization with bounded character limits.
    If document text exceeds max_chars:
      - Balances budget between Introduction (Head), Middle Sections, and Conclusions (Tail)
      - Guarantees both introductory setup and concluding outcomes are preserved
      - Guarantees total length does not exceed max_chars
    
    Returns:
        tuple: (prepared_text, is_condensed, duration_ms)
    """
    t0 = time.perf_counter()
    limit = max_chars or get_max_summary_characters()
    non_empty = [p for p in pages if p.get("text", "").strip()]

    if not non_empty:
        return "", False, (time.perf_counter() - t0) * 1000

    total_raw_len = sum(len(p.get("text", "")) for p in non_empty)
    if total_raw_len <= limit:
        # Full text fits within budget
        compiled = ""
        for p in non_empty:
            compiled += f"\n[Page {p.get('page_number', 1)}]:\n{p.get('text', '').strip()}\n"
        dt = (time.perf_counter() - t0) * 1000
        return compiled, False, dt

    n_pages = len(non_empty)
    notice = f"\n\n[Note: Document condensed from {n_pages} pages ({total_raw_len} chars) for focused executive summarization.]"
    content_budget = max(200, limit - len(notice))

    # Balanced condensation strategy
    if n_pages <= 2:
        head_pages = non_empty[:1]
        tail_pages = non_empty[1:]
        middle_pages = []
    else:
        head_pages = non_empty[:2]
        tail_pages = non_empty[-1:] if n_pages == 3 else non_empty[-2:]
        middle_pages = non_empty[2:-2] if n_pages > 4 else (non_empty[2:-1] if n_pages == 4 else [])

    # Allocate character budgets: 40% head, 40% tail, 20% middle
    if middle_pages:
        head_budget = int(content_budget * 0.40)
        tail_budget = int(content_budget * 0.40)
        mid_budget = content_budget - head_budget - tail_budget
    else:
        head_budget = int(content_budget * 0.50)
        tail_budget = content_budget - head_budget
        mid_budget = 0

    parts = []

    # 1. Head Pages (Intro / Setup)
    head_per_page = max(50, head_budget // len(head_pages))
    for p in head_pages:
        text = p.get("text", "").strip()
        header = f"\n[Page {p.get('page_number')}]:\n"
        max_body = max(20, head_per_page - len(header))
        if len(text) > max_body:
            text = text[:max_body] + " ... [Page excerpt]"
        parts.append(header + text + "\n")

    # 2. Middle Pages (Body Excerpts)
    if middle_pages and mid_budget > 0:
        mid_per_page = max(50, mid_budget // len(middle_pages))
        for p in middle_pages:
            text = p.get("text", "").strip()
            header = f"\n[Page {p.get('page_number')} (Excerpt)]:\n"
            max_body = max(20, mid_per_page - len(header))
            if len(text) > max_body:
                text = text[:max_body] + " ... [Page excerpt]"
            parts.append(header + text + "\n")

    # 3. Tail Pages (Conclusions / Final Takeaways)
    tail_per_page = max(50, tail_budget // len(tail_pages))
    for p in tail_pages:
        text = p.get("text", "").strip()
        header = f"\n[Page {p.get('page_number')} (Conclusions / Summary)]:\n"
        max_body = max(20, tail_per_page - len(header))
        if len(text) > max_body:
            text = text[:max_body] + " ... [Page excerpt]"
        parts.append(header + text + "\n")

    compiled = "".join(parts)
    if len(compiled) + len(notice) > limit:
        compiled = compiled[:max(0, limit - len(notice))]

    compiled += notice
    dt = (time.perf_counter() - t0) * 1000
    return compiled, True, dt


def summarize_document(
    pages: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    max_chars: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates a structured summary of an extracted PDF document.
    Emits structured performance logs and enforces bounded prompt size.

    Returns:
        dict: {
            "success": bool,
            "summary": str,
            "error": Optional[str],
            "model_used": str,
            "timing_ms": Dict[str, float]
        }
    """
    overall_t0 = time.perf_counter()
    resolved_model = model_name or get_model_name()

    # Edge case: No pages or empty text
    non_empty_pages = [p for p in pages if p.get("text", "").strip()]
    if not non_empty_pages:
        total_time = (time.perf_counter() - overall_t0) * 1000
        logger.info(
            "[PERF] Summarize empty document | Pages: 0 | Total: %.2f ms",
            total_time
        )
        return {
            "success": False,
            "summary": "",
            "error": "This document contains no extractable text to summarize.",
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": 0.0, "api_ms": 0.0, "total_ms": total_time}
        }

    raw_char_count = sum(len(p.get("text", "")) for p in non_empty_pages)

    # 1. Prepare and boundedly condense text
    compiled_text, is_condensed, prep_ms = prepare_condensed_text(non_empty_pages, max_chars=max_chars)

    prompt = f"""You are a professional document analysis assistant. Provide a clear, well-structured executive summary of the following PDF document.

Instructions:
1. Executive Overview: Briefly summarize the document's main purpose.
2. Key Findings & Main Points: Detail important points, citing the relevant source page (e.g. [Page 2]).
3. Conclusions & Takeaways: Summarize the final outcomes or recommendations.
4. If the document has insufficient content, state that clearly.

Document Content:
{compiled_text}
"""

    # 2. Execute Gemini API call with latency measurement
    api_t0 = time.perf_counter()
    try:
        active_client = client or create_gemini_client()
        response = active_client.models.generate_content(
            model=resolved_model,
            contents=prompt
        )
        api_ms = (time.perf_counter() - api_t0) * 1000
        total_ms = (time.perf_counter() - overall_t0) * 1000

        logger.info(
            "[PERF] Gemini Summarize Success | Model: %s | Pages: %d | Raw Chars: %d | Prompt Chars: %d | Condensed: %s | Prep: %.2f ms | Gemini API: %.2f ms | Total: %.2f ms",
            resolved_model,
            len(non_empty_pages),
            raw_char_count,
            len(compiled_text),
            is_condensed,
            prep_ms,
            api_ms,
            total_ms
        )

        return {
            "success": True,
            "summary": response.text or "No summary could be generated.",
            "error": None,
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": prep_ms, "api_ms": api_ms, "total_ms": total_ms}
        }
    except Exception as exc:
        api_ms = (time.perf_counter() - api_t0) * 1000
        total_ms = (time.perf_counter() - overall_t0) * 1000

        logger.warning(
            "[PERF] Gemini Summarize Error | Model: %s | Pages: %d | Prep: %.2f ms | Gemini API: %.2f ms | Total: %.2f ms | Error: %s",
            resolved_model,
            len(non_empty_pages),
            prep_ms,
            api_ms,
            total_ms,
            type(exc).__name__
        )

        return {
            "success": False,
            "summary": "",
            "error": _format_error_message(exc),
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": prep_ms, "api_ms": api_ms, "total_ms": total_ms}
        }


def answer_pdf_question(
    pages: List[Dict[str, Any]],
    question: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Answers questions about a PDF, grounded strictly in retrieved passages.
    Returns the answer and source page citations with structured performance timings.

    Returns:
        dict: {
            "success": bool,
            "answer": str,
            "page_citations": List[int],
            "error": Optional[str],
            "model_used": str,
            "timing_ms": Dict[str, float]
        }
    """
    overall_t0 = time.perf_counter()
    resolved_model = model_name or get_model_name()

    cleaned_question = (question or "").strip()
    if not cleaned_question:
        total_time = (time.perf_counter() - overall_t0) * 1000
        return {
            "success": False,
            "answer": "",
            "page_citations": [],
            "evidence_passages": [],
            "error": "Question cannot be empty. Please ask a specific question.",
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": 0.0, "api_ms": 0.0, "total_ms": total_time}
        }

    non_empty_pages = [p for p in pages if p.get("text", "").strip()]
    if not non_empty_pages:
        total_time = (time.perf_counter() - overall_t0) * 1000
        return {
            "success": False,
            "answer": "",
            "page_citations": [],
            "evidence_passages": [],
            "error": "The uploaded document contains no readable text to answer questions.",
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": 0.0, "api_ms": 0.0, "total_ms": total_time}
        }

    # 1. Chunk document & retrieve relevant passages
    prep_t0 = time.perf_counter()
    chunks = chunk_extracted_pages(non_empty_pages, chunk_size=1000, chunk_overlap=150)
    relevant_chunks = retrieve_relevant_chunks(chunks, cleaned_question, top_k=top_k)
    prep_ms = (time.perf_counter() - prep_t0) * 1000

    page_citations = sorted(list(set(c["page_number"] for c in relevant_chunks)))
    evidence_passages = [
        {"page_number": c["page_number"], "text": c["text"], "chunk_id": c.get("chunk_id")}
        for c in relevant_chunks
    ]

    context_blocks = []
    for c in relevant_chunks:
        context_blocks.append(f"[Page {c['page_number']}]:\n{c['text']}")
    context_str = "\n\n---\n\n".join(context_blocks)

    # 2. Grounded Prompt
    prompt = f"""You are a document question-answering assistant. Answer the user's question based STRICTLY and ONLY on the provided PDF passages.

CRITICAL RULES:
1. Ground your answer solely in the passages provided below. Do NOT use outside knowledge or speculate.
2. For every key fact, cite the specific page number in your text (e.g. [Page 3]).
3. If the provided passages do NOT contain enough information or evidence to answer the question, you MUST reply:
   "The uploaded document does not contain enough evidence to answer this question."

Relevant Document Passages:
{context_str}

User Question:
{cleaned_question}
"""

    api_t0 = time.perf_counter()
    try:
        active_client = client or create_gemini_client()
        response = active_client.models.generate_content(
            model=resolved_model,
            contents=prompt
        )
        api_ms = (time.perf_counter() - api_t0) * 1000
        total_ms = (time.perf_counter() - overall_t0) * 1000

        logger.info(
            "[PERF] Gemini QA Success | Model: %s | Retrieved Chars: %d | Citations: %s | Prep: %.2f ms | Gemini API: %.2f ms | Total: %.2f ms",
            resolved_model,
            len(context_str),
            page_citations,
            prep_ms,
            api_ms,
            total_ms
        )

        return {
            "success": True,
            "answer": response.text or "No response generated.",
            "page_citations": page_citations,
            "evidence_passages": evidence_passages,
            "error": None,
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": prep_ms, "api_ms": api_ms, "total_ms": total_ms}
        }
    except Exception as exc:
        api_ms = (time.perf_counter() - api_t0) * 1000
        total_ms = (time.perf_counter() - overall_t0) * 1000

        logger.warning(
            "[PERF] Gemini QA Error | Model: %s | Prep: %.2f ms | Gemini API: %.2f ms | Total: %.2f ms | Error: %s",
            resolved_model,
            prep_ms,
            api_ms,
            total_ms,
            type(exc).__name__
        )

        return {
            "success": False,
            "answer": "",
            "page_citations": page_citations,
            "evidence_passages": [],
            "error": _format_error_message(exc),
            "model_used": resolved_model,
            "timing_ms": {"prep_ms": prep_ms, "api_ms": api_ms, "total_ms": total_ms}
        }
