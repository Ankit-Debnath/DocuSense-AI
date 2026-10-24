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
from pdf_processing.chunker import chunk_extracted_pages, retrieve_relevant_chunks, is_broad_query

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

    if "api_key" in err_str or "unauthenticated" in err_str:
        return "Invalid or missing Google Gemini API key. Please check your GEMINI_API_KEY in your .env file."
    if "not_found" in err_str or "model" in err_str:
        return "The configured Gemini model was not found or is unavailable. Please check GEMINI_MODEL."
    if "invalid_argument" in err_str:
        return "Invalid or missing Google Gemini API key. Please check your GEMINI_API_KEY in your .env file."
    if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
        return "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
    if "deadline" in err_str or "timeout" in err_str or "connection" in err_str:
        return "Network timeout while communicating with Google Gemini. Please verify your connection."

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
    max_chars: Optional[int] = None,
    document_title: Optional[str] = None,
    total_pages: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates a structured, publication-grade executive summary of an extracted PDF document.
    Adheres to formal research report conventions: Document Information, Executive Overview,
    Key Findings (with quantitative tables), Detailed Analysis, Limitations & Considerations,
    and Conclusions.
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

    resolved_title = document_title or "Extracted PDF Document"
    page_count_hint = total_pages or len(non_empty_pages)

    prompt = f"""You are an elite academic researcher and technical document analyst. Provide a formal, publication-grade executive summary of the following PDF document.

Your output must follow this rigorous, professionally organized structure. Adapt the categories to the specific document content without creating empty sections or repeating information.

## Structure to Follow:

# Executive Research & Document Report

## Document Information
- **Title:** {resolved_title} (Extract official title from the document if different)
- **Total Pages:** {page_count_hint} pages
- **Document Type:** [Identify specific type: e.g. Academic Research Paper, Technical Specification, Engineering Whitepaper, Financial Report, Legal Agreement, Course Lecture, or Executive Briefing]
- **Subject / Domain:** [e.g. Computer Science / Machine Learning, Distributed Systems, Corporate Finance, Biomedical Engineering, etc.]

## Executive Overview
Provide a formal, authoritative overview (2-3 structured paragraphs) articulating the document's core purpose, problem statement, methodological scope, and principal contribution. Cite primary source pages (e.g. [Page 1]).

## Key Findings & Core Contributions
Organize the most significant findings into clearly separated thematic subsections tailored to the document's actual content.
For research and technical papers, use categories such as:
### Research Objectives & Problem Formulation
### Proposed Methodology & Architecture
### Experimental Setup & Evaluation Protocol
### Quantitative Results & Empirical Benchmarks
(CRITICAL: Whenever quantitative metrics, experimental figures, accuracy scores, or benchmarks are reported in the text, present them in a clean Markdown table with headers:
| Metric / Benchmark | Reported Result | Experimental Context / Notes |
|---|---:|---|
Preserve exact figures, percentages, error margins, and original units as reported in the document.)
### Technical Contributions & Innovations
(For non-research documents such as contracts, reports, or guides, adapt categories logically to key terms, deliverables, milestones, or obligations).

## Detailed Analysis
Deliver a systematic, in-depth explanation of core mechanisms, theoretical principles, arguments, algorithms, or workflows in logical progression.
Use numbered headings and subheadings to establish a clear hierarchy:
### 1. [First Major Concept / Workflow / Mathematical Model]
### 2. [Second Major Concept / Comparative Mechanism]
### 3. [Third Major Concept / System Component]
Explain the underlying rationale and mechanics with rigor. Include relevant equations in standard LaTeX notation ($...$ inline, $$...$$ display) where present.

## Limitations & Critical Considerations
Distinguish between limitations explicitly acknowledged in the source text and analytical considerations:
- **Author-Acknowledged Limitations:** [Explicitly noted constraints, dataset bounds, or assumptions [Page X]]
- **Analytical & Implementation Considerations:** [Technical prerequisites, scope constraints, computational requirements, or practical caveats inferred from the analysis]

## Conclusion & Strategic Implications
Synthesize the document's final conclusions, academic/practical significance, operational recommendations, and future trajectory.

---

CRITICAL REQUIREMENTS:
1. CITATIONS: Attribute key claims, findings, and data points with precise source page citations (e.g. [Page 2] or [Pages 3–4]).
2. TONE & STYLE: Academic, objective, and authoritative English. Use complete sentences, logical paragraph transitions, and precise technical vocabulary. Avoid filler, exaggerated claims, or conversational language.
3. MATHEMATICAL INTEGRITY: Preserve equations and mathematical formulations using standard LaTeX notation ($...$ and $$...$$).
4. FACTUAL INTEGRITY: Strictly ground every claim in the provided document text. Do NOT fabricate metrics, external benchmarks, or unsubstantiated claims.
5. NO REDUNDANCY: Do not repeat sentences across sections or output empty sections.

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
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Answers questions about a PDF, grounded strictly in retrieved passages.
    Adapts depth and structure based on query scope: simple factual questions receive concise answers;
    analytical questions receive structured academic reports; comprehensive queries receive section-by-section breakdown.
    Returns the answer and source page citations with structured performance timings.

    Returns:
        dict: {
            "success": bool,
            "answer": str,
            "page_citations": List[int],
            "evidence_passages": List[Dict[str, Any]],
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

    # 1. Chunk document & retrieve relevant passages (adaptive top_k)
    prep_t0 = time.perf_counter()
    is_broad = is_broad_query(cleaned_question)
    effective_top_k = (top_k if top_k is not None else (8 if is_broad else 4))

    chunks = chunk_extracted_pages(non_empty_pages, chunk_size=1000, chunk_overlap=150)
    relevant_chunks = retrieve_relevant_chunks(chunks, cleaned_question, top_k=effective_top_k)
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

    # 2. Grounded Prompt with Adaptive Response Structure
    prompt = f"""You are an expert academic research assistant and technical document analyst.
Answer the user's question grounded STRICTLY and ONLY in the provided PDF passages.

CRITICAL FACTUAL RELIABILITY RULES:
1. Strict Grounding: Ground your response strictly and solely in the provided document passages. Do NOT extrapolate beyond the text or speculate.
2. Insufficient Evidence: If the provided passages do NOT contain enough evidence to answer the question, you MUST reply:
   "The uploaded document does not contain enough evidence to answer this question."
3. Page Citations: Cite specific source page numbers (e.g. [Page 4] or [Pages 3–5]) alongside every key claim, figure, or definition they support.

ADAPTIVE RESPONSE STRUCTURE:
Assess the scope and intent of the user's inquiry and structure your response accordingly:

A. For simple, direct, or factual inquiries (e.g. specific numbers, dates, author names, concise definitions):
   - Provide a direct, concise, and focused response in 1-2 paragraphs with the exact page citation [Page X].
   - Do NOT create unnecessary multi-section reports for simple queries.

B. For analytical, multi-part, or explanatory questions (e.g. "How does the training algorithm work?", "Compare model A with model B", "What are the limitations?"):
   - Structure your response cleanly using these numbered sections:
     ### Direct Answer
     A clear, authoritative summary statement directly answering the core inquiry.

     ### Detailed Explanation
     A thorough explanation of mechanisms, theory, and implementation details arranged in logical paragraphs or numbered points.

     ### Key Points & Technical Distinctions
     Bulleted takeaways, formal definitions, formulas, or algorithmic steps. When comparing multiple quantitative metrics, benchmark scores, or configurations reported in the passages, organize them into a clean Markdown table:
     | Metric / Feature | Reported Value / Finding | Notes / Page Citation |
     |---|---:|---|

     ### Examples & Supporting Context (Include only when supported by the text)
     Illustrative examples or specific experimental conditions documented in the source.

     ### Source References
     A concise list summarizing the cited source pages and their relevance.

C. For broad or comprehensive requests (e.g. "Explain this PDF in detail", "Provide a section-by-section breakdown"):
   - Provide a comprehensive, section-by-section analysis covering the document's objectives, architecture/methodology, empirical results, and implications grounded across the retrieved passages.

STYLE & NOTATION REQUIREMENTS:
- Use formal, academic English with precise technical vocabulary. Use complete sentences.
- Preserve mathematical expressions, formulas, and variables in standard LaTeX ($...$ inline, $$...$$ display).
- Expand technical abbreviations on first use when relevant.
- Distinctly differentiate experimentally verified outcomes from authors' hypotheses.

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


def explain_page_content(
    page_number: int,
    page_text: str,
    total_pages: Optional[int] = None,
    document_title: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a structured, technically accurate walkthrough of an individual PDF page.
    Adheres strictly to the page content:
      A. Page Heading: Page X — [Descriptive Title]
      B. Page Overview: Concise formal purpose and contribution of this page
      C. Detailed Explanation: Relevant subsections (Methodology, Equations, Setup, Results Table)
      D. Important Technical Details: Models, parameters, formulas, assumptions
      E. Key Takeaways: 2-5 crisp bullet points
      F. Source Context: Exact page citations, equations/figures/tables/sections referenced

    Returns:
        dict: {
            "success": bool,
            "page_number": int,
            "explanation": str,
            "error": Optional[str],
            "model_used": str,
            "timing_ms": Dict[str, float]
        }
    """
    overall_t0 = time.perf_counter()
    resolved_model = model_name or get_model_name()

    clean_page_text = (page_text or "").strip()
    if not clean_page_text:
        total_time = (time.perf_counter() - overall_t0) * 1000
        return {
            "success": False,
            "page_number": page_number,
            "explanation": "",
            "error": "No extractable text found on this page.",
            "model_used": resolved_model,
            "timing_ms": {"api_ms": 0.0, "total_ms": total_time}
        }

    resolved_title = document_title or "Document"
    total_page_hint = f" of {total_pages}" if total_pages else ""

    prompt = f"""You are an elite academic professor and technical research analyst. Provide a rigorous, technically accurate, and structured explanation of Page {page_number}{total_page_hint} from the document "{resolved_title}".

CORE OBJECTIVE — EXPLAIN IN YOUR OWN WORDS:
1. Explain the page's meaning, mechanisms, concepts, and significance in your own clear words.
2. Do NOT reproduce or copy raw text from the page verbatim.
3. Do NOT perform a trivial paragraph-by-paragraph paraphrase.
4. Do NOT display raw PDF text extracts or verbatim quotes.
5. Ground strictly in the text of Page {page_number}. Do NOT fabricate findings, formulas, citations, or numbers.
6. Preserve exact numerical values, units, variables, and model names as stated on the page.
7. Distinguish between statements made by the authors and external analytical observations.
8. Adapt to the specific content type of the page (e.g. Abstract, Introduction, Related Work, Methodology, Experiments, Results, Figures/Tables, Limitations/Conclusion, or References).
9. If the page contains very little text, only a figure caption, or continues a previous paragraph, explain it concisely without inventing unrelated sections or generating filler.
10. Preserve mathematical notation using LaTeX format ($...$ inline, $$...$$ display).

STRUCTURE YOUR EXPLANATION AS FOLLOWS:

# Page {page_number} — [Descriptive, Accurate Topic Title, e.g. "Abstract and Research Overview" or "Dynamic Channel Gating Methodology"]

## Page Overview
A concise, formal synthesis (2-4 sentences) in your own words explaining the page's central message, its purpose, and its role in the overall document.

## Detailed Explanation
Organize the core content into logically ordered subsections with descriptive headings that explain how concepts and mechanisms work:
(Select only relevant subsections adapted to what actually appears on this page, such as:
### [Core Concept / Research Problem / Proposed Method / System Architecture]
### [Algorithmic Formulation / Mathematical Mechanics]
### [Experimental Configuration / Benchmark Evaluation]
### [Comparative Results & Observations]
(When numerical comparisons or benchmark metrics appear on this page, present them in a clear Markdown table:
| Metric / Parameter | Value / Finding | Context |
|---|---:|---|
)
)

## Important Technical Details
Detail critical technical specifications explicitly documented on this page:
- **Architectural & Algorithmic Components:** [Specific layers, models, gating mechanisms, or procedures]
- **Mathematical Formulations:** [Equations, variables, loss functions in LaTeX ($...$ / $$...$$)]
- **Experimental Parameters & Metrics:** [Datasets, batch sizes, hyperparameters, latency, accuracy with units]
- **Key Assumptions & Constraints:** [Scope bounds noted on this page]

## Key Takeaways
Provide 2 to 4 crisp, high-impact bullet points summarizing the most important concepts and contributions established on this page.

## Source References & Citations
- **Page Number:** Page {page_number}
- **Referenced Elements:** [Explicitly mention any Equation numbers, Figure numbers, Table numbers, or Section numbers appearing on this page, or note "None referenced directly on this page"]

---

Extracted Text from Page {page_number}:
\"\"\"
{clean_page_text}
\"\"\"
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
            "[PERF] Gemini Page Explain Success | Model: %s | Page: %d | Chars: %d | API: %.2f ms | Total: %.2f ms",
            resolved_model,
            page_number,
            len(clean_page_text),
            api_ms,
            total_ms
        )

        return {
            "success": True,
            "page_number": page_number,
            "explanation": response.text or "No explanation could be generated for this page.",
            "error": None,
            "model_used": resolved_model,
            "timing_ms": {"api_ms": api_ms, "total_ms": total_ms}
        }
    except Exception as exc:
        api_ms = (time.perf_counter() - api_t0) * 1000
        total_ms = (time.perf_counter() - overall_t0) * 1000

        logger.warning(
            "[PERF] Gemini Page Explain Error | Model: %s | Page: %d | API: %.2f ms | Total: %.2f ms | Error: %s",
            resolved_model,
            page_number,
            api_ms,
            total_ms,
            type(exc).__name__
        )

        return {
            "success": False,
            "page_number": page_number,
            "explanation": "",
            "error": _format_error_message(exc),
            "model_used": resolved_model,
            "timing_ms": {"api_ms": api_ms, "total_ms": total_ms}
        }
