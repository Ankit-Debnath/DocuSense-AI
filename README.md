# PDF Analyzer — GenAI-Powered Document Assistant

A full-stack, local-first web application built with **Python**, **Flask**, **SQLite**, and the official **Google GenAI SDK** (`google-genai`). PDF Analyzer enables users to extract text page-by-page from PDF documents, generate structured executive summaries, and ask document-grounded questions with verified source-page citations.

---

## 1. Project Overview & Problem Statement

Reading and extracting actionable intelligence from dense, multi-page PDF documents—such as research papers, financial reports, technical documentation, and contracts—is tedious and time-consuming. Users often face:
- Repetitive costs and high latency from re-summarizing documents through cloud APIs.
- Ungrounded LLM "hallucinations" that speculate beyond the actual document text.
- Security and privacy risks from uploading sensitive files to opaque third-party SaaS platforms.

**PDF Analyzer** solves these challenges by combining:
- **Local PDF processing** using `pypdf`.
- **Intelligent prompt condensation** preserving introductions, core sections, and conclusions within configurable token budgets.
- **Persistent SQLite summary caching** ensuring that documents are summarized only once, delivering instant (< 2 ms) re-reads and zero redundant API costs.
- **Passage-grounded Q&A** that extracts and ranks relevant chunks, citing source pages and refusing to speculate when evidence is absent.
- **Enterprise-grade local security**, including CSRF protection, open redirect defense, path traversal prevention, and local vendor-bundled XSS sanitization.

---

## 2. Key Features

- **Robust PDF Upload & Validation:**
  - Enforces 10 MB maximum file size via Werkzeug request handling.
  - Validates `.pdf` extension and checks binary header magic bytes (`%PDF-`).
  - Generates sanitized, collision-free filenames (`<uuid>_<safe_name>.pdf`) to eliminate overwrite conflicts.
- **Page-by-Page Extraction:**
  - Extracts text, page numbers, and character metrics using `pypdf`.
  - Handles password-protected, corrupted, or 0-page PDFs gracefully.
  - Provides friendly notices when scanned/rasterized image-only PDFs require OCR.
- **AI Document Summarization:**
  - Generates structured executive overviews, key findings, and takeaways via Google Gemini.
  - Employs balanced text condensation (40% head, 20% middle, 40% tail) for documents exceeding budget limits (`MAX_SUMMARY_CHARACTERS`).
- **High-Performance SQLite Summary Caching:**
  - Persists successful summaries in `document_summaries` table.
  - Instant cache hits on document re-open without invoking external Gemini API calls.
  - Supports explicit "Regenerate Summary" with prior cache invalidation.
  - Failed API calls (rate limits, timeouts) are never stored in cache.
- **Grounded Document Q&A:**
  - Chunks document pages with configurable overlap (1,000 characters with 150-character overlap).
  - Retrieves relevant passages using query term frequency and page mapping.
  - Constrains Gemini to answer strictly using retrieved context, citing exact source pages (`[Page X]`).
  - Interactive citation pills allow users to jump directly to and highlight the corresponding page accordion.
- **Polished, Responsive UI:**
  - Renders Markdown headings, lists, bold text, and tables with responsive styling.
  - Renders mathematical notation ($$Display Math$$ and $Inline Math$) via KaTeX.
  - Clean collapsible accordions for page-by-page content and grounded evidence cards.
- **Robust Security & Defense-in-Depth:**
  - **CSRF Protection:** Initialized globally via `Flask-WTF` on all state-changing endpoints.
  - **Open Redirect Defense:** Centralized `is_safe_redirect_url()` validator blocks protocol-relative (`//evil.com`) and backslash tricks.
  - **Path Traversal Defense:** Canonical path and `os.path.commonpath` verification prevents unauthorized disk file access.
  - **Fail-Closed HTML Sanitization:** DOMPurify sanitizes Markdown output; fails closed to escaped plain text if sanitizer is unavailable.
  - **Local Vendor Bundling:** Marked.js, DOMPurify, and KaTeX are bundled locally under `static/vendor/` with zero external CDN dependencies.

---

## 3. Technology Stack

- **Backend Framework:** Python 3.10+ (tested on Python 3.14.3), Flask 3.1.3, Werkzeug 3.1.9
- **Form Security & CSRF:** Flask-WTF 1.2.2, WTForms 3.3.1
- **GenAI / LLM SDK:** Google GenAI SDK (`google-genai` 2.33.0)
- **Default AI Model:** `gemini-flash-lite-latest` (~3–5s low-latency inference)
- **PDF Extraction:** `pypdf` 6.21.2
- **Database:** SQLite3 (`instance/pdf_analyzer.db`) with parameterized queries
- **Frontend:** Vanilla JavaScript (ES6+), CSS3 custom design system
- **Markdown & Math Engines:** Marked.js 15.0.7, DOMPurify 3.2.4, KaTeX 0.16.21 (locally bundled)
- **Automated Testing:** `pytest` 9.1.1 (53 automated tests)

---

## 4. Project Directory Structure

```text
7th_project/
├── .env.example              # Environment variables template with placeholder values
├── .gitignore                # Excludes secrets, databases, venv, and uploaded PDFs
├── app.py                    # Flask application factory, error handlers, entry point
├── config.py                 # Configuration class (CSRF, paths, upload & Gemini limits)
├── requirements.txt          # Pinned Python package dependencies
├── ai/
│   ├── __init__.py
│   ├── analyzer.py           # Summarization, prompt condensation, and grounded Q&A
│   └── llm_client.py         # Google GenAI client initialization, timeouts, key loading
├── database/
│   ├── __init__.py
│   ├── db.py                 # SQLite connection lifecycle and table schema definitions
│   └── models.py             # Parameterized database queries and summary cache operations
├── pdf_processing/
│   ├── __init__.py
│   ├── chunker.py            # Text chunking with overlap and relevance scoring
│   ├── extractor.py          # Page-by-page text extraction with pypdf
│   └── validator.py          # PDF extension, header signature, and filename sanitization
├── routes/
│   ├── __init__.py
│   ├── pdf_routes.py         # Active route blueprint (upload, summarize, ask, history, save-key)
│   ├── ai_routes.py          # Scaffold module for future AI microservices
│   └── main_routes.py        # Scaffold module for future top-level page views
├── static/
│   ├── css/
│   │   └── style.css         # Modern, responsive application stylesheet
│   ├── js/
│   │   └── main.js           # Client-side Markdown rendering, KaTeX, jump pills, loading states
│   └── vendor/               # Bundled local vendor assets (no external CDN required)
│       ├── auto-render.min.js
│       ├── katex.min.css
│       ├── katex.min.js
│       ├── marked.min.js
│       ├── purify.min.js
│       └── fonts/            # KaTeX woff2 font files
├── templates/
│   ├── base.html             # Base layout with navbar, flash alerts, and CSRF meta tag
│   ├── index.html            # Upload page with drag/select form
│   ├── history.html          # Upload history table showing processed documents
│   └── result.html           # Full document view: stats, AI summary, Q&A, and page accordions
├── tests/
│   ├── test_database.py      # Database initialization, records, and caching tests
│   ├── test_extractor.py     # PDF extraction, corrupted files, and blank pages tests
│   ├── test_gemini.py        # Mocked summarization, Q&A grounding, latency, and cache route tests
│   ├── test_security_fixes.py# Open redirect, CSRF protection, and local vendor assets tests
│   └── test_upload.py        # File validation, size limits, and safe filename tests
└── uploads/
    └── .gitkeep              # Runtime directory for uploaded PDFs (files excluded by .gitignore)
```

---

## 5. Prerequisites

- **Operating System:** Windows 10/11 (macOS / Linux compatible)
- **Python:** Version 3.10, 3.11, 3.12, 3.13, or 3.14 (tested and verified on Python 3.14.3)
- **Shell:** Windows PowerShell or Command Prompt
- **Google Gemini API Key:** Obtainable free of charge from [Google AI Studio](https://aistudio.google.com/)

---

## 6. Installation & Setup (Windows PowerShell)

Follow these steps from an active PowerShell window:

### Step 1: Navigate to Project Directory
```powershell
cd D:\machine_learning_projects\7th_project
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### Step 3: Install Pinned Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to create your local `.env` configuration:
```powershell
Copy-Item .env.example .env
```

Open `.env` in your text editor and set your Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSyYourActualGeminiApiKeyHere
```

> **Note:** The application will start without an API key, but AI features (Summarization and Q&A) require a valid `GEMINI_API_KEY`. You can also enter and save your API key directly through the web UI on the results page.

---

## 7. Configuration Reference

The application reads configuration from `.env` and `config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SECRET_KEY` | `dev-secret-key-change-in-production` | Secret key used to sign session cookies and CSRF tokens. |
| `GEMINI_API_KEY` | `None` | Google Gemini API key used by the `google-genai` SDK. |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Model used for summarization and Q&A (`gemini-flash-lite-latest` recommended for ~3–5s latency; `gemini-flash-latest` also supported). |
| `GEMINI_TIMEOUT_SECONDS` | `30` | Client HTTP timeout in seconds to prevent indefinite API hangs. |
| `MAX_SUMMARY_CHARACTERS`| `16000` | Character limit for bounded prompt condensation sent to Gemini. |
| `WTF_CSRF_ENABLED` | `True` | Enforces CSRF token validation across all state-changing endpoints. |
| `MAX_CONTENT_LENGTH` | `10485760` (10 MB) | Maximum permitted upload file size in bytes. |

---

## 8. Running the Application

With your virtual environment activated, start the local Flask development server:
```powershell
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### Application Workflow:
1. **Upload:** Select any `.pdf` document up to 10 MB on the homepage and click **Upload PDF**.
2. **Review Metrics:** View total extracted pages, character counts, and expandable page-by-page content.
3. **Generate Summary:** Click **Generate Summary** to call Gemini and generate a structured overview with key findings.
4. **Document Q&A:** Ask targeted questions in the chat input. Answers cite specific pages and display expandable source context passages.
5. **History:** Click **View Document History** in the navigation bar to see a log of all processed documents.

---

## 9. Running the Automated Test Suite

The project includes 53 comprehensive unit and integration tests covering database queries, PDF parsing, text chunking, Gemini API error handling, CSRF enforcement, open redirects, and path traversal defense.

**All automated tests mock Gemini API interactions and consume zero API quota or credits.**

Execute the test suite using `pytest`:
```powershell
.\venv\Scripts\python.exe -m pytest -v
```

For quiet/compact output:
```powershell
.\venv\Scripts\python.exe -m pytest -q
```

---

## 10. How Summary Caching Works

To eliminate redundant API calls and prevent unexpected billing:
1. When a PDF is uploaded, a collision-free safe filename is generated (e.g. `a1b2c3d4_report.pdf`).
2. When the user requests a summary, the route queries SQLite table `document_summaries` for `filename = a1b2c3d4_report.pdf`.
3. **Cache Hit:** If found, the summary is loaded from SQLite in < 2 ms. The UI displays a `⚡ Cached in SQLite` badge and bypasses Gemini completely.
4. **Cache Miss:** If not cached, the document text is condensed, Gemini is queried, and the successful summary is committed to SQLite.
5. **Regeneration:** Clicking **Regenerate Summary** sends `regenerate=true`, which explicitly deletes the existing cached record before querying Gemini for fresh analysis.
6. **Failure Safety:** If an API call fails (e.g., quota exceeded), the failure is never written to cache.

---

## 11. Security Architecture & Current Limitations

### Implemented Protections:
- **CSRF Defense:** `Flask-WTF` validates tokens on every form and JSON POST request. Missing or invalid tokens return HTTP 400.
- **Open Redirect Protection:** Centralized `is_safe_redirect_url()` validator strictly allows only safe internal relative paths, rejecting protocol-relative URLs (`//evil.com`) and backslash tricks.
- **Path Traversal Defense:** `_validate_stored_filename()` verifies that document references are strict basenames located within the uploads directory.
- **XSS Prevention:** Markdown output is sanitized using DOMPurify with strict tag/attribute whitelists. If DOMPurify is absent, the system fails closed by displaying safely escaped plain text.
- **Secret Safety:** `.env` and `instance/` are gitignored. API keys are loaded via environment variables and never written to application logs.

### Current Architectural Limitations:
- **Single-Tenant / Local Tool:** The application does not currently implement multi-user accounts, passwords, or document access control lists.
- **Local Network Exposure:** Anyone who accesses `http://127.0.0.1:5000` (or the host's IP if bound externally) can view documents listed in `/history`.
- **HTTP Development Server:** Designed for local development; requires a TLS-terminating reverse proxy (e.g. Nginx, Caddy) for production deployment.

---

## 12. Future Improvements (Roadmap)

The following capabilities represent future enhancements and are not part of the current single-user release:
- **Multi-User Authentication & Tenancy:** User registration, password hashing (`bcrypt`), and private document ownership.
- **Optical Character Recognition (OCR):** Integration with Tesseract OCR or Gemini multimodal vision for scanned, image-only PDFs.
- **Vector Database Integration:** Embedding storage using ChromaDB or FAISS for multi-document semantic retrieval.
- **Persistent Q&A Conversation Threads:** Storing multi-turn Q&A sessions in SQLite rather than transient Flask session cookies.
- **Asynchronous Task Queue:** Celery/Redis background worker processing for large (100+ page) documents.
- **Export Capabilities:** Exporting generated executive summaries as formatted PDF or Markdown files.
