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
- **Production WSGI Server:** Waitress 3.0.2 (pure-Python, multithreaded production WSGI server, included in requirements)
- **Form Security & CSRF:** Flask-WTF 1.2.2, WTForms 3.3.1
- **GenAI / LLM SDK:** Google GenAI SDK (`google-genai` 2.33.0)
- **Default AI Model:** `gemini-flash-lite-latest` (~3–5s low-latency inference)
- **Embedding Model:** `gemini-embedding-001` (Official Google GenAI document & query embeddings)
- **PDF Extraction:** `pypdf` 6.21.2
- **Database:** SQLite3 (`instance/pdf_analyzer.db`) with parameterized queries
- **Frontend:** Vanilla JavaScript (ES6+), CSS3 custom design system
- **Markdown & Math Engines:** Marked.js 15.0.7, DOMPurify 3.2.4, KaTeX 0.16.21 (locally bundled)
- **Automated Testing:** `pytest` 9.1.1 (Comprehensive unit, integration, and security test suite)

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
│   ├── embedding_client.py   # Google GenAI text embedding service (gemini-embedding-001)
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
| `SECRET_KEY` | `dev-secret-key-change-in-production` (dev only) | Secret key for signing session cookies and CSRF tokens. In production, an explicit key of >=16 characters is strictly required. |
| `FLASK_ENV` | `development` | Environment mode (`development`, `production`, or `testing`). |
| `PRODUCTION` | Unset | Explicit production flag (`true`, `1`, `yes`) that triggers production mode and secret validation. |
| `FLASK_DEBUG` / `DEBUG` | `0` (False) | Enables debug mode for the local development server only. Defaults to False. |
| `HOST` | `127.0.0.1` | Host interface for local development server (`python app.py`). |
| `PORT` | `5000` | Port for local development server (`python app.py`). |
| `ENABLE_HSTS` | `False` | Emits `Strict-Transport-Security` header. Must only be enabled when served over HTTPS. |
| `USE_PROXY_FIX` | `False` | Enables Werkzeug ProxyFix middleware to trust forwarded headers behind a reverse proxy. |
| `PROXY_COUNT` | `1` | Number of trusted upstream proxies when `USE_PROXY_FIX` is enabled (must be a positive integer). |
| `TRUST_INCOMING_REQUEST_ID` | `False` | When True, trusts and sanitizes incoming `X-Request-ID` headers matching `^[a-zA-Z0-9_\-]{1,64}$` from upstream proxy. |
| `DATABASE_PATH` | `instance/pdf_analyzer.db` | Absolute or relative path to the SQLite database file. Parent directories are created automatically if missing. |
| `UPLOAD_FOLDER` | `uploads/` | Absolute or relative path to the directory where uploaded PDFs are stored. Created automatically if missing. |
| `SESSION_COOKIE_SECURE` | Auto (`True` in prod, `False` in dev) | Enforces `Secure` flag on session cookies over HTTPS. Can be explicitly overridden (`true`/`false`). |
| `GEMINI_API_KEY` | `None` | Google Gemini API key used by the `google-genai` SDK. |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Model used for summarization and Q&A (`gemini-flash-lite-latest` recommended for ~3–5s latency; `gemini-flash-latest` also supported). |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Official Google Gemini embedding model used for document and query vector embeddings. |
| `EMBEDDING_BATCH_SIZE` | `50` | Batch size for document chunk embeddings to respect provider API batch ceilings. |
| `CHUNK_SIZE` | `800` | Target maximum character count per text chunk for semantic RAG retrieval. |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks to preserve contextual continuity. |
| `GEMINI_TIMEOUT_SECONDS` | `30` | Client HTTP timeout in seconds to prevent indefinite API hangs. |
| `MAX_SUMMARY_CHARACTERS`| `16000` | Character limit for bounded prompt condensation sent to Gemini. |
| `MAX_PDF_PAGES` | `100` | Maximum permitted pages per uploaded PDF. |
| `MAX_EXTRACTED_CHARACTERS` | `250000` | Maximum permitted extracted characters across all pages. |
| `MAX_QA_HISTORY_TURNS` | `5` | Maximum conversation turns retained in client session cookie. |
| `MAX_QA_QUESTION_CHARS` | `120` | Character truncation limit for questions saved in session history. |
| `MAX_QA_ANSWER_CHARS` | `140` | Character truncation limit for answer excerpts saved in session history. |
| `WTF_CSRF_ENABLED` | `True` | Enforces CSRF token validation across all state-changing endpoints. |
| `MAX_CONTENT_LENGTH` | `10485760` (10 MB) | Maximum permitted upload file size in bytes. |

### Generating a Secure Production Secret
In production environments, a cryptographically random secret must be configured. Generate one using Python's standard `secrets` module:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Configure the generated secret in your production environment variables (e.g., via `.env` or container secrets):
```env
SECRET_KEY=replace-with-a-new-random-secret
```
> **Security Requirement:** Never commit real secrets or `.env` files to version control. The application automatically aborts startup if a production environment uses default, short (<16 chars), or known weak keys.

---

## 8. Running the Application & Production Deployment

### 8.1 Local Development Server (Development Only)
For local development, testing, and debugging, run the built-in Flask development server:
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

> **Warning:** The Flask development server is single-threaded and not designed for production workloads or hostile network exposure. Never use `python app.py` for production deployments.

---

### 8.2 Production Serving with Waitress (Recommended)
For production deployments on Windows or Linux, serve the application using **Waitress** (a pure-Python, multithreaded, production-grade WSGI server pinned in `requirements.txt`):

```powershell
waitress-serve --listen=127.0.0.1:8000 app:app
```
*(Alternatively, run as a module: `python -m waitress --listen=127.0.0.1:8000 app:app`)*

- **WSGI Callable (`app:app`):** References the `app` instance exported directly by [`app.py`](app.py).
- **Loopback Binding (`127.0.0.1:8000`):** Recommended when running behind a local reverse proxy (Nginx, Caddy, or IIS). The application is protected from direct untrusted access.
- **Network Interface Binding (`0.0.0.0:8000`):** Binds to all network interfaces. Only use this if the host is protected by an external firewall or cloud network security group, as direct exposure lacks TLS encryption unless terminated upstream.

> **Note on Gunicorn:** Gunicorn is **not** included in the project's default `requirements.txt` because it relies on POSIX system calls and does not run natively on Windows. If deploying strictly to Linux and you prefer Gunicorn, install it separately into your virtual environment (`pip install gunicorn`) and run:
> ```bash
> gunicorn -w 4 -b 127.0.0.1:8000 app:app
> ```

---

### 8.3 Production Deployment Step-by-Step Guide

Follow these steps to deploy DocuSense-AI in a production environment (single-machine VPS or containerized host):

#### Step 1: Verify Python Runtime
Ensure Python 3.10 or higher is installed (tested and verified on Python 3.14.3):
```bash
python3 --version
```

#### Step 2: Set Up an Isolated Virtual Environment
Create and activate a dedicated virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Pinned Dependencies
Install dependencies from `requirements.txt`:
```bash
python3 -m pip install -r requirements.txt
```

#### Step 4: Configure Production Environment Variables
Set production environment variables in your deployment environment (or via a protected `.env` file):
```env
# Mode and Security
FLASK_ENV=production
SECRET_KEY=replace-with-a-new-random-secret
SESSION_COOKIE_SECURE=true

# Reverse Proxy Integration
USE_PROXY_FIX=true
PROXY_COUNT=1
ENABLE_HSTS=true

# Persistent Volume Storage Mounts
UPLOAD_FOLDER=/var/data/uploads
DATABASE_PATH=/var/data/pdf_analyzer.db

# API Credentials
GEMINI_API_KEY=your_production_gemini_api_key
```

#### Step 5: Configure Reverse Proxy (e.g. Nginx)
Place the application behind a reverse proxy to terminate TLS (HTTPS), buffer slow clients, and enforce rate limits:
```nginx
server {
    listen 443 ssl http2;
    server_name docusense.example.com;

    ssl_certificate /etc/letsencrypt/live/docusense.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/docusense.example.com/privkey.pem;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }
}
```

#### Step 6: Configure Storage Permissions & Persistence
- Ensure persistent storage directories exist and are owned by the application service account:
  ```bash
  mkdir -p /var/data/uploads /var/data
  chmod 700 /var/data/uploads
  ```
- **Crucial:** In container platforms (Render, Railway, Fly.io), `/var/data` must be mounted as a persistent disk volume. Ephemeral filesystems will wipe stored PDFs, SQLite embeddings, and summary caches upon container redeployment.

#### Step 7: Start the Production Server
Start Waitress bound to loopback:
```bash
waitress-serve --listen=127.0.0.1:8000 app:app
```

---

### 8.4 SQLite Concurrency, Backup Safety, & Limitations

- **Single-Machine Architecture:**
  The application utilizes SQLite in Write-Ahead Logging (`WAL`) mode with a 15-second busy timeout. This architecture delivers high-concurrency reading and writing on a **single machine** or single container instance with local disk access.
- **Multi-Node Limitation:**
  Do **not** attempt to run multiple concurrent application worker instances sharing an SQLite database file over networked filesystems (NFS, SMB/CIFS, AWS EFS). Network filesystem locks are unreliable and will cause database corruption or lock errors under SQLite WAL mode.
- **Deployment Status:**
  The application has not been deployed to a distributed multi-node cluster or subjected to high-throughput load testing. Single-instance production deployments with local persistent storage are the intended operational boundary.
- **Online Backup Procedure:**
  In WAL mode, writes are staged in `pdf_analyzer.db-wal`. Do not copy `pdf_analyzer.db` alone while the server is active. Instead, use SQLite's safe online backup API or CLI:
  ```bash
  sqlite3 /var/data/pdf_analyzer.db ".backup /var/data/backups/backup_$(date +%Y%m%d_%H%M%S).db"
  ```

---

### 8.5 Operational Health Checks & Monitoring
Configure platform health probes using the built-in observability endpoints:
- **Liveness Probe (`GET /healthz`):** Verifies the Flask process is running and responsive. (Probe interval: 10–30s).
- **Readiness Probe (`GET /ready`):** Verifies the application can execute `SELECT 1;` against the configured SQLite database. (Probe interval: 10–30s).
- **Request Correlation:** All incoming requests are assigned a unique `X-Request-ID` header, correlated with access and error logs for end-to-end diagnostics.

---

## 9. Running the Automated Test Suite

The project includes a comprehensive unit, integration, and security test suite covering database queries, SQLite WAL concurrency, PDF parsing, Gemini prompt condensation, CSRF enforcement, open redirects, path traversal defense, HTTP security headers, and production deployment safety.

**All automated tests mock external Gemini API interactions and consume zero API quota or credits.**

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

## 11. Semantic & Hybrid RAG Retrieval Architecture

PDF Analyzer implements a custom, local-first hybrid Retrieval-Augmented Generation (RAG) architecture:

### A. Semantic Vector Retrieval
- **Embedding Generation:** Queries and document chunks are embedded using Google's official text embedding model (`gemini-embedding-001`) with specific task types (`RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`).
- **Vector Storage:** Embeddings are stored directly inside SQLite's `document_chunks` table as compact, little-endian IEEE 754 32-bit float BLOBs (`struct.pack("<f")`), cutting storage consumption by 50% compared to 64-bit doubles while maintaining full float32 precision.
- **Cosine Similarity:** Similarity between query vectors and document chunk vectors is computed in pure Python with mathematical validation:
  $$\operatorname{cosine}(A,B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Scoping & Integrity:** Vector search is strictly document-scoped (never compares or retrieves chunks from other documents). Model identifiers are normalized, and dimension mismatches or corrupted BLOBs are safely caught and skipped without aborting the user request.

### B. Reciprocal Rank Fusion (RRF)
Rather than naively mixing raw cosine scores and BM25 term scores, PDF Analyzer unifies retrieval using **Reciprocal Rank Fusion (RRF)**:
$$RRF(d) = \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}$$
where $k = 60$ and $\operatorname{rank}_r(d)$ is 1-based.
- **Retriever A (Lexical):** Term frequency, stopword-filtered keyword matching, and document distribution scoring.
- **Retriever B (Semantic):** Cosine similarity over stored dense embedding vectors.
- **Fusion Logic:** Chunks identified in both retrieval methods receive composite reciprocal boosts. Duplicates are eliminated while original PDF page citations and metadata are strictly preserved.

### C. On-Demand Indexing & Concurrency Safety
- **Lazy Indexing:** When a user queries a document that has not yet been indexed, `ensure_document_indexed()` dynamically extracts text, chunks pages, generates embeddings in bounded batches (`EMBEDDING_BATCH_SIZE = 50`), and commits chunks atomically to SQLite.
- **Double-Checked Concurrency Lock:** Simultaneous queries on the same document are serialized using a lightweight per-document mutex (`threading.Lock()`), preventing redundant Gemini embedding API requests.
- **Index Reuse:** Once indexed, all subsequent queries immediately reuse stored embeddings without re-extracting or re-embedding.

### D. Graceful Fallback Behavior
- If the Google Gemini embedding API experiences network timeouts, rate limits (HTTP 429), or service errors, the pipeline automatically and seamlessly falls back to keyword-based lexical retrieval.
- If a query dimension or model mismatch occurs, the request automatically falls back to keyword retrieval. When embedding generation is unavailable, the system falls back to lexical retrieval. Answer generation still depends on the availability of the configured Gemini generation model.

---

## 12. Security Architecture & Current Limitations

### Implemented Protections:
- **CSRF Defense:** `Flask-WTF` validates tokens on every form and JSON POST request. Missing or invalid tokens return HTTP 400.
- **Open Redirect Protection:** Centralized `is_safe_redirect_url()` validator strictly allows only safe internal relative paths, rejecting protocol-relative URLs (`//evil.com`) and backslash tricks.
- **Path Traversal Defense:** `_validate_stored_filename()` verifies that document references are strict basenames located within the uploads directory.
- **HTTP Security Headers & CSP:** Emits `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and a restrictive Content Security Policy (`script-src 'self'`).
- **Fail-Closed HTML Sanitization:** DOMPurify sanitizes Markdown output; fails closed to escaped plain text if sanitizer is unavailable.
- **Production Secret Validation:** Startup fails fast if production secrets are missing, weak, or default.
- **Secret Safety:** `.env`, `.env.local`, and `instance/` are gitignored. API keys are loaded via environment variables and never written to application logs.

### Current Architectural Limitations:
- **Single-Tenant / Local Tool:** The application does not currently implement multi-user accounts, passwords, or document access control lists.
- **Local Network Exposure:** Anyone who accesses the application port (if bound externally without authentication or firewalling) can view documents listed in `/history`.
- **Reverse Proxy Requirement:** For production deployments, a reverse proxy (e.g. Nginx, Caddy) is required for TLS termination, rate limiting, and public traffic management.

---

## 13. Health Checks & Operational Observability

DocuSense-AI includes native health check endpoints and structured request observability designed for container orchestrators (Kubernetes, Docker Swarm), platform-as-a-service providers (Render, Railway, Fly.io, AWS ECS), and monitoring systems:

### A. Liveness vs. Readiness Probes

| Endpoint | Probe Type | Purpose | Dependencies Checked | Failure Response |
| :--- | :--- | :--- | :--- | :--- |
| `/healthz` | **Liveness** | Confirms the Flask process is alive, running, and accepting HTTP requests. Orchestrators restart crashed or deadlock containers when this fails. | None (pure in-memory JSON response; zero DB queries, zero external network calls, zero file I/O). | 5xx / Connection refused |
| `/ready` | **Readiness** | Confirms the application is ready to process incoming traffic by verifying database connectivity. Orchestrators halt traffic routing to the instance when this fails. | SQLite database access via configured `DATABASE_PATH` using lightweight read-only `SELECT 1;`. | `503 Service Unavailable` with `checks.database: error` |

**Difference between Liveness and Readiness:**
- `/healthz` tests only whether the Python process itself is functioning. If the database is temporarily locked or slow, the liveness probe remains green to avoid unnecessary container restarts.
- `/ready` tests whether essential local infrastructure (SQLite) is functional. If the database cannot be opened or queried, `/ready` returns HTTP 503 so upstream load balancers refrain from routing user traffic to that container until connectivity is restored. Neither endpoint invokes external Gemini APIs.

### B. Health-Check Response Payloads

**Liveness (`GET /healthz` or `HEAD /healthz`):**
```json
{
  "status": "ok"
}
```

**Readiness Success (`GET /ready` or `HEAD /ready`):**
```json
{
  "status": "ready",
  "checks": {
    "database": "ok"
  }
}
```

**Readiness Failure (`GET /ready` when database fails):**
```json
{
  "status": "not_ready",
  "checks": {
    "database": "error"
  }
}
```
*Note: Responses are strictly sanitized and never disclose internal filesystem paths, SQL statements, database filenames, exception tracebacks, or credentials.*

### C. Request ID Generation and Propagation

- Every incoming HTTP request is assigned a unique, cryptographically random request ID (`uuid.uuid4().hex`).
- The ID is stored in Flask's `g.request_id` context and propagated to downstream logging functions.
- The `X-Request-ID` HTTP response header is attached to all responses, including successful views (200), redirects (302), client errors (400, 404, 405), and unhandled server errors (500).
- **Security Trust Model:** Client-supplied `X-Request-ID` headers are untrusted by default. The server generates an authoritative ID. If `TRUST_INCOMING_REQUEST_ID=True` is explicitly configured behind a trusted edge proxy, incoming values are strictly sanitized against `^[a-zA-Z0-9_\-]{1,64}$` to eliminate CRLF newline injection and log-forging attacks.

### D. Structured Request Logging

The application logs every request upon completion using Python's standard `logging` library (`pdf_analyzer.request`):

- **Logged Fields:**
  - `request_id`: Unique correlation ID matching the `X-Request-ID` header.
  - `method`: HTTP method (e.g. `GET`, `POST`).
  - `path`: Safe request URL path (e.g. `/healthz`, `/ask`).
  - `status`: HTTP response status code.
  - `duration_ms`: Request execution latency in milliseconds (strictly non-negative).
  - Severity level: `INFO` for status < 400, `WARNING` for 4xx client errors, `ERROR` for 5xx server errors.

- **Example Log Output:**
  ```text
  2026-11-03 12:33:23,635 [INFO] pdf_analyzer.request: request_id=906c10b7c8d149029328fa3178b2583d method=GET path=/healthz status=200 duration_ms=0.19ms
  2026-11-03 12:33:23,639 [INFO] pdf_analyzer.request: request_id=6b0687120b6b46c582071741630462c2 method=GET path=/ready status=200 duration_ms=2.70ms
  2026-11-03 12:35:14,292 [ERROR] pdf_analyzer.request: request_id=ce7f6b7e86eb40e9acd6adee88def11b unhandled_exception=RuntimeError
  ```

- **Sensitive Data Exclusion:**
  To reduce unnecessary exposure of sensitive information, the application avoids logging:
  - **Query strings** are never logged (`request.path` is used instead of `request.full_path`).
  - **Request bodies** and form fields are never logged.
  - **Uploaded PDF contents** and parsed text are never logged.
  - **User questions** and AI generated answers are never logged.
  - **Session cookies**, CSRF tokens, and authorization headers are never logged.
  - **API keys** and secrets are never logged.

---

## 14. Future Improvements (Roadmap)

The following capabilities represent future enhancements and are not part of the current single-user release:
- **Multi-User Authentication & Tenancy:** User registration, password hashing (`bcrypt`), and private document ownership.
- **Optical Character Recognition (OCR):** Integration with Tesseract OCR or Gemini multimodal vision for scanned, image-only PDFs.
- **Vector Database Integration:** Embedding storage using ChromaDB or FAISS for multi-document semantic retrieval.
- **Persistent Q&A Conversation Threads:** Storing multi-turn Q&A sessions in SQLite rather than transient Flask session cookies.
- **Asynchronous Task Queue:** Celery/Redis background worker processing for large (100+ page) documents.
- **Export Capabilities:** Exporting generated executive summaries as formatted PDF or Markdown files.
