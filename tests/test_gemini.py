import io
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pypdf
from app import create_app
from config import Config
from ai.llm_client import create_gemini_client, get_model_name, save_api_key, get_api_key
from ai.analyzer import summarize_document, answer_pdf_question, prepare_condensed_text, explain_page_content
from pdf_processing.chunker import chunk_extracted_pages, retrieve_relevant_chunks


class TestGeminiConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    GEMINI_API_KEY = "mock_test_api_key_12345"
    GEMINI_MODEL = "gemini-2.5-flash"


class GeminiAITestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(TestGeminiConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def get_csrf_token(self):
        """Fetches a valid CSRF token from the application homepage."""
        res = self.client.get("/")
        m = re.search(r'name="csrf_token"\s+value="([^"]+)"', res.data.decode("utf-8"))
        return m.group(1) if m else ""

    def test_client_initialization_with_key(self):
        """Test that create_gemini_client initializes properly when a key is provided."""
        client = create_gemini_client(api_key="mock_key_for_test")
        self.assertIsNotNone(client)

    def test_client_raises_value_error_when_key_missing(self):
        """Test that creating a client without an API key raises a clear ValueError."""
        with patch("ai.llm_client.get_api_key", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                create_gemini_client(api_key=None)
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_chunking_preserves_page_numbers_and_overlap(self):
        """Test that chunker creates overlapping chunks tagged with page numbers."""
        pages = [
            {"page_number": 1, "text": "A" * 1200, "char_count": 1200},
            {"page_number": 2, "text": "Page two text content.", "char_count": 22}
        ]
        chunks = chunk_extracted_pages(pages, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(chunks), 2)
        # Check that page numbers are preserved
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[-1]["page_number"], 2)

    def test_passage_retrieval_ranking(self):
        """Test that relevant passages are retrieved based on query keywords."""
        chunks = [
            {"chunk_id": 1, "page_number": 1, "text": "The company reported quarterly revenue of 10 million dollars."},
            {"chunk_id": 2, "page_number": 2, "text": "The weather in Seattle is rainy during autumn."},
            {"chunk_id": 3, "page_number": 3, "text": "Revenue growth was driven by cloud software sales."}
        ]
        retrieved = retrieve_relevant_chunks(chunks, query="What was the company revenue?", top_k=2)
        self.assertEqual(len(retrieved), 2)
        page_nums = [c["page_number"] for c in retrieved]
        self.assertIn(1, page_nums)
        self.assertIn(3, page_nums)
        self.assertNotIn(2, page_nums)

    def test_summarize_empty_document(self):
        """Test that an empty document is handled gracefully without calling the API."""
        pages = [{"page_number": 1, "text": "   ", "char_count": 0}]
        mock_client = MagicMock()
        result = summarize_document(pages, client=mock_client)
        self.assertFalse(result["success"])
        self.assertIn("no extractable text", result["error"].lower())
        mock_client.models.generate_content.assert_not_called()

    def test_summarize_document_success_mocked(self):
        """Test successful document summarization with mocked Gemini response."""
        pages = [
            {"page_number": 1, "text": "Antigravity is an AI pair programming agent.", "char_count": 42}
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1. Overview: AI pair programmer.\n2. Key Points: Assists coding [Page 1]."
        mock_client.models.generate_content.return_value = mock_response

        result = summarize_document(pages, client=mock_client, model_name="gemini-2.5-flash")
        self.assertTrue(result["success"])
        self.assertIn("AI pair programmer", result["summary"])
        self.assertEqual(result["model_used"], "gemini-2.5-flash")
        mock_client.models.generate_content.assert_called_once()

    def test_summarize_document_rate_limit_error_mocked(self):
        """Test graceful translation of quota/rate-limit exceptions."""
        pages = [{"page_number": 1, "text": "Valid document text.", "char_count": 20}]
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded")

        result = summarize_document(pages, client=mock_client)
        self.assertFalse(result["success"])
        self.assertIn("rate limit or quota exceeded", result["error"])

    def test_answer_question_grounding_and_citations_mocked(self):
        """Test question answering returns answer and page citations from retrieved passages."""
        pages = [
            {"page_number": 1, "text": "Project Apollo landed on the moon in July 1969.", "char_count": 48},
            {"page_number": 2, "text": "Unrelated chapter about ocean exploration.", "char_count": 42}
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Project Apollo landed on the moon in July 1969 [Page 1]."
        mock_client.models.generate_content.return_value = mock_response

        result = answer_pdf_question(pages, "When did Apollo land on the moon?", client=mock_client)
        self.assertTrue(result["success"])
        self.assertIn("July 1969", result["answer"])
        self.assertEqual(result["page_citations"], [1])

    def test_answer_question_insufficient_evidence_mocked(self):
        """Test question answering when the document lacks evidence."""
        pages = [
            {"page_number": 1, "text": "This document only talks about baking bread.", "char_count": 42}
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The uploaded document does not contain enough evidence to answer this question."
        mock_client.models.generate_content.return_value = mock_response

        result = answer_pdf_question(pages, "What is the capital of France?", client=mock_client)
        self.assertTrue(result["success"])
        self.assertIn("does not contain enough evidence", result["answer"])

    def test_save_api_key_helper(self):
        """Test that save_api_key updates memory and active environment."""
        with patch("builtins.open", unittest.mock.mock_open(read_data="GEMINI_API_KEY=\n")):
            success = save_api_key("new_mock_key_999")
            self.assertTrue(success)
            self.assertEqual(os.environ.get("GEMINI_API_KEY"), "new_mock_key_999")

    def test_web_routes_ask_and_summarize(self):
        """Test /ask and /summarize web endpoints using mocked Gemini responses."""
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length 44 >> stream
BT
/F1 24 Tf
100 700 Td
(Hello World from PDF) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000262 00000 n 
0000000341 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
434
%%EOF"""
        test_pdf_path = os.path.join(self.test_upload_dir, "test_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        # 1. Test /summarize route with mocked summarize_document
        with patch("routes.pdf_routes.summarize_document") as mock_summarize:
            mock_summarize.return_value = {
                "success": True,
                "summary": "Mocked executive summary.",
                "error": None,
                "model_used": "gemini-2.5-flash"
            }
            res = self.client.post("/summarize", data={
                "stored_filename": "test_doc.pdf",
                "original_filename": "doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Mocked executive summary", res.data)

        # 2. Test /ask route with mocked answer_pdf_question
        with patch("routes.pdf_routes.answer_pdf_question") as mock_ask:
            mock_ask.return_value = {
                "success": True,
                "answer": "Mocked answer with grounding [Page 1].",
                "page_citations": [1],
                "error": None,
                "model_used": "gemini-2.5-flash"
            }
            res = self.client.post("/ask", data={
                "stored_filename": "test_doc.pdf",
                "original_filename": "doc.pdf",
                "question": "What is this about?",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Mocked answer with grounding", res.data)
            self.assertIn(b"Page 1", res.data)

        # 3. Test /save-key route
        with patch("routes.pdf_routes.save_api_key") as mock_save:
            res = self.client.post("/save-key", data={
                "gemini_api_key": "new_mock_key_abc",
                "return_url": "/",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 302)
            mock_save.assert_called_once_with("new_mock_key_abc")

    def test_summarize_route_uses_sqlite_cache(self):
        """Test that /summarize caches generated summaries and avoids repeated Gemini calls."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Cached Test Doc) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "cache_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.summarize_document") as mock_summarize:
            mock_summarize.return_value = {
                "success": True,
                "summary": "First-time generated summary.",
                "error": None,
                "model_used": "gemini-flash-lite-latest"
            }

            # 1. First call: Cache miss, calls summarize_document
            res1 = self.client.post("/summarize", data={
                "stored_filename": "cache_doc.pdf",
                "original_filename": "cache_doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res1.status_code, 200)
            self.assertIn(b"First-time generated summary", res1.data)
            self.assertIn(b'id="formatted-summary"', res1.data)
            self.assertIn(b'id="summary-raw-content"', res1.data)
            self.assertIn(b'marked.min.js', res1.data)
            self.assertIn(b'purify.min.js', res1.data)
            self.assertIn(b'katex.min.js', res1.data)
            self.assertEqual(mock_summarize.call_count, 1)

            # 2. Second call: Cache hit, does NOT call summarize_document
            res2 = self.client.post("/summarize", data={
                "stored_filename": "cache_doc.pdf",
                "original_filename": "cache_doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res2.status_code, 200)
            self.assertIn(b"First-time generated summary", res2.data)
            self.assertIn(b"Cached in SQLite", res2.data)
            self.assertEqual(mock_summarize.call_count, 1)  # Unchanged!

            # 3. Third call: Explicit regenerate forces fresh call
            mock_summarize.return_value = {
                "success": True,
                "summary": "Freshly regenerated summary.",
                "error": None,
                "model_used": "gemini-flash-lite-latest"
            }
            res3 = self.client.post("/summarize", data={
                "stored_filename": "cache_doc.pdf",
                "original_filename": "cache_doc.pdf",
                "regenerate": "true",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res3.status_code, 200)
            self.assertIn(b"Freshly regenerated summary", res3.data)
            self.assertEqual(mock_summarize.call_count, 2)

    def test_prepare_condensed_text_bounded_limits(self):
        """Test bounded condensation preserves document structure while respecting budgets."""
        # 1. Short document stays uncompressed
        short_pages = [
            {"page_number": 1, "text": "Short intro.", "char_count": 12},
            {"page_number": 2, "text": "Short conclusion.", "char_count": 17}
        ]
        text, is_condensed, dt = prepare_condensed_text(short_pages, max_chars=1000)
        self.assertFalse(is_condensed)
        self.assertIn("Short intro.", text)
        self.assertIn("Short conclusion.", text)

        # 2. Long document exceeding max_chars gets condensed intelligently
        long_pages = [
            {"page_number": 1, "text": "Opening abstract: " + ("Alpha " * 200), "char_count": 1200},
            {"page_number": 2, "text": "Methodology: " + ("Beta " * 200), "char_count": 1000},
            {"page_number": 3, "text": "Analysis: " + ("Gamma " * 200), "char_count": 1200},
            {"page_number": 4, "text": "Results: " + ("Delta " * 200), "char_count": 1000},
            {"page_number": 5, "text": "Final conclusions: " + ("Omega " * 200), "char_count": 1200}
        ]
        text, is_condensed, dt = prepare_condensed_text(long_pages, max_chars=2000)
        self.assertTrue(is_condensed)
        self.assertLessEqual(len(text), 2200)  # within budget + note
        self.assertIn("Opening abstract", text)
        self.assertIn("Final conclusions", text)
        self.assertIn("Document condensed from 5 pages", text)

    def test_client_timeout_configured(self):
        """Test that create_gemini_client configures the http_options timeout."""
        with patch("google.genai.Client") as mock_client_cls:
            create_gemini_client(api_key="mock_key", timeout_seconds=25)
            mock_client_cls.assert_called_once_with(
                api_key="mock_key",
                http_options={"timeout": 25000}
            )

    def test_path_traversal_protection_on_summarize_and_ask(self):
        """Test that directory traversal payloads in stored_filename are safely rejected."""
        traversal_payloads = [
            "../../etc/passwd",
            "..\\..\\boot.ini",
            "../secret.pdf",
            "/etc/hosts",
            "C:\\Windows\\System32\\drivers\\etc\\hosts"
        ]
        for payload in traversal_payloads:
            # 1. Test /summarize rejection
            res_sum = self.client.post("/summarize", data={
                "stored_filename": payload,
                "original_filename": "doc.pdf",
                "csrf_token": self.get_csrf_token()
            }, follow_redirects=True)
            self.assertEqual(res_sum.status_code, 200)
            self.assertIn(b"Invalid document reference or file not found.", res_sum.data)

            # 2. Test /ask rejection
            res_ask = self.client.post("/ask", data={
                "stored_filename": payload,
                "original_filename": "doc.pdf",
                "question": "What is in this file?",
                "csrf_token": self.get_csrf_token()
            }, follow_redirects=True)
            self.assertEqual(res_ask.status_code, 200)
            self.assertIn(b"Invalid document reference or file not found.", res_ask.data)

    def test_cache_db_failure_gracefully_falls_back_to_live_generation(self):
        """Test that SQLite cache lookup or save errors do not crash the summarization route."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Fallback Doc) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "db_fail_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.get_document_summary", return_value=None), \
             patch("routes.pdf_routes.save_document_summary", return_value=False), \
             patch("routes.pdf_routes.summarize_document") as mock_summarize:
            mock_summarize.return_value = {
                "success": True,
                "summary": "Resilient summary generated despite DB error.",
                "error": None,
                "model_used": "gemini-flash-lite-latest"
            }
            res = self.client.post("/summarize", data={
                "stored_filename": "db_fail_doc.pdf",
                "original_filename": "db_fail_doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Resilient summary generated despite DB error.", res.data)
            mock_summarize.assert_called_once()

    def test_failed_summaries_are_not_cached_in_sqlite(self):
        """Test that failed Gemini API calls are never saved into the SQLite summary cache."""
        from database.models import get_document_summary

        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (No Cache Doc) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "failed_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.summarize_document") as mock_summarize:
            mock_summarize.return_value = {
                "success": False,
                "summary": "",
                "error": "Gemini API rate limit or quota exceeded.",
                "model_used": "gemini-flash-lite-latest"
            }
            res = self.client.post("/summarize", data={
                "stored_filename": "failed_doc.pdf",
                "original_filename": "failed_doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Gemini API rate limit or quota exceeded.", res.data)

            # Verify no record was cached in the database
            cached = get_document_summary("failed_doc.pdf", db_path=self.test_db_path)
            self.assertIsNone(cached)

    def test_clear_chat_history_on_ask(self):
        """Test clearing conversation history via POST /ask with clear_chat=true."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Chat Doc) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "chat_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        # Pre-seed session history
        with self.client.session_transaction() as sess:
            sess["qa_history_chat_doc.pdf"] = [
                {"question": "What is this?", "answer": "A document", "page_citations": [1], "success": True}
            ]

        # Send clear_chat request
        res = self.client.post("/ask", data={
            "stored_filename": "chat_doc.pdf",
            "original_filename": "chat_doc.pdf",
            "clear_chat": "true",
            "csrf_token": self.get_csrf_token()
        })
        self.assertEqual(res.status_code, 200)

        # Check session is cleared
        with self.client.session_transaction() as sess:
            self.assertNotIn("qa_history_chat_doc.pdf", sess)

    def test_summary_error_does_not_render_key_input_form(self):
        """Test that API key errors display the alert without rendering any key input or save form."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Key Test) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "key_err_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.summarize_document") as mock_summarize:
            mock_summarize.return_value = {
                "success": False,
                "summary": "",
                "error": "Invalid or missing Google Gemini API key. Please check your GEMINI_API_KEY in your .env file.",
                "model_used": "gemini-flash-lite-latest"
            }
            res = self.client.post("/summarize", data={
                "stored_filename": "key_err_doc.pdf",
                "original_filename": "key_err_doc.pdf",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res.status_code, 200)
            self.assertIn(b"Invalid or missing Google Gemini API key", res.data)
            # Ensure no option/input for pasting or saving the key is rendered in the UI
            self.assertNotIn(b"Configure Your Google Gemini API Key", res.data)
            self.assertNotIn(b"name=\"gemini_api_key\"", res.data)
            self.assertNotIn(b"btn-save-key", res.data)
            self.assertNotIn(b"Paste your Gemini API key", res.data)

    def test_get_api_key_rejects_placeholder_values(self):
        """Test that get_api_key ignores placeholder strings and returns None if no real key is set."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "your_gemini_api_key_here"}, clear=False), \
             patch("os.path.exists", return_value=False):
            key = get_api_key()
            self.assertIsNone(key)

    def test_summarize_document_prompt_structure(self):
        """Test that summarize_document constructs a publication-grade research prompt."""
        pages = [{"page_number": 1, "text": "Paper introducing early-exit neural networks.", "char_count": 45}]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Executive Research & Document Report\n## Document Information\n- **Title:** Test"
        mock_client.models.generate_content.return_value = mock_response

        summarize_document(
            pages,
            client=mock_client,
            document_title="Deep_Residual_Learning.pdf",
            total_pages=12
        )

        call_args = mock_client.models.generate_content.call_args
        self.assertIsNotNone(call_args)
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or ""
        self.assertIn("Executive Research & Document Report", prompt)
        self.assertIn("Document Information", prompt)
        self.assertIn("Deep_Residual_Learning.pdf", prompt)
        self.assertIn("12 pages", prompt)
        self.assertIn("Executive Overview", prompt)
        self.assertIn("Key Findings & Core Contributions", prompt)
        self.assertIn("Quantitative Results & Empirical Benchmarks", prompt)
        self.assertIn("Detailed Analysis", prompt)
        self.assertIn("Limitations & Critical Considerations", prompt)
        self.assertIn("Conclusion & Strategic Implications", prompt)

    def test_answer_pdf_question_prompt_adaptive_structure(self):
        """Test that answer_pdf_question constructs an adaptive academic response prompt."""
        pages = [{"page_number": 1, "text": "ResNet utilizes identity skip connections.", "char_count": 42}]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Direct Answer: ResNet uses skip connections [Page 1]."
        mock_client.models.generate_content.return_value = mock_response

        answer_pdf_question(
            pages,
            "How do residual connections prevent vanishing gradients?",
            client=mock_client
        )

        call_args = mock_client.models.generate_content.call_args
        self.assertIsNotNone(call_args)
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or ""
        self.assertIn("Direct Answer", prompt)
        self.assertIn("Detailed Explanation", prompt)
        self.assertIn("Key Points & Technical Distinctions", prompt)
        self.assertIn("Source References", prompt)
        self.assertIn("The uploaded document does not contain enough evidence to answer this question.", prompt)

    def test_is_broad_query_detection(self):
        """Test detection of broad/comprehensive queries vs focused inquiries."""
        from pdf_processing.chunker import is_broad_query
        self.assertTrue(is_broad_query("Explain this PDF in detail"))
        self.assertTrue(is_broad_query("Can you walk me through this document?"))
        self.assertTrue(is_broad_query("Provide an overview of the paper"))
        self.assertFalse(is_broad_query("What was the batch size used in training?"))
        self.assertFalse(is_broad_query("Who is the primary author?"))

    def test_broad_query_passage_distribution(self):
        """Test that broad queries retrieve passages distributed across the document."""
        from pdf_processing.chunker import retrieve_relevant_chunks
        chunks = [
            {"chunk_id": i, "page_number": i, "text": f"Page content paragraph {i}"}
            for i in range(1, 21)
        ]
        distributed = retrieve_relevant_chunks(chunks, query="Explain this PDF in detail", top_k=4)
        self.assertEqual(len(distributed), 4)
        page_nums = [c["page_number"] for c in distributed]
        # Should not all be from page 1, should span across the document
        self.assertLess(page_nums[0], 5)
        self.assertGreater(page_nums[-1], 15)


    def test_explain_page_content_structure_and_prompt(self):
        """Test that explain_page_content constructs the required academic prompt and returns parsed result."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            "# Page 3 — Methodology and Dynamic Architecture\n\n"
            "This page establishes the mathematical formulation for entropy-based early exiting.\n\n"
            "### Proposed Formulation\nDetailed technical analysis of the loss function.\n\n"
            r"### Important Technical Details\n- Metric: Cross-entropy $\mathcal{H}(p)$" "\n\n"
            r"### Key Takeaways\n- Early-exit threshold $\tau=0.15$" "\n\n"
            "### Source Context\n- Page 3; Equation (4); Figure 2."
        )
        mock_client.models.generate_content.return_value = mock_response

        result = explain_page_content(
            page_number=3,
            page_text="This page introduces the mathematical formulation of cross-entropy early exit with threshold tau = 0.15.",
            total_pages=15,
            document_title="Deep_Exit.pdf",
            client=mock_client,
            model_name="gemini-2.5-flash"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["page_number"], 3)
        self.assertEqual(result["model_used"], "gemini-2.5-flash")
        self.assertIn("# Page 3 — Methodology", result["explanation"])
        self.assertIn("timing_ms", result)

        # Inspect prompt passed to Gemini
        call_args = mock_client.models.generate_content.call_args
        self.assertIsNotNone(call_args)
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents") or ""
        self.assertIn("Page 3 of 15", prompt)
        self.assertIn("Deep_Exit.pdf", prompt)
        self.assertIn("# Page 3", prompt)
        self.assertIn("Page Overview", prompt)
        self.assertIn("Detailed Explanation", prompt)
        self.assertIn("Important Technical Details", prompt)
        self.assertIn("Key Takeaways", prompt)
        self.assertIn("Source References & Citations", prompt)

    def test_explain_page_content_empty_text(self):
        """Test that explain_page_content handles empty or blank pages without calling the API."""
        mock_client = MagicMock()
        result = explain_page_content(page_number=4, page_text="   \n \t ", client=mock_client)
        self.assertFalse(result["success"])
        self.assertIn("no extractable text", result["error"].lower())
        mock_client.models.generate_content.assert_not_called()

    def test_explain_page_content_rate_limit_error(self):
        """Test graceful translation of quota/rate-limit exceptions in explain_page_content."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded")

        result = explain_page_content(
            page_number=1,
            page_text="Valid page text content.",
            client=mock_client
        )
        self.assertFalse(result["success"])
        self.assertIn("rate limit or quota exceeded", result["error"])

    def test_explain_page_endpoint_form_and_caching(self):
        """Test /explain-page endpoint handles form requests, caches in SQLite, and regenerates properly."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Page 1 content) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "explain_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.explain_page_content") as mock_explain:
            mock_explain.return_value = {
                "success": True,
                "page_number": 1,
                "explanation": "# Page 1 — Introduction and Abstract\nWalkthrough body.",
                "model_used": "gemini-flash-lite-latest",
                "timing_ms": {"api_ms": 120.0}
            }

            # 1. First call: Cache miss, calls explain_page_content
            res1 = self.client.post("/explain-page", data={
                "stored_filename": "explain_doc.pdf",
                "original_filename": "paper.pdf",
                "page_number": "1",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res1.status_code, 200)
            data1 = res1.get_json()
            self.assertTrue(data1["success"])
            self.assertFalse(data1["cached"])
            self.assertIn("Page 1 — Introduction and Abstract", data1["explanation"])
            self.assertEqual(mock_explain.call_count, 1)

            # 2. Second call: Cache hit from SQLite, does not call explain_page_content
            res2 = self.client.post("/explain-page", data={
                "stored_filename": "explain_doc.pdf",
                "original_filename": "paper.pdf",
                "page_number": "1",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res2.status_code, 200)
            data2 = res2.get_json()
            self.assertTrue(data2["success"])
            self.assertTrue(data2["cached"])
            self.assertEqual(mock_explain.call_count, 1)

            # 3. Third call: Force regenerate with regenerate="true"
            mock_explain.return_value = {
                "success": True,
                "page_number": 1,
                "explanation": "# Page 1 — Regenerated Walkthrough\nNew body.",
                "model_used": "gemini-flash-lite-latest",
                "timing_ms": {"api_ms": 110.0}
            }
            res3 = self.client.post("/explain-page", data={
                "stored_filename": "explain_doc.pdf",
                "original_filename": "paper.pdf",
                "page_number": "1",
                "regenerate": "true",
                "csrf_token": self.get_csrf_token()
            })
            self.assertEqual(res3.status_code, 200)
            data3 = res3.get_json()
            self.assertTrue(data3["success"])
            self.assertFalse(data3["cached"])
            self.assertIn("Regenerated Walkthrough", data3["explanation"])
            self.assertEqual(mock_explain.call_count, 2)

    def test_explain_page_endpoint_json_payload(self):
        """Test /explain-page endpoint when invoked via JSON payload with CSRF headers."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Page 1 content) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "json_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        with patch("routes.pdf_routes.explain_page_content") as mock_explain:
            mock_explain.return_value = {
                "success": True,
                "page_number": 1,
                "explanation": "# Page 1 — JSON API Test\nJSON response.",
                "model_used": "gemini-flash-lite-latest",
                "timing_ms": {"api_ms": 95.0}
            }

            token = self.get_csrf_token()
            res = self.client.post(
                "/explain-page",
                json={
                    "stored_filename": "json_doc.pdf",
                    "original_filename": "paper.pdf",
                    "page_number": 1
                },
                headers={"X-CSRFToken": token}
            )
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertIn("JSON API Test", data["explanation"])

    def test_explain_page_endpoint_validation_and_errors(self):
        """Test input validation, out-of-bounds pages, and file traversal in /explain-page."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Page 1) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "valid_one_page.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        token = self.get_csrf_token()

        # 1. Non-digit page number -> 400
        res = self.client.post("/explain-page", data={
            "stored_filename": "valid_one_page.pdf",
            "page_number": "abc",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid or missing page number", res.get_json()["error"])

        # 2. Path traversal stored_filename -> 404
        res = self.client.post("/explain-page", data={
            "stored_filename": "../../secret.pdf",
            "page_number": "1",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 404)

        # 3. Page number out of bounds -> 404
        res = self.client.post("/explain-page", data={
            "stored_filename": "valid_one_page.pdf",
            "page_number": "999",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found in document", res.get_json()["error"])

    def test_page_wise_interface_excludes_raw_extracted_text(self):
        """Regression test: verify raw extracted PDF text is completely excluded from the page-wise UI."""
        pdf_raw_sentence = "CONFIDENTIAL_UNPROCESSED_RAW_EXTRACTED_SENTENCE_123"
        pdf_content = (
            b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n"
            b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"5 0 obj << /Length 65 >> stream\n"
            b"BT /F1 12 Tf 100 700 Td (" + pdf_raw_sentence.encode("utf-8") + b") Tj ET\n"
            b"endstream\nendobj\nxref\n0 6\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \n"
            b"trailer << /Root 1 0 R /Size 6 >>\nstartxref\n455\n%%EOF"
        )

        test_pdf_path = os.path.join(self.test_upload_dir, "clean_ui_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        token = self.get_csrf_token()
        res = self.client.post("/summarize", data={
            "stored_filename": "clean_ui_doc.pdf",
            "original_filename": "clean_ui_doc.pdf",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # 1. Assert no raw-text accordion or pre tag exists
        self.assertNotIn("raw-text-accordion", html)
        self.assertNotIn("extracted-text", html)
        self.assertNotIn("page-char-metric", html)

        # 2. Assert raw extracted sentence is NOT in the pages section
        pages_section_start = html.find('class="pages-section"')
        self.assertGreater(pages_section_start, -1)
        pages_section_html = html[pages_section_start:]
        self.assertNotIn(pdf_raw_sentence, pages_section_html)

    def test_failed_explanation_generation_does_not_fall_back_to_raw_text(self):
        """Regression test: verify API failure does not substitute raw text into the explanation response."""
        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Page 1) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "fail_fallback_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        token = self.get_csrf_token()

        with patch("routes.pdf_routes.explain_page_content") as mock_explain:
            mock_explain.return_value = {
                "success": False,
                "page_number": 1,
                "explanation": "",
                "error": "Gemini API Quota Exceeded. Please retry.",
                "model_used": "gemini-flash-lite-latest",
                "timing_ms": {"api_ms": 10.0}
            }

            res = self.client.post("/explain-page", data={
                "stored_filename": "fail_fallback_doc.pdf",
                "page_number": "1",
                "csrf_token": token
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertFalse(data["success"])
            self.assertEqual(data.get("explanation"), "")
            self.assertIn("Quota Exceeded", data["error"])
            self.assertNotEqual(data.get("explanation"), "Page 1")

    def test_cached_explanation_displayed_without_raw_text(self):
        """Regression test: verify cached explanation displays cleanly without any raw text fallback."""
        from database.models import save_page_explanation

        pdf_content = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj\n4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n5 0 obj << /Length 30 >> stream\nBT /F1 12 Tf 100 700 Td (Page 1) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000262 00000 n \n0000000341 00000 n \ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n420\n%%EOF"
        test_pdf_path = os.path.join(self.test_upload_dir, "cached_view_doc.pdf")
        with open(test_pdf_path, "wb") as f:
            f.write(pdf_content)

        # Seed cached explanation in SQLite
        save_page_explanation(
            filename="cached_view_doc.pdf",
            page_number=1,
            explanation="# Page 1 — Deep Overview\nGenuine academic explanation.",
            model_used="gemini-flash-lite-latest",
            db_path=self.test_db_path
        )

        token = self.get_csrf_token()
        res = self.client.post("/summarize", data={
            "stored_filename": "cached_view_doc.pdf",
            "original_filename": "cached_view_doc.pdf",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        self.assertIn("Genuine academic explanation", html)
        self.assertIn("&#10003; Explained", html)
        self.assertNotIn("raw-text-accordion", html)
        self.assertNotIn("extracted-text", html)

    def test_empty_or_image_only_pages_handled_appropriately(self):
        """Regression test: verify blank/scanned pages show clear notice and do not crash or show raw text."""
        import pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        blank_pdf_path = os.path.join(self.test_upload_dir, "blank_page_doc.pdf")
        with open(blank_pdf_path, "wb") as f:
            writer.write(f)

        token = self.get_csrf_token()
        res = self.client.post("/summarize", data={
            "stored_filename": "blank_page_doc.pdf",
            "original_filename": "blank_page_doc.pdf",
            "csrf_token": token
        })
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        self.assertIn("empty-page-box", html)
        self.assertIn("This page contains no extractable text", html)
        self.assertNotIn("raw-text-accordion", html)


if __name__ == "__main__":
    unittest.main()
