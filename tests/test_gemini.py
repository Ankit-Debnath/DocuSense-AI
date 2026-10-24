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
from ai.analyzer import summarize_document, answer_pdf_question, prepare_condensed_text
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


if __name__ == "__main__":
    unittest.main()
