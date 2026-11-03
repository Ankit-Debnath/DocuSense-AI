"""
Integration Tests for Hybrid Retrieval, On-Demand Indexing, and /ask Route
Tests end-to-end question answering with hybrid RRF retrieval, on-demand indexing,
concurrent double-checked locking, dimension mismatch fallback, and API error resilience.
Zero external API calls are made (Gemini client is mocked).
"""

import io
import json
import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import pypdf
from app import create_app
from database.db import init_db
from database.models import (
    insert_document,
    get_document_by_id,
    count_document_chunks,
    get_document_chunks,
    save_document_chunks
)
from ai.indexing_service import ensure_document_indexed
from ai.analyzer import answer_pdf_question


def create_sample_pdf_bytes(text: str = "Apollo 11 landed on the moon on July 20, 1969.") -> bytes:
    """Creates a valid PDF binary with extractable text."""
    pdf_content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length {len(text) + 20} >> stream
BT
/F1 20 Tf
50 700 Td
({text}) Tj
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
450
%%EOF"""
    return pdf_content.encode("latin-1")


class OnDemandIndexingIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.upload_dir = os.path.join(self.test_dir, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_indexing.db")
        init_db(self.db_path)

        # Write sample PDF to uploads
        self.pdf_filename = "moon.pdf"
        self.pdf_path = os.path.join(self.upload_dir, self.pdf_filename)
        with open(self.pdf_path, "wb") as f:
            f.write(create_sample_pdf_bytes("Neil Armstrong took one small step for man in 1969."))

        self.doc_id = insert_document(
            filename=self.pdf_filename,
            original_filename="original_moon.pdf",
            page_count=1,
            character_count=52,
            status="processed",
            content_hash="hash_moon",
            db_path=self.db_path
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_client(self, dimension: int = 768):
        mock_client = MagicMock()
        def side_effect(model, contents, config):
            resp = MagicMock()
            items = []
            for _ in contents:
                item = MagicMock()
                item.values = [0.05] * dimension
                items.append(item)
            resp.embeddings = items
            return resp
        mock_client.models.embed_content.side_effect = side_effect
        return mock_client

    def test_missing_index_triggers_on_demand_indexing(self):
        """When an unindexed document is queried, ensure_document_indexed indexes it."""
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), 0)

        mock_client = self._create_mock_client(dimension=768)
        status = ensure_document_indexed(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            client=mock_client,
            db_path=self.db_path
        )

        self.assertTrue(status["indexed"])
        self.assertTrue(status["freshly_indexed"])
        self.assertGreater(status["chunk_count"], 0)
        self.assertEqual(count_document_chunks(self.doc_id, db_path=self.db_path), status["chunk_count"])

    def test_existing_index_reused_without_reindexing(self):
        """When an index already exists, ensure_document_indexed immediately reuses it."""
        # Pre-populate index
        mock_client = self._create_mock_client(dimension=768)
        ensure_document_indexed(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            client=mock_client,
            db_path=self.db_path
        )
        initial_count = count_document_chunks(self.doc_id, db_path=self.db_path)
        self.assertGreater(initial_count, 0)

        # Second call must NOT call Gemini API again
        call_tracker = MagicMock()
        status2 = ensure_document_indexed(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            client=call_tracker,
            db_path=self.db_path
        )

        self.assertTrue(status2["indexed"])
        self.assertFalse(status2["freshly_indexed"])
        self.assertEqual(status2["chunk_count"], initial_count)
        call_tracker.models.embed_content.assert_not_called()

    def test_concurrent_indexing_requests_serialize_safely(self):
        """Simultaneous concurrent requests on the same document do not duplicate chunks or corrupt DB."""
        mock_client = self._create_mock_client(dimension=768)

        def worker():
            ensure_document_indexed(
                document_id=self.doc_id,
                file_path=self.pdf_path,
                client=mock_client,
                db_path=self.db_path
            )

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 set of chunks must exist (no duplicates)
        count = count_document_chunks(self.doc_id, db_path=self.db_path)
        chunks = get_document_chunks(self.doc_id, db_path=self.db_path)
        self.assertEqual(len(chunks), count)
        chunk_indices = [c["chunk_index"] for c in chunks]
        self.assertEqual(len(chunk_indices), len(set(chunk_indices)))


class AskRouteHybridRetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.upload_dir = os.path.join(self.test_dir, "uploads")
        self.instance_dir = os.path.join(self.test_dir, "instance")
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.instance_dir, exist_ok=True)
        self.db_path = os.path.join(self.instance_dir, "pdf_analyzer.db")

        test_upload_dir = self.upload_dir
        test_instance_dir = self.instance_dir
        test_db_path = self.db_path

        from config import Config
        class HybridTestConfig(Config):
            TESTING = True
            SECRET_KEY = "test-key-for-hybrid-qa"
            WTF_CSRF_ENABLED = False
            UPLOAD_FOLDER = test_upload_dir
            INSTANCE_FOLDER = test_instance_dir
            DATABASE_PATH = test_db_path

        self.app = create_app(HybridTestConfig)
        self.client = self.app.test_client()

        with self.app.app_context():
            init_db(self.db_path)

        # Write sample PDF to uploads
        self.pdf_filename = "hybrid_doc.pdf"
        self.pdf_path = os.path.join(self.upload_dir, self.pdf_filename)
        with open(self.pdf_path, "wb") as f:
            f.write(create_sample_pdf_bytes("Quantum teleportation transfers quantum information between locations."))

        # Insert document record in SQLite
        with self.app.app_context():
            self.doc_id = insert_document(
                filename=self.pdf_filename,
                original_filename="quantum.pdf",
                page_count=1,
                character_count=65,
                status="processed",
                content_hash="hash_quantum",
                db_path=self.db_path
            )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_answer_pdf_question_hybrid_success(self):
        """Verify answer_pdf_question performs hybrid retrieval with RRF when index and client are valid."""
        pages = [{"page_number": 1, "text": "Quantum teleportation transfers quantum information."}]

        mock_client = MagicMock()
        # Mock embed_content for query and document
        def embed_side_effect(model, contents, config):
            resp = MagicMock()
            resp.embeddings = [MagicMock(values=[0.1, 0.2])]
            return resp
        mock_client.models.embed_content.side_effect = embed_side_effect

        # Mock generate_content for answer
        mock_gen_resp = MagicMock()
        mock_gen_resp.text = json.dumps({
            "answer": "Quantum teleportation transmits quantum states using entanglement [Page 1].",
            "citations": [
                {
                    "evidence_id": "E1",
                    "page_number": 1,
                    "supporting_quote": "Quantum teleportation transfers quantum information."
                }
            ],
            "insufficient_evidence": False
        })
        mock_client.models.generate_content.return_value = mock_gen_resp

        # Pre-index the document
        chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Quantum teleportation transfers quantum information.",
            "char_count": 52,
            "embedding": [0.1, 0.2],
            "embedding_dim": 2,
            "embedding_model": "gemini-embedding-001"
        }]
        save_document_chunks(self.doc_id, chunks, db_path=self.db_path)

        with self.app.app_context():
            result = answer_pdf_question(
                pages=pages,
                question="What is quantum teleportation?",
                client=mock_client,
                document_id=self.doc_id,
                file_path=self.pdf_path,
                db_path=self.db_path
            )

        self.assertTrue(result["success"])
        self.assertIn("quantum", result["answer"].lower())
        self.assertEqual(result["page_citations"], [1])
        self.assertEqual(result["retrieval_method"], "hybrid")
        self.assertGreater(len(result["evidence_passages"]), 0)

    def test_embedding_api_failure_falls_back_to_keyword(self):
        """When embedding API fails, answer_pdf_question smoothly falls back to keyword retrieval."""
        pages = [{"page_number": 1, "text": "Quantum teleportation transfers quantum information."}]

        mock_client = MagicMock()
        # Embedding fails with API error
        mock_client.models.embed_content.side_effect = RuntimeError("503 Service Unavailable")

        # Answer generation succeeds
        mock_gen_resp = MagicMock()
        mock_gen_resp.text = json.dumps({
            "answer": "Answer generated using keyword fallback context [Page 1].",
            "citations": [
                {
                    "evidence_id": "E1",
                    "page_number": 1,
                    "supporting_quote": "Quantum teleportation transfers quantum information."
                }
            ],
            "insufficient_evidence": False
        })
        mock_client.models.generate_content.return_value = mock_gen_resp

        with self.app.app_context():
            result = answer_pdf_question(
                pages=pages,
                question="What is quantum teleportation?",
                client=mock_client,
                document_id=self.doc_id,
                file_path=self.pdf_path,
                db_path=self.db_path
            )

        self.assertTrue(result["success"])
        self.assertIn("keyword", result["retrieval_method"])
        self.assertEqual(result["page_citations"], [1])

    def test_dimension_mismatch_falls_back_to_keyword(self):
        """When query vector dimension != stored vector dimension, fall back to keyword retrieval."""
        pages = [{"page_number": 1, "text": "Quantum teleportation transfers quantum information."}]

        # Stored chunks have 2 dimensions
        chunks = [{
            "chunk_index": 0,
            "page_number": 1,
            "chunk_text": "Quantum teleportation transfers quantum information.",
            "char_count": 52,
            "embedding": [0.1, 0.2],
            "embedding_dim": 2,
            "embedding_model": "gemini-embedding-001"
        }]
        save_document_chunks(self.doc_id, chunks, db_path=self.db_path)

        mock_client = MagicMock()
        # Query embedding returns 3 dimensions!
        mock_resp = MagicMock()
        mock_resp.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
        mock_client.models.embed_content.return_value = mock_resp

        mock_gen_resp = MagicMock()
        mock_gen_resp.text = json.dumps({
            "answer": "Answer generated via keyword fallback [Page 1].",
            "citations": [
                {
                    "evidence_id": "E1",
                    "page_number": 1,
                    "supporting_quote": "Quantum teleportation transfers quantum information."
                }
            ],
            "insufficient_evidence": False
        })
        mock_client.models.generate_content.return_value = mock_gen_resp

        with self.app.app_context():
            result = answer_pdf_question(
                pages=pages,
                question="What is quantum teleportation?",
                client=mock_client,
                document_id=self.doc_id,
                file_path=self.pdf_path,
                db_path=self.db_path
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["retrieval_method"], "keyword_fallback")
        self.assertEqual(result["page_citations"], [1])

    def test_ask_route_end_to_end_integration(self):
        """Test POST /ask route with document_id resolution and hybrid retrieval."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.embeddings = [MagicMock(values=[0.1, 0.2])]
        mock_client.models.embed_content.return_value = mock_resp

        mock_gen = MagicMock()
        mock_gen.text = json.dumps({
            "answer": "Grounded hybrid answer from Gemini. [Page 1]",
            "citations": [
                {
                    "evidence_id": "E1",
                    "page_number": 1,
                    "supporting_quote": "Quantum teleportation transfers quantum information"
                }
            ],
            "insufficient_evidence": False
        })
        mock_client.models.generate_content.return_value = mock_gen

        with patch("ai.analyzer.create_gemini_client", return_value=mock_client), \
             patch("ai.embedding_client.create_gemini_client", return_value=mock_client):
            response = self.client.post(
                "/ask",
                data={
                    "stored_filename": self.pdf_filename,
                    "original_filename": "quantum.pdf",
                    "question": "Explain quantum teleportation."
                }
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Grounded hybrid answer from Gemini.", response.data)
