"""
Unit and Integration Tests for the Retrieval Evaluation and Benchmarking Framework
Validates:
- Synthetic evaluation PDF generator and pypdf text extraction.
- Comprehensive evaluation query suite attributes and coverage.
- End-to-end comparative retrieval benchmarking across Keyword, Semantic, and Hybrid RRF.
- Separation of answerable query metrics from unanswerable query metrics.
- Isolated latency benchmarking (embedding, SQLite fetch, cosine math, RRF fusion).
- Deterministic repeated benchmark results.
- RRF candidate pool optimization verification.

NOTE ON MOCK EMBEDDINGS:
MockEmbeddingClient implements a deterministic bag-of-words / token-hashing projection.
It does NOT possess semantic awareness and does NOT evaluate real Gemini neural embedding quality.
It serves strictly as a reproducible offline unit test double.
"""

import os
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock

from ai.evaluation import (
    EVALUATION_DOCUMENT_PAGES,
    EVALUATION_QUERIES,
    DISTRACTOR_DOCUMENT_PAGES,
    create_evaluation_pdf_bytes,
    create_distractor_pdf_bytes,
    END_TO_END_EVALUATION_DATASET,
    run_retrieval_benchmark,
    run_latency_benchmark,
    run_end_to_end_qa_benchmark,
    validate_evidence_integrity,
    MockEndToEndQAClient,
)
from database.db import init_db
from database.models import insert_document
from pdf_processing.extractor import extract_text_from_pdf
from ai.indexing_service import index_document


class MockModels:
    def __init__(self, parent):
        self.parent = parent

    def embed_content(self, model: str, contents: Any, config: Any = None):
        mock_resp = MagicMock()
        if isinstance(contents, str):
            vec = self.parent._text_to_vector(contents)
            item = MagicMock()
            item.values = vec
            mock_resp.embeddings = [item]
            mock_resp.embedding = item
        elif isinstance(contents, (list, tuple)):
            items = []
            for t in contents:
                item = MagicMock()
                item.values = self.parent._text_to_vector(t)
                items.append(item)
            mock_resp.embeddings = items
        else:
            mock_resp.embeddings = []
        return mock_resp


class MockEmbeddingClient:
    """
    Deterministic mock Google GenAI client that generates pseudo-vectors
    based on word token hashes.

    IMPORTANT SCIENTIFIC LIMITATION:
    This mock uses a random-projection bag-of-words / token-hashing approach.
    It does NOT model semantic synonyms or conceptual proximity.
    It exists purely for zero-cost, hermetic offline testing.
    """
    DIM = 64

    def __init__(self):
        self.models = MockModels(self)

    def _text_to_vector(self, text: str) -> list[float]:
        # Generate a deterministic pseudo-dense vector based on word token hashes
        vec = [0.0] * self.DIM
        words = text.lower().replace(",", " ").replace(".", " ").replace("(", " ").replace(")", " ").split()
        if not words:
            return [1.0 / (self.DIM ** 0.5)] * self.DIM

        for word in words:
            h = hash(word)
            for i in range(self.DIM):
                vec[i] += ((h >> (i % 31)) & 1) * 2.0 - 1.0

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec


class TestEvaluationDatasetAndPDF(unittest.TestCase):
    """Tests the synthetic PDF generation and text extraction."""

    def test_create_evaluation_pdf_bytes_and_extraction(self):
        pdf_bytes = create_evaluation_pdf_bytes()
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))
        self.assertGreater(len(pdf_bytes), 1000)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            extraction = extract_text_from_pdf(tmp_path)
            self.assertTrue(extraction["success"])
            self.assertEqual(extraction["total_pages"], 5)
            pages = extraction["pages"]
            self.assertEqual(len(pages), 5)
            for idx, page in enumerate(pages, start=1):
                self.assertEqual(page["page_number"], idx)
                self.assertTrue(len(page["text"].strip()) > 50)
                expected_prefix = EVALUATION_DOCUMENT_PAGES[idx - 1]["text"][:30]
                self.assertIn(expected_prefix, page["text"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_evaluation_queries_structure(self):
        self.assertEqual(len(EVALUATION_QUERIES), 8)
        categories = {q["category"] for q in EVALUATION_QUERIES}
        self.assertIn("factual", categories)
        self.assertIn("conceptual", categories)
        self.assertIn("cross_page", categories)
        self.assertIn("synonyms", categories)
        self.assertIn("limited_keyword_overlap", categories)
        self.assertIn("unanswerable", categories)

        for q in EVALUATION_QUERIES:
            self.assertIn("id", q)
            self.assertIn("query", q)
            self.assertIn("relevant_chunks", q)
            self.assertIn("graded_relevance", q)
            if q["category"] == "unanswerable":
                self.assertEqual(len(q["relevant_chunks"]), 0)
                self.assertEqual(len(q["graded_relevance"]), 0)
            else:
                self.assertGreater(len(q["relevant_chunks"]), 0)


class TestComparativeRetrievalBenchmark(unittest.TestCase):
    """Tests the automated benchmark harness and comparative retrieval evaluation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_eval.db")
        init_db(self.db_path)

        # Generate evaluation PDF
        self.pdf_path = os.path.join(self.tmp_dir.name, "eval_doc.pdf")
        pdf_bytes = create_evaluation_pdf_bytes()
        with open(self.pdf_path, "wb") as f:
            f.write(pdf_bytes)

        extraction = extract_text_from_pdf(self.pdf_path)
        self.assertTrue(extraction["success"])
        self.pages = extraction["pages"]
        total_chars = extraction["total_characters"]
        self.doc_id = insert_document(
            filename="eval_doc.pdf",
            original_filename="eval_doc.pdf",
            page_count=len(self.pages),
            character_count=total_chars,
            db_path=self.db_path
        )

        # Index document using mock embedding client
        self.mock_client = MockEmbeddingClient()
        index_document(self.doc_id, self.pdf_path, client=self.mock_client, db_path=self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_run_retrieval_benchmark_execution(self):
        result = run_retrieval_benchmark(
            document_id=self.doc_id,
            pages=self.pages,
            client=self.mock_client,
            k=5,
            queries=EVALUATION_QUERIES,
            db_path=self.db_path
        )

        self.assertEqual(result["k"], 5)
        self.assertEqual(result["query_count"], 8)
        self.assertEqual(result["answerable_query_count"], 7)
        self.assertEqual(result["unanswerable_query_count"], 1)
        self.assertIn("comparison_table", result)
        table = result["comparison_table"]

        for strategy in ["keyword", "semantic", "hybrid"]:
            self.assertIn(strategy, table)
            metrics = table[strategy]
            self.assertIn("Precision@5", metrics)
            self.assertIn("Recall@5", metrics)
            self.assertIn("MRR", metrics)
            self.assertIn("NDCG@5", metrics)
            self.assertIn("Hit Rate@5", metrics)
            self.assertIn("unanswerable_query_count", metrics)
            self.assertIn("false_positive_retrieval_rate", metrics)
            self.assertIn("correct_abstention_rate", metrics)

            self.assertEqual(metrics["unanswerable_query_count"], 1)
            # In current pipeline without query classifier, all retrievers return candidates
            self.assertEqual(metrics["false_positive_retrieval_rate"], 1.0)
            self.assertEqual(metrics["correct_abstention_rate"], 0.0)

            for m in ["Precision@5", "Recall@5", "MRR", "NDCG@5", "Hit Rate@5"]:
                val = metrics[m]
                self.assertGreaterEqual(val, 0.0, f"{strategy} {m} must be >= 0.0")
                self.assertLessEqual(val, 1.0, f"{strategy} {m} must be <= 1.0")

        # Answerable-query scores verification
        self.assertAlmostEqual(table["keyword"]["MRR"], 1.0)
        self.assertAlmostEqual(table["keyword"]["NDCG@5"], 1.0)
        self.assertAlmostEqual(table["keyword"]["Recall@5"], 1.0)

    def test_deterministic_repeated_benchmark_results(self):
        """Verifies that running the benchmark multiple times on the same data produces identical results."""
        run1 = run_retrieval_benchmark(
            document_id=self.doc_id,
            pages=self.pages,
            client=self.mock_client,
            k=5,
            queries=EVALUATION_QUERIES,
            db_path=self.db_path
        )
        run2 = run_retrieval_benchmark(
            document_id=self.doc_id,
            pages=self.pages,
            client=self.mock_client,
            k=5,
            queries=EVALUATION_QUERIES,
            db_path=self.db_path
        )
        self.assertEqual(run1["comparison_table"], run2["comparison_table"])
        self.assertEqual(len(run1["per_query_results"]), len(run2["per_query_results"]))

    def test_run_latency_benchmark_isolated_phases(self):
        """Verifies that latency benchmark isolates embedding generation, SQL query, cosine math, and RRF fusion."""
        latency_result = run_latency_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            query="What is Power Usage Effectiveness PUE in hyperscale data centers?",
            iterations=3,
            client=self.mock_client,
            db_path=self.db_path
        )

        self.assertEqual(latency_result["execution_mode"], "sequential")
        self.assertTrue(latency_result["is_mock_embedding"])

        # Check isolated timing phases
        phases = [
            "keyword_retrieval_mean_ms",
            "embedding_generation_mean_ms",
            "sqlite_chunk_fetch_mean_ms",
            "deserialization_and_cosine_math_mean_ms",
            "rrf_fusion_mean_ms",
            "complete_semantic_retrieval_mean_ms",
            "complete_hybrid_retrieval_mean_ms",
            "warm_query_indexing_check_ms",
            "cold_first_query_indexing_ms"
        ]
        for phase in phases:
            self.assertIn(phase, latency_result)
            self.assertGreater(latency_result[phase], 0.0, f"{phase} must be positive")

        # Cold indexing must take longer than warm check
        self.assertGreaterEqual(
            latency_result["cold_first_query_indexing_ms"],
            latency_result["warm_query_indexing_check_ms"]
        )

    def test_candidate_pool_expansion_in_hybrid_retrieval(self):
        """
        Verifies that retrieve_hybrid_chunks queries candidate_pool_size = max(top_k * 3, 15)
        so that passages ranked beyond top-K in individual retrievers can be considered by RRF.
        """
        from ai.retrieval_service import retrieve_hybrid_chunks
        from unittest.mock import patch

        with patch("ai.retrieval_service.retrieve_relevant_chunks") as mock_kw, \
             patch("ai.retrieval_service.retrieve_semantic_chunks") as mock_sem:
            mock_kw.return_value = [{"chunk_id": 1, "text": "t1", "page_number": 1, "chunk_index": 0}]
            mock_sem.return_value = [{"chunk_id": 1, "text": "t1", "page_number": 1, "chunk_index": 0}]

            _ = retrieve_hybrid_chunks(
                document_id=self.doc_id,
                query="consensus protocols",
                pages=self.pages,
                top_k=5,
                client=self.mock_client,
                file_path=self.pdf_path,
                db_path=self.db_path
            )

            # Assert top_k passed to underlying retrievers was expanded (max(5 * 3, 15) = 15)
            mock_kw.assert_called_once()
            _, kw_kwargs = mock_kw.call_args
            self.assertEqual(kw_kwargs.get("top_k"), 15)

            mock_sem.assert_called_once()
            _, sem_kwargs = mock_sem.call_args
            self.assertEqual(sem_kwargs.get("top_k"), 15)

    def test_evidence_integrity_on_retrieved_hybrid_results(self):
        """Verifies that actual chunks returned by retrieve_hybrid_chunks pass evidence integrity checks."""
        from ai.retrieval_service import retrieve_hybrid_chunks
        result = retrieve_hybrid_chunks(
            document_id=self.doc_id,
            query="What is Power Usage Effectiveness?",
            pages=self.pages,
            top_k=5,
            client=self.mock_client,
            file_path=self.pdf_path,
            db_path=self.db_path
        )
        chunks = result["chunks"]
        self.assertGreater(len(chunks), 0)

        integrity = validate_evidence_integrity(
            chunks,
            expected_document_id=self.doc_id,
            max_pages=5
        )
        self.assertTrue(integrity["valid"])
        self.assertEqual(len(integrity["issues"]), 0)
        self.assertTrue(integrity["document_id_consistent"])
        self.assertFalse(integrity["has_duplicates"])
        self.assertTrue(integrity["metadata_complete"])

    def test_benchmark_with_custom_k(self):
        """Tests running retrieval benchmark with k=3."""
        result = run_retrieval_benchmark(
            document_id=self.doc_id,
            pages=self.pages,
            client=self.mock_client,
            k=3,
            queries=EVALUATION_QUERIES[:3],
            db_path=self.db_path
        )
        self.assertEqual(result["k"], 3)
        self.assertEqual(result["query_count"], 3)
        for strat in ["keyword", "semantic", "hybrid"]:
            self.assertIn("Precision@3", result["comparison_table"][strat])
            self.assertIn("Recall@3", result["comparison_table"][strat])

    def test_create_evaluation_pdf_bytes_custom_pages(self):
        """Tests generating a custom PDF with 2 custom pages."""
        custom_pages = [
            {"page_number": 1, "text": "Custom first page text."},
            {"page_number": 2, "text": "Custom second page content."}
        ]
        pdf_bytes = create_evaluation_pdf_bytes(custom_pages)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            extraction = extract_text_from_pdf(tmp_path)
            self.assertTrue(extraction["success"])
            self.assertEqual(extraction["total_pages"], 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestEndToEndQAEvaluationHarness(unittest.TestCase):
    """
    Integration tests for the complete End-to-End QA Evaluation Benchmark Harness.
    Exercises retrieval, evidence normalization, structured answer parsing,
    citation verification, generation-level abstention, and latency instrumentation.
    Zero external or billable API calls: strictly offline and deterministic.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_eval_e2e.db")
        init_db(self.db_path)

        # Primary Document (Document ID 1)
        pdf_bytes = create_evaluation_pdf_bytes()
        self.pdf_path = os.path.join(self.test_dir, "eval_primary.pdf")
        with open(self.pdf_path, "wb") as f:
            f.write(pdf_bytes)

        extraction = extract_text_from_pdf(self.pdf_path)
        self.pages = extraction["pages"]

        self.doc_id = insert_document(
            filename="eval_primary.pdf",
            original_filename="primary.pdf",
            page_count=5,
            character_count=sum(len(p["text"]) for p in self.pages),
            status="processed",
            db_path=self.db_path
        )

        # Secondary Distractor Document (Document ID 2)
        distractor_bytes = create_distractor_pdf_bytes()
        self.distractor_path = os.path.join(self.test_dir, "eval_distractor.pdf")
        with open(self.distractor_path, "wb") as f:
            f.write(distractor_bytes)

        distractor_ext = extract_text_from_pdf(self.distractor_path)
        self.distractor_pages = distractor_ext["pages"]

        self.distractor_doc_id = insert_document(
            filename="eval_distractor.pdf",
            original_filename="distractor.pdf",
            page_count=2,
            character_count=sum(len(p["text"]) for p in self.distractor_pages),
            status="processed",
            db_path=self.db_path
        )

        self.mock_qa_client = MockEndToEndQAClient()

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_dataset_attributes_and_coverage(self):
        """Verifies END_TO_END_EVALUATION_DATASET covers all 9 categories with valid schema."""
        self.assertEqual(len(END_TO_END_EVALUATION_DATASET), 9)

        expected_categories = {
            "factual_single",
            "multi_passage",
            "cross_page",
            "ambiguous",
            "misleading_distractor",
            "prompt_injection",
            "insufficient_partial",
            "unanswerable_out_of_domain",
            "cross_document_leakage"
        }
        dataset_categories = {q["category"] for q in END_TO_END_EVALUATION_DATASET}
        self.assertEqual(dataset_categories, expected_categories)

        for q in END_TO_END_EVALUATION_DATASET:
            self.assertIn("question_id", q)
            self.assertIn("document_id", q)
            self.assertIn("question_text", q)
            self.assertIn("category", q)
            self.assertIn("is_answerable", q)
            self.assertIn("should_abstain", q)
            self.assertIn("target_pages", q)
            self.assertIn("expected_answer_facts", q)
            self.assertIn("relevant_passages", q)
            self.assertIn("expected_citations", q)
            self.assertIn("forbidden_hallucinations", q)
            self.assertIn("fact_evidence_mapping", q)
            self.assertIsInstance(q["is_answerable"], bool)
            self.assertIsInstance(q["should_abstain"], bool)

            if q["is_answerable"]:
                self.assertFalse(q["should_abstain"])
                self.assertGreater(len(q["expected_answer_facts"]), 0)
                self.assertGreater(len(q["expected_citations"]), 0)
                self.assertGreater(len(q["fact_evidence_mapping"]), 0)
            else:
                self.assertTrue(q["should_abstain"])
                self.assertEqual(len(q["expected_answer_facts"]), 0)
                self.assertEqual(len(q["expected_citations"]), 0)

    def test_distractor_pdf_creation_and_extraction(self):
        """Verifies create_distractor_pdf_bytes generates a valid 2-page PDF."""
        distractor_bytes = create_distractor_pdf_bytes()
        self.assertTrue(distractor_bytes.startswith(b"%PDF-1.4"))
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(distractor_bytes)
            tmp_path = tmp.name
        try:
            extraction = extract_text_from_pdf(tmp_path)
            self.assertTrue(extraction["success"])
            self.assertEqual(extraction["total_pages"], 2)
            self.assertIn("mRNA-1345", extraction["pages"][0]["text"])
            self.assertIn("Nirsevimab", extraction["pages"][1]["text"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_end_to_end_qa_benchmark_execution(self):
        """Executes run_end_to_end_qa_benchmark and verifies all metrics sections."""
        report = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=self.mock_qa_client,
            db_path=self.db_path
        )
        self.assertEqual(report["total_queries"], 9)
        self.assertEqual(report["execution_mode"], "sequential")
        self.assertTrue(report["is_mocked_client"])

        # Abstention metrics verification
        abs_stats = report["abstention_metrics"]
        self.assertEqual(abs_stats["total_queries"], 9)
        self.assertEqual(abs_stats["required_abstentions"], 3)
        self.assertEqual(abs_stats["true_abstentions"], 3)
        self.assertEqual(abs_stats["false_answers"], 0)
        self.assertAlmostEqual(abs_stats["abstention_precision"], 1.0)
        self.assertAlmostEqual(abs_stats["abstention_recall"], 1.0)
        self.assertAlmostEqual(abs_stats["false_answer_rate"], 0.0)
        self.assertAlmostEqual(abs_stats["answer_coverage"], 6.0 / 9.0)
        self.assertAlmostEqual(abs_stats["selective_accuracy"], 1.0)

        # Answer quality metrics verification
        qual_stats = report["answer_quality_metrics"]
        self.assertEqual(qual_stats["answerable_queries"], 6)
        self.assertEqual(qual_stats["unanswerable_queries"], 3)
        self.assertAlmostEqual(qual_stats["mean_reference_fact_coverage"], 1.0)
        self.assertEqual(qual_stats["queries_with_hallucinations"], 0)
        self.assertEqual(qual_stats["semantic_support_evaluation"], "not_evaluated")

        # Citation metrics verification
        cit_stats = report["citation_metrics"]
        self.assertEqual(cit_stats["evaluated_query_count"], 9)
        self.assertAlmostEqual(cit_stats["mean_citation_precision"], 1.0)
        self.assertAlmostEqual(cit_stats["mean_citation_recall"], 1.0)

        # Latency statistics verification
        lat_stats = report["latency_percentiles"]
        self.assertEqual(lat_stats["count"], 9)
        self.assertGreater(lat_stats["mean"], 0.0)
        self.assertGreater(lat_stats["p50"], 0.0)
        self.assertGreater(lat_stats["p95"], 0.0)

    def test_end_to_end_qa_benchmark_deterministic_repeatability(self):
        """Verifies repeated benchmark runs produce deterministic results."""
        rep1 = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=self.mock_qa_client,
            db_path=self.db_path
        )
        rep2 = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=self.mock_qa_client,
            db_path=self.db_path
        )
        self.assertEqual(
            rep1["abstention_metrics"]["abstention_precision"],
            rep2["abstention_metrics"]["abstention_precision"]
        )
        self.assertEqual(
            rep1["answer_quality_metrics"]["mean_reference_fact_coverage"],
            rep2["answer_quality_metrics"]["mean_reference_fact_coverage"]
        )

    def test_cross_document_isolation_in_e2e_evaluation(self):
        """q_e2e_09 queries Document 1 for facts in Document 2; verifies system strictly abstains."""
        q_leakage = [q for q in END_TO_END_EVALUATION_DATASET if q["category"] == "cross_document_leakage"]
        self.assertEqual(len(q_leakage), 1)

        report = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=self.mock_qa_client,
            queries=q_leakage,
            db_path=self.db_path
        )
        record = report["evaluated_records"][0]
        self.assertTrue(record["is_abstention"])
        self.assertTrue(record["abstention_correct"])
        self.assertTrue(record["is_correct"])

    def test_prompt_injection_safety_in_e2e_evaluation(self):
        """
        q_e2e_06 contains jailbreak text alongside a legitimate question.
        Verifies system ignores injection instruction, answers legitimate question using
        trusted evidence, produces valid citations, and avoids forbidden output.
        """
        q_injection = [q for q in END_TO_END_EVALUATION_DATASET if q["category"] == "prompt_injection"]
        self.assertEqual(len(q_injection), 1)

        report = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=self.mock_qa_client,
            queries=q_injection,
            db_path=self.db_path
        )
        record = report["evaluated_records"][0]

        # 1. Malicious instruction ignored and forbidden output avoided
        self.assertEqual(record["forbidden_hallucinations_detected"], [])
        self.assertNotIn("candy", record["answer"].lower())
        self.assertNotIn("[page 99]", record["answer"].lower())

        # 2. Legitimate question answered (system must not erroneously abstain)
        self.assertFalse(record["is_abstention"])
        self.assertFalse(record["should_abstain"])
        self.assertTrue(record["abstention_correct"])

        # 3. Answer grounded in trusted document evidence
        self.assertGreaterEqual(record["reference_fact_coverage"], 1.0)
        self.assertIn("residual skip connections", record["answer"].lower())
        self.assertEqual(record["facts_missing"], [])
        self.assertTrue(record["is_correct"])

        # 4. Valid citations produced
        self.assertGreater(len(record["citations"]), 0)
        self.assertEqual(record["citations"][0]["page_number"], 4)
        self.assertAlmostEqual(record["citation_precision"], 1.0)
        self.assertAlmostEqual(record["citation_recall"], 1.0)
        self.assertEqual(record["citation_status"], "verified")
        self.assertEqual(record["correctness_proxy"], "deterministic_reference_fact_matching")
        self.assertEqual(record["semantic_support"], "not_evaluated")

    def test_custom_client_malformed_response_fails_closed(self):
        """When client returns malformed JSON, system fails closed and evaluation catches it."""
        from unittest.mock import MagicMock
        bad_client = MockEndToEndQAClient()
        bad_mock_resp = MagicMock()
        bad_mock_resp.text = "{malformed json without brackets"
        bad_client.models.generate_content = MagicMock(return_value=bad_mock_resp)

        q_single = [END_TO_END_EVALUATION_DATASET[0]]
        report = run_end_to_end_qa_benchmark(
            document_id=self.doc_id,
            file_path=self.pdf_path,
            pages=self.pages,
            client=bad_client,
            queries=q_single,
            db_path=self.db_path
        )
        rec = report["evaluated_records"][0]
        self.assertFalse(rec["is_correct"])
        self.assertEqual(rec["citation_status"], "invalid")


if __name__ == "__main__":
    unittest.main()
