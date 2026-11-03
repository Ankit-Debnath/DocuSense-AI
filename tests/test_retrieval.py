"""
Unit Tests for Semantic Retrieval, Cosine Similarity, and Reciprocal Rank Fusion (RRF)
Tests vector math, error boundaries, model/dimension compatibility,
corrupted BLOB resilience, document-scoped filtering, and rank fusion.
Zero external API calls are made (Gemini client is mocked).
"""

import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ai.retrieval_service import (
    cosine_similarity,
    normalize_model_name,
    retrieve_semantic_chunks,
    reciprocal_rank_fusion,
    retrieve_hybrid_chunks,
    DEFAULT_RRF_K
)
from database.db import init_db
from database.models import (
    insert_document,
    save_document_chunks,
    serialize_embedding
)


class CosineSimilarityTestCase(unittest.TestCase):
    def test_identical_vectors_yield_one(self):
        """Identical vectors must return 1.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0, places=6)

    def test_orthogonal_vectors_yield_zero(self):
        """Orthogonal vectors must return 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0, places=6)

    def test_opposite_vectors_yield_minus_one(self):
        """Opposite vectors must return -1.0."""
        vec1 = [1.0, -1.0]
        vec2 = [-1.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0, places=6)

    def test_dimension_mismatch_raises_value_error(self):
        """Mismatched dimensions must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_empty_vectors_raise_value_error(self):
        """Empty vectors must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([], [])
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [])

    def test_zero_magnitude_vectors_raise_value_error(self):
        """Zero-magnitude vectors must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            cosine_similarity([0.0, 0.0], [1.0, 2.0])
        self.assertIn("zero-magnitude", str(ctx.exception).lower())

    def test_nan_and_inf_raise_value_error(self):
        """Vectors with NaN or Inf must raise ValueError."""
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan"), 1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cosine_similarity([float("inf"), 1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, float("-inf")])

    def test_invalid_types_raise_type_error(self):
        """Booleans, strings, and non-sequences must raise TypeError."""
        with self.assertRaises(TypeError):
            cosine_similarity("not a list", [1.0, 2.0])  # type: ignore
        with self.assertRaises(TypeError):
            cosine_similarity([True, 1.0], [1.0, 2.0])
        with self.assertRaises(TypeError):
            cosine_similarity([1.0, "text"], [1.0, 2.0])


class ModelNameNormalizationTestCase(unittest.TestCase):
    def test_normalizes_prefixes_and_casing(self):
        """Verify models/ prefix is cleanly stripped and case-normalized."""
        self.assertEqual(normalize_model_name("models/gemini-embedding-001"), "gemini-embedding-001")
        self.assertEqual(normalize_model_name("gemini-embedding-001"), "gemini-embedding-001")
        self.assertEqual(normalize_model_name("models/GEMINI-EMBEDDING-001"), "gemini-embedding-001")
        self.assertEqual(normalize_model_name(""), "")
        self.assertEqual(normalize_model_name(None), "")


class SemanticRetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_retrieval.db")
        init_db(self.db_path)

        # Create two separate test documents
        self.doc1_id = insert_document(
            filename="doc1.pdf",
            original_filename="doc1.pdf",
            page_count=2,
            character_count=500,
            status="processed",
            content_hash="hash_doc1",
            db_path=self.db_path
        )
        self.doc2_id = insert_document(
            filename="doc2.pdf",
            original_filename="doc2.pdf",
            page_count=1,
            character_count=250,
            status="processed",
            content_hash="hash_doc2",
            db_path=self.db_path
        )

        # Chunks for doc1:
        # chunk 0: high similarity with query [1.0, 0.0]
        # chunk 1: low similarity with query [1.0, 0.0]
        self.chunks_doc1 = [
            {
                "chunk_index": 0,
                "page_number": 1,
                "chunk_text": "Apollo program landed humans on the Moon.",
                "char_count": 41,
                "embedding": [0.99, 0.01],
                "embedding_dim": 2,
                "embedding_model": "gemini-embedding-001"
            },
            {
                "chunk_index": 1,
                "page_number": 2,
                "chunk_text": "Deep sea exploration uncovers marine organisms.",
                "char_count": 47,
                "embedding": [0.01, 0.99],
                "embedding_dim": 2,
                "embedding_model": "gemini-embedding-001"
            }
        ]
        save_document_chunks(self.doc1_id, self.chunks_doc1, db_path=self.db_path)

        # Chunks for doc2:
        self.chunks_doc2 = [
            {
                "chunk_index": 0,
                "page_number": 1,
                "chunk_text": "Doc 2 confidential chunk text.",
                "char_count": 30,
                "embedding": [1.0, 0.0],
                "embedding_dim": 2,
                "embedding_model": "gemini-embedding-001"
            }
        ]
        save_document_chunks(self.doc2_id, self.chunks_doc2, db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_semantic_retrieval_ranking_and_document_scoping(self):
        """Verify semantic retrieval ranks chunks descending by similarity and respects document scope."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        item = MagicMock()
        item.values = [1.0, 0.0]  # Query points strictly along dimension 0
        mock_resp.embeddings = [item]
        mock_client.models.embed_content.return_value = mock_resp

        results = retrieve_semantic_chunks(
            document_id=self.doc1_id,
            query="When did Apollo reach the moon?",
            top_k=2,
            client=mock_client,
            db_path=self.db_path
        )

        self.assertEqual(len(results), 2)
        # First chunk must be Apollo (higher similarity ~0.99)
        self.assertIn("Apollo", results[0]["text"])
        self.assertGreater(results[0]["similarity_score"], results[1]["similarity_score"])
        self.assertEqual(results[0]["document_id"], self.doc1_id)
        self.assertEqual(results[0]["page_number"], 1)

        # Document scoping check: doc2 chunk must NEVER appear in results
        result_texts = [r["text"] for r in results]
        self.assertNotIn("Doc 2 confidential chunk text.", result_texts)

    def test_dimension_mismatch_returns_empty_safely(self):
        """When query vector dimension != stored vector dimension, return empty without crashing."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        item = MagicMock()
        item.values = [1.0, 0.0, 0.5]  # 3 dimensions, but doc1 has 2 dimensions!
        mock_resp.embeddings = [item]
        mock_client.models.embed_content.return_value = mock_resp

        results = retrieve_semantic_chunks(
            document_id=self.doc1_id,
            query="Apollo moon landing",
            top_k=2,
            client=mock_client,
            db_path=self.db_path
        )
        self.assertEqual(results, [])

    def test_model_mismatch_returns_empty_safely(self):
        """When active embedding model != stored embedding model, return empty without crashing."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        item = MagicMock()
        item.values = [1.0, 0.0]
        mock_resp.embeddings = [item]
        mock_client.models.embed_content.return_value = mock_resp

        results = retrieve_semantic_chunks(
            document_id=self.doc1_id,
            query="Apollo moon landing",
            top_k=2,
            client=mock_client,
            model_name="different-embedding-model-v2",
            db_path=self.db_path
        )
        self.assertEqual(results, [])

    def test_corrupted_blob_is_skipped_cleanly(self):
        """When a chunk has a malformed BLOB, it is skipped and valid chunks are still scored."""
        bad_chunks = [
            {
                "chunk_index": 0,
                "page_number": 1,
                "chunk_text": "Valid chunk.",
                "char_count": 12,
                "embedding": [1.0, 0.0],
                "embedding_dim": 2,
                "embedding_model": "gemini-embedding-001"
            },
            {
                "chunk_index": 1,
                "page_number": 2,
                "chunk_text": "Corrupted chunk.",
                "char_count": 16,
                "embedding": b"corrupt_odd_bytes",  # Invalid 17-byte BLOB
                "embedding_dim": 2,
                "embedding_model": "gemini-embedding-001"
            }
        ]
        save_document_chunks(self.doc1_id, bad_chunks, db_path=self.db_path)

        mock_client = MagicMock()
        mock_resp = MagicMock()
        item = MagicMock()
        item.values = [1.0, 0.0]
        mock_resp.embeddings = [item]
        mock_client.models.embed_content.return_value = mock_resp

        results = retrieve_semantic_chunks(
            document_id=self.doc1_id,
            query="Valid test",
            top_k=2,
            client=mock_client,
            db_path=self.db_path
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Valid chunk.")


class ReciprocalRankFusionTestCase(unittest.TestCase):
    def test_chunk_in_both_rankings_has_higher_score(self):
        """A chunk appearing in both lexical and semantic lists gets contributions from both."""
        lexical = [
            {"page_number": 1, "chunk_index": 0, "text": "Common chunk A"},
            {"page_number": 1, "chunk_index": 1, "text": "Lexical-only B"}
        ]
        semantic = [
            {"page_number": 1, "chunk_index": 0, "text": "Common chunk A"},
            {"page_number": 2, "chunk_index": 2, "text": "Semantic-only C"}
        ]

        fused = reciprocal_rank_fusion(lexical, semantic, top_k=3, k=60)

        # Chunk A is rank 1 in both: RRF = 1/61 + 1/61 = 2/61 ~ 0.032787
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["text"], "Common chunk A")
        expected_a = (1.0 / 61) + (1.0 / 61)
        self.assertAlmostEqual(fused[0]["rrf_score"], expected_a, places=5)
        self.assertIn("lexical", fused[0]["sources"])
        self.assertIn("semantic", fused[0]["sources"])

        # Chunks B and C both rank 2 in single source: 1/62 ~ 0.016129
        expected_single = 1.0 / 62
        self.assertAlmostEqual(fused[1]["rrf_score"], expected_single, places=5)

    def test_duplicate_elimination(self):
        """A chunk present in both rankings must appear exactly once in the final result."""
        lexical = [{"page_number": 1, "chunk_index": 0, "text": "Duplicate chunk"}]
        semantic = [{"page_number": 1, "chunk_index": 0, "text": "Duplicate chunk"}]

        fused = reciprocal_rank_fusion(lexical, semantic, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "Duplicate chunk")

    def test_empty_lexical_returns_semantic_results(self):
        """When lexical results are empty, valid semantic results are returned directly."""
        semantic = [{"page_number": 1, "chunk_index": 0, "text": "Semantic chunk"}]
        fused = reciprocal_rank_fusion([], semantic, top_k=2)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "Semantic chunk")
        self.assertEqual(fused[0]["sources"], ["semantic"])

    def test_empty_semantic_returns_lexical_results(self):
        """When semantic results are empty, lexical results are returned directly."""
        lexical = [{"page_number": 1, "chunk_index": 0, "text": "Lexical chunk"}]
        fused = reciprocal_rank_fusion(lexical, [], top_k=2)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "Lexical chunk")
        self.assertEqual(fused[0]["sources"], ["lexical"])

    def test_top_k_limiting(self):
        """RRF output must be limited to top_k candidates."""
        lexical = [{"page_number": 1, "chunk_index": i, "text": f"Chunk {i}"} for i in range(10)]
        fused = reciprocal_rank_fusion(lexical, [], top_k=3)
        self.assertEqual(len(fused), 3)
