"""
Unit Tests for Google GenAI Embedding Service Module
Tests model selection, task types (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY),
vector validation, input handling, error translations, and mock retries.
Zero external API calls are made.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from ai.embedding_client import (
    embed_documents,
    embed_query,
    embed_single_document,
    validate_embedding_vector,
    EmbeddingError,
    TASK_TYPE_DOCUMENT,
    TASK_TYPE_QUERY,
    DEFAULT_EMBEDDING_MODEL,
)


class EmbeddingServiceTestCase(unittest.TestCase):
    def _create_mock_embedding_item(self, values):
        """Helper to create a mock ContentEmbedding item with .values attribute."""
        item = MagicMock()
        item.values = values
        return item

    def _create_mock_client(self, vectors):
        """Helper to create a mock GenAI client returning specific vectors."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [self._create_mock_embedding_item(v) for v in vectors]
        mock_client.models.embed_content.return_value = mock_response
        return mock_client

    def test_vector_validation_success(self):
        """Test that valid numeric vectors pass validation cleanly."""
        vec = [0.1, -0.25, 0.99, 100.0, 0.0]
        validated = validate_embedding_vector(vec)
        self.assertEqual(validated, [0.1, -0.25, 0.99, 100.0, 0.0])
        self.assertTrue(all(isinstance(v, float) for v in validated))

        # Enforced dimension matching
        validated_dim = validate_embedding_vector(vec, expected_dim=5)
        self.assertEqual(len(validated_dim), 5)

    def test_vector_validation_rejects_empty_and_non_sequence(self):
        """Test that empty or non-sequence vectors are rejected with ValueError."""
        with self.assertRaises(ValueError):
            validate_embedding_vector([])
        with self.assertRaises(ValueError):
            validate_embedding_vector(None)
        with self.assertRaises(ValueError):
            validate_embedding_vector("not_a_vector")

    def test_vector_validation_rejects_nan_and_infinity(self):
        """Test that NaN and infinity values are rejected."""
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, float("nan"), 0.5])
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, float("inf"), 0.5])
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, float("-inf"), 0.5])

    def test_vector_validation_rejects_booleans_and_non_numbers(self):
        """Test that booleans and non-numeric types are rejected."""
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, True, 0.5])
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, False, 0.5])
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, "string", 0.5])
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, None, 0.5])

    def test_vector_validation_rejects_dimension_mismatch(self):
        """Test that vector dimensional mismatches are rejected."""
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, 0.2, 0.3], expected_dim=4)
        with self.assertRaises(ValueError):
            validate_embedding_vector([0.1, 0.2, 0.3, 0.4], expected_dim=3)

    def test_embed_documents_empty_input_makes_no_api_call(self):
        """Test that passing empty texts makes zero API calls and returns empty list."""
        mock_client = MagicMock()
        res1 = embed_documents([], client=mock_client)
        self.assertEqual(res1, [])
        mock_client.models.embed_content.assert_not_called()

        res2 = embed_documents(None, client=mock_client)
        self.assertEqual(res2, [])
        mock_client.models.embed_content.assert_not_called()

    def test_embed_documents_success_with_default_model_and_task_type(self):
        """Test embedding document chunks uses gemini-embedding-001 and RETRIEVAL_DOCUMENT."""
        v1 = [0.1, 0.2, 0.3]
        v2 = [0.4, 0.5, 0.6]
        mock_client = self._create_mock_client([v1, v2])

        texts = ["First passage about machine learning.", "Second passage about deep networks."]
        vectors = embed_documents(texts, client=mock_client)

        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], v1)
        self.assertEqual(vectors[1], v2)

        # Assert correct model and task type passed to API
        mock_client.models.embed_content.assert_called_once()
        _, kwargs = mock_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(kwargs["contents"], texts)
        self.assertEqual(kwargs["config"].task_type, TASK_TYPE_DOCUMENT)

    def test_embed_documents_accepts_chunk_dictionaries(self):
        """Test that embed_documents accepts a list of chunk dicts containing 'text' key."""
        v1 = [0.1, 0.2]
        v2 = [0.3, 0.4]
        mock_client = self._create_mock_client([v1, v2])

        chunk_dicts = [
            {"chunk_id": 1, "page_number": 1, "text": "Chunk one content."},
            {"chunk_id": 2, "page_number": 1, "text": "Chunk two content."},
        ]
        vectors = embed_documents(chunk_dicts, client=mock_client)

        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0], v1)
        self.assertEqual(vectors[1], v2)

        _, kwargs = mock_client.models.embed_content.call_args
        self.assertEqual(kwargs["contents"], ["Chunk one content.", "Chunk two content."])

    def test_embed_documents_rejects_invalid_item_types(self):
        """Test that invalid chunk elements in the list raise TypeError."""
        mock_client = MagicMock()
        with self.assertRaises(TypeError):
            embed_documents(["Valid string", 12345], client=mock_client)
        with self.assertRaises(TypeError):
            embed_documents([{"no_text_key": "val"}], client=mock_client)

    def test_embed_documents_rejects_inconsistent_dimensions_in_batch(self):
        """Test that a batch returning inconsistent vector dimensions fails with EmbeddingError."""
        v1 = [0.1, 0.2, 0.3]
        v2 = [0.4, 0.5]  # Dim 2 instead of 3
        mock_client = self._create_mock_client([v1, v2])

        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(["Doc 1", "Doc 2"], client=mock_client)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_embed_documents_rejects_count_mismatch(self):
        """Test that a response returning fewer or more embeddings than inputs fails."""
        mock_client = self._create_mock_client([[0.1, 0.2]])  # Only 1 vector returned for 2 inputs
        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(["Doc 1", "Doc 2"], client=mock_client)
        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_embed_documents_model_override(self):
        """Test that custom model_name override is respected."""
        mock_client = self._create_mock_client([[0.1, 0.2]])
        custom_model = "custom-test-embedding-model"
        embed_documents(["Text"], client=mock_client, model_name=custom_model)

        _, kwargs = mock_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], custom_model)

    def test_embed_query_success_with_task_type(self):
        """Test embedding a user query uses RETRIEVAL_QUERY and returns single vector."""
        q_vec = [0.8, -0.5, 0.2]
        mock_client = self._create_mock_client([q_vec])

        query = "What is the training convergence rate?"
        vec = embed_query(query, client=mock_client)

        self.assertEqual(vec, q_vec)
        mock_client.models.embed_content.assert_called_once()
        _, kwargs = mock_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(kwargs["contents"], query)
        self.assertEqual(kwargs["config"].task_type, TASK_TYPE_QUERY)

    def test_embed_query_rejects_empty_string(self):
        """Test that empty or whitespace queries raise ValueError without calling API."""
        mock_client = MagicMock()
        with self.assertRaises(ValueError):
            embed_query("", client=mock_client)
        with self.assertRaises(ValueError):
            embed_query("    \t  \n  ", client=mock_client)
        with self.assertRaises(ValueError):
            embed_query(None, client=mock_client)
        mock_client.models.embed_content.assert_not_called()

    def test_embed_single_document_convenience(self):
        """Test embed_single_document returns a single vector."""
        doc_vec = [0.33, 0.66, 0.99]
        mock_client = self._create_mock_client([doc_vec])

        vec = embed_single_document("Single passage of text.", client=mock_client)
        self.assertEqual(vec, doc_vec)

        with self.assertRaises(ValueError):
            embed_single_document("   ", client=mock_client)

    def test_embed_documents_api_transient_error_retries_and_recovers(self):
        """Test that transient 503 server errors retry and succeed upon recovery."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [self._create_mock_embedding_item([0.1, 0.2])]

        # Fail once with 503, succeed on second attempt
        mock_client.models.embed_content.side_effect = [
            Exception("503 Service Unavailable"),
            mock_response
        ]

        with patch("ai.analyzer._default_sleep") as mock_sleep:
            vectors = embed_documents(["Text"], client=mock_client)

        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0], [0.1, 0.2])
        self.assertEqual(mock_client.models.embed_content.call_count, 2)

    def test_embed_documents_permanent_error_fails_immediately(self):
        """Test that non-retryable 401 unauthenticated errors fail fast without retry."""
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("401 Unauthorized: Invalid API key")

        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(["Text"], client=mock_client)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        self.assertIn("Invalid or missing Google Gemini API key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
