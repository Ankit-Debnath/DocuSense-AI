"""
Unit Tests for Google GenAI Embedding Safe Batching
Tests bounded batch sizes, order preservation, partial batches, failure handling,
dimension consistency across batches, and configuration defaults.
Zero external API calls.
"""

import unittest
from unittest.mock import MagicMock, patch

from ai.embedding_client import (
    embed_documents,
    get_embedding_batch_size,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EmbeddingError
)


class EmbeddingBatchingTestCase(unittest.TestCase):
    def _create_mock_item(self, values):
        item = MagicMock()
        item.values = values
        return item

    def test_default_batch_size_resolution(self):
        """Verify default batch size resolves to 50 or environment override."""
        self.assertEqual(get_embedding_batch_size(), DEFAULT_EMBEDDING_BATCH_SIZE)

        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "25"}):
            self.assertEqual(get_embedding_batch_size(), 25)

        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "invalid"}):
            self.assertEqual(get_embedding_batch_size(), DEFAULT_EMBEDDING_BATCH_SIZE)

        with patch.dict("os.environ", {"EMBEDDING_BATCH_SIZE": "-10"}):
            self.assertEqual(get_embedding_batch_size(), DEFAULT_EMBEDDING_BATCH_SIZE)

    def test_invalid_batch_size_raises_value_error(self):
        """Verify passing non-positive or non-int batch_size raises ValueError."""
        mock_client = MagicMock()
        with self.assertRaises(ValueError):
            embed_documents(["text"], client=mock_client, batch_size=0)
        with self.assertRaises(ValueError):
            embed_documents(["text"], client=mock_client, batch_size=-5)
        with self.assertRaises(ValueError):
            embed_documents(["text"], client=mock_client, batch_size="50")  # type: ignore

    def test_empty_input_makes_zero_api_calls(self):
        """Verify empty input does not invoke API."""
        mock_client = MagicMock()
        res = embed_documents([], client=mock_client, batch_size=10)
        self.assertEqual(res, [])
        mock_client.models.embed_content.assert_not_called()

    def test_batching_multiple_full_and_partial_batches(self):
        """Verify 125 inputs with batch_size=50 are partitioned into 50, 50, 25."""
        total_items = 125
        batch_size = 50
        texts = [f"Chunk number {i}" for i in range(total_items)]

        mock_client = MagicMock()
        # Mock embed_content to return distinct vectors matching the batch size
        def side_effect(model, contents, config):
            resp = MagicMock()
            resp.embeddings = [self._create_mock_item([float(i), float(i + 1)]) for i in range(len(contents))]
            return resp

        mock_client.models.embed_content.side_effect = side_effect

        results = embed_documents(texts, client=mock_client, batch_size=batch_size)

        self.assertEqual(len(results), total_items)
        self.assertEqual(mock_client.models.embed_content.call_count, 3)

        call_args_list = mock_client.models.embed_content.call_args_list
        # Batch 1: 50 items
        self.assertEqual(len(call_args_list[0].kwargs["contents"]), 50)
        self.assertEqual(call_args_list[0].kwargs["contents"][0], "Chunk number 0")
        self.assertEqual(call_args_list[0].kwargs["contents"][-1], "Chunk number 49")

        # Batch 2: 50 items
        self.assertEqual(len(call_args_list[1].kwargs["contents"]), 50)
        self.assertEqual(call_args_list[1].kwargs["contents"][0], "Chunk number 50")
        self.assertEqual(call_args_list[1].kwargs["contents"][-1], "Chunk number 99")

        # Batch 3: 25 items (partial final batch)
        self.assertEqual(len(call_args_list[2].kwargs["contents"]), 25)
        self.assertEqual(call_args_list[2].kwargs["contents"][0], "Chunk number 100")
        self.assertEqual(call_args_list[2].kwargs["contents"][-1], "Chunk number 124")

    def test_order_preservation_across_batches(self):
        """Verify vectors exactly maintain 1-to-1 correspondence across all batch boundaries."""
        texts = [f"Doc_{i}" for i in range(15)]
        mock_client = MagicMock()

        def side_effect(model, contents, config):
            resp = MagicMock()
            # Return vector containing the doc index as first element
            resp.embeddings = []
            for txt in contents:
                idx = int(txt.split("_")[1])
                resp.embeddings.append(self._create_mock_item([float(idx), 0.5]))
            return resp

        mock_client.models.embed_content.side_effect = side_effect

        results = embed_documents(texts, client=mock_client, batch_size=4)
        self.assertEqual(len(results), 15)
        for i, vec in enumerate(results):
            self.assertEqual(vec[0], float(i))

    def test_failure_in_intermediate_batch_raises_error(self):
        """Verify API failure in an intermediate batch raises EmbeddingError and halts."""
        texts = [f"Doc_{i}" for i in range(12)]
        mock_client = MagicMock()

        call_count = 0
        def side_effect(model, contents, config):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("API Rate limit or connection reset")
            resp = MagicMock()
            resp.embeddings = [self._create_mock_item([1.0, 2.0]) for _ in contents]
            return resp

        mock_client.models.embed_content.side_effect = side_effect

        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(texts, client=mock_client, batch_size=5)
        self.assertIn("failed", str(ctx.exception).lower())

    def test_batch_count_mismatch_raises_error(self):
        """Verify when API returns fewer embeddings than sent in a batch, EmbeddingError is raised."""
        texts = ["A", "B", "C", "D"]
        mock_client = MagicMock()

        # Send 4, return only 3
        resp = MagicMock()
        resp.embeddings = [self._create_mock_item([1.0, 2.0]) for _ in range(3)]
        mock_client.models.embed_content.return_value = resp

        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(texts, client=mock_client, batch_size=4)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_dimension_mismatch_across_batches_raises_error(self):
        """Verify vector dimension change across batches is detected and rejected."""
        texts = ["A", "B", "C", "D"]
        mock_client = MagicMock()

        call_count = 0
        def side_effect(model, contents, config):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                # 3 dimensions
                resp.embeddings = [self._create_mock_item([0.1, 0.2, 0.3]) for _ in contents]
            else:
                # 4 dimensions (mismatch!)
                resp.embeddings = [self._create_mock_item([0.1, 0.2, 0.3, 0.4]) for _ in contents]
            return resp

        mock_client.models.embed_content.side_effect = side_effect

        with self.assertRaises(EmbeddingError) as ctx:
            embed_documents(texts, client=mock_client, batch_size=2)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())
