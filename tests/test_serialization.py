"""
Unit Tests for Vector Embedding Serialization Helpers
Tests IEEE 754 32-bit float serialization (Python struct '<f'), precision,
round-trip fidelity, NaN/Inf rejection, and corrupt BLOB handling.
Zero external API calls.
"""

import math
import struct
import unittest

from database.models import serialize_embedding, deserialize_embedding


class EmbeddingSerializationTestCase(unittest.TestCase):
    def test_roundtrip_serialization_success(self):
        """Verify vector serialization and deserialization roundtrip maintains float values."""
        original = [0.0, 1.0, -1.0, 0.123456, -987.654, 3.1415927]
        blob = serialize_embedding(original)

        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), len(original) * 4)  # 4 bytes per 32-bit float

        recovered = deserialize_embedding(blob)
        self.assertEqual(len(recovered), len(original))
        for orig, rec in zip(original, recovered):
            self.assertTrue(math.isclose(orig, rec, rel_tol=1e-5, abs_tol=1e-6))

    def test_float32_precision_characteristics(self):
        """Verify IEEE 754 32-bit float precision matches expected machine epsilon (~1e-7)."""
        precise_val = 0.12345678912345678
        blob = serialize_embedding([precise_val])
        recovered = deserialize_embedding(blob)[0]

        # 32-bit float has 24 bits of mantissa (~7 decimal digits)
        self.assertAlmostEqual(precise_val, recovered, places=6)

    def test_expected_dimension_validation(self):
        """Verify deserialize_embedding enforces expected dimension when specified."""
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = serialize_embedding(vec)

        recovered = deserialize_embedding(blob, expected_dimension=4)
        self.assertEqual(len(recovered), 4)

        with self.assertRaises(ValueError) as ctx:
            deserialize_embedding(blob, expected_dimension=768)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_serialize_rejects_empty_and_non_sequence(self):
        """Verify serialize_embedding rejects empty, None, and non-sequence inputs."""
        with self.assertRaises(ValueError):
            serialize_embedding([])
        with self.assertRaises(ValueError):
            serialize_embedding(None)
        with self.assertRaises(ValueError):
            serialize_embedding("not a sequence")

    def test_serialize_rejects_nan_and_infinity(self):
        """Verify serialize_embedding rejects NaN and Infinite values."""
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, float("nan"), 2.0])
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, float("inf"), 2.0])
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, float("-inf"), 2.0])

    def test_serialize_rejects_booleans_and_non_numbers(self):
        """Verify serialize_embedding rejects booleans, strings, and non-numbers."""
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, True, 2.0])
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, False, 2.0])
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, "string", 2.0])
        with self.assertRaises(ValueError):
            serialize_embedding([1.0, None, 2.0])

    def test_deserialize_rejects_non_bytes(self):
        """Verify deserialize_embedding rejects non-bytes objects."""
        with self.assertRaises(TypeError):
            deserialize_embedding([0.1, 0.2])
        with self.assertRaises(TypeError):
            deserialize_embedding("not bytes")
        with self.assertRaises(TypeError):
            deserialize_embedding(None)

    def test_deserialize_rejects_empty_blob(self):
        """Verify deserialize_embedding rejects empty BLOB."""
        with self.assertRaises(ValueError):
            deserialize_embedding(b"")

    def test_deserialize_rejects_misaligned_length(self):
        """Verify deserialize_embedding rejects byte lengths that are not multiples of 4."""
        # 7 bytes is not a multiple of 4
        with self.assertRaises(ValueError) as ctx:
            deserialize_embedding(b"1234567")
        self.assertIn("multiple of 4", str(ctx.exception).lower())

    def test_deserialize_rejects_corrupted_nan_blob(self):
        """Verify deserialize_embedding rejects BLOBs that unpack to NaN or Inf."""
        # Pack an IEEE 754 NaN float
        nan_bytes = struct.pack("<f", float("nan"))
        with self.assertRaises(ValueError) as ctx:
            deserialize_embedding(nan_bytes)
        self.assertIn("non-finite", str(ctx.exception).lower())
