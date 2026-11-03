"""
Unit Tests for Semantic Chunking Module
Validates paragraph- and sentence-preserving chunking, boundary handling,
overlap behavior, deterministic ordering, and backward compatibility.
"""

import unittest
from pdf_processing.chunker import (
    chunk_extracted_pages,
    semantic_chunk_text,
    retrieve_relevant_chunks,
    is_broad_query,
    get_chunk_size,
    get_chunk_overlap,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


class SemanticChunkerTestCase(unittest.TestCase):
    def test_empty_input(self):
        """Test that empty page lists or blank pages produce zero chunks."""
        self.assertEqual(chunk_extracted_pages([]), [])
        self.assertEqual(chunk_extracted_pages([{"page_number": 1, "text": ""}]), [])
        self.assertEqual(chunk_extracted_pages([{"page_number": 1, "text": "   \n\n\t  "}]), [])
        self.assertEqual(semantic_chunk_text(""), [])
        self.assertEqual(semantic_chunk_text("    \n\t  "), [])

    def test_one_short_paragraph(self):
        """Test a short paragraph that fits entirely within the chunk size."""
        text = "This is a concise single paragraph well below the target chunk size limit."
        pages = [{"page_number": 1, "text": text}]
        chunks = chunk_extracted_pages(pages, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["char_count"], len(text))

    def test_multiple_paragraphs_preserved(self):
        """Test that multiple paragraphs are kept intact and not split prematurely."""
        para1 = "This is the first paragraph. It introduces the subject matter cleanly."
        para2 = "This is the second paragraph. It provides complementary context."
        text = f"{para1}\n\n{para2}"
        pages = [{"page_number": 1, "text": text}]

        # With generous chunk_size, both paragraphs fit in one chunk
        chunks = chunk_extracted_pages(pages, chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertIn(para1, chunks[0]["text"])
        self.assertIn(para2, chunks[0]["text"])

        # With smaller chunk_size, paragraphs split on the paragraph boundary
        chunks_small = chunk_extracted_pages(pages, chunk_size=100, chunk_overlap=30)
        self.assertGreaterEqual(len(chunks_small), 2)
        # First chunk should contain para1 without chopping mid-word
        self.assertIn("This is the first paragraph", chunks_small[0]["text"])

    def test_multiple_sentences_preserved(self):
        """Test that chunking splits on sentence boundaries rather than cutting mid-sentence."""
        s1 = "Neural networks require gradient descent optimization."
        s2 = "Stochastic gradient descent updates weights per mini-batch."
        s3 = "Adam optimizer computes adaptive learning rates from moments."
        s4 = "Weight decay provides L2 regularization to prevent overfitting."
        text = f"{s1} {s2} {s3} {s4}"

        chunks = semantic_chunk_text(text, chunk_size=120, chunk_overlap=40)
        self.assertGreater(len(chunks), 1)

        for c in chunks:
            # Check that chunks end cleanly on punctuation, not mid-word
            self.assertTrue(
                c.endswith((".", "!", "?")),
                f"Chunk does not end on sentence boundary: {c!r}"
            )

    def test_long_paragraph_and_long_sentence_handling(self):
        """Test that oversized paragraphs and long sentences are divided without losing text."""
        long_sentence = "Extremely long run-on technical sentence with numerous dependent clauses " * 20
        chunks = semantic_chunk_text(long_sentence, chunk_size=300, chunk_overlap=80)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 300)
            self.assertGreater(len(c), 0)

        # Monolithic continuous token without spaces
        monolith = "A" * 1200
        mono_chunks = semantic_chunk_text(monolith, chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(mono_chunks), 3)
        self.assertEqual(len(mono_chunks[0]), 500)
        self.assertEqual(len(mono_chunks[1]), 500)
        self.assertEqual(len(mono_chunks[2]), 400)

    def test_overlap_behavior(self):
        """Test that consecutive chunks share overlapping content."""
        sentences = [f"Numbered fact statement {i} about artificial intelligence." for i in range(1, 15)]
        text = " ".join(sentences)
        chunks = semantic_chunk_text(text, chunk_size=250, chunk_overlap=80)

        self.assertGreater(len(chunks), 2)
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # Next chunk should start with or contain the tail of current chunk
            tail_words = [w for w in curr_chunk.split()[-4:] if len(w) > 3]
            overlap_found = any(w in next_chunk for w in tail_words)
            self.assertTrue(
                overlap_found,
                f"No overlapping text detected between chunk {i} and {i+1}:\n"
                f"Chunk {i} tail: {curr_chunk[-60:]}\nChunk {i+1} head: {next_chunk[:60]}"
            )

    def test_page_number_preservation(self):
        """Test that each chunk accurately retains its original source page number."""
        pages = [
            {"page_number": 1, "text": "Page one content with introductory facts. " * 10},
            {"page_number": 2, "text": "Page two content with methodological details. " * 10},
            {"page_number": 3, "text": "Page three content with experimental results. " * 10},
        ]
        chunks = chunk_extracted_pages(pages, chunk_size=200, chunk_overlap=50)

        page_1_chunks = [c for c in chunks if c["page_number"] == 1]
        page_2_chunks = [c for c in chunks if c["page_number"] == 2]
        page_3_chunks = [c for c in chunks if c["page_number"] == 3]

        self.assertGreater(len(page_1_chunks), 1)
        self.assertGreater(len(page_2_chunks), 1)
        self.assertGreater(len(page_3_chunks), 1)

        for c in page_1_chunks:
            self.assertIn("Page one content", c["text"])
            self.assertNotIn("Page two content", c["text"])

        for c in page_2_chunks:
            self.assertIn("Page two content", c["text"])
            self.assertNotIn("Page one content", c["text"])

    def test_no_empty_chunks(self):
        """Test that chunking never produces empty or whitespace-only chunks."""
        text = "Valid paragraph.\n\n\n\n\n\nAnother valid paragraph with facts.\n\n\n"
        chunks = semantic_chunk_text(text, chunk_size=100, chunk_overlap=20)
        for c in chunks:
            self.assertTrue(len(c.strip()) > 0)

    def test_no_loss_of_meaningful_input_text(self):
        """Test that words from the original input text appear in the generated chunks."""
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        text = ". ".join(words) + "."
        chunks = semantic_chunk_text(text, chunk_size=30, chunk_overlap=10)
        combined = " ".join(chunks)

        for word in words:
            self.assertIn(word, combined)

    def test_stable_ordering_and_deterministic_results(self):
        """Test that repeated runs produce identical chunk boundaries and identifiers."""
        text = "Deterministic testing text. Paragraph one.\n\nParagraph two with analysis."
        pages = [{"page_number": 1, "text": text}]

        run1 = chunk_extracted_pages(pages, chunk_size=80, chunk_overlap=20)
        run2 = chunk_extracted_pages(pages, chunk_size=80, chunk_overlap=20)

        self.assertEqual(len(run1), len(run2))
        for c1, c2 in zip(run1, run2):
            self.assertEqual(c1["chunk_id"], c2["chunk_id"])
            self.assertEqual(c1["chunk_index"], c2["chunk_index"])
            self.assertEqual(c1["page_number"], c2["page_number"])
            self.assertEqual(c1["text"], c2["text"])
            self.assertEqual(c1["char_count"], c2["char_count"])

    def test_invalid_configuration_raises_value_error(self):
        """Test that invalid chunk size and overlap configurations raise clear ValueErrors."""
        # Non-positive chunk_size
        with self.assertRaises(ValueError):
            semantic_chunk_text("Valid text", chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            semantic_chunk_text("Valid text", chunk_size=-10, chunk_overlap=0)

        # Negative overlap
        with self.assertRaises(ValueError):
            semantic_chunk_text("Valid text", chunk_size=500, chunk_overlap=-1)

        # Overlap >= chunk_size
        with self.assertRaises(ValueError):
            semantic_chunk_text("Valid text", chunk_size=500, chunk_overlap=500)
        with self.assertRaises(ValueError):
            semantic_chunk_text("Valid text", chunk_size=500, chunk_overlap=600)

        # Same for chunk_extracted_pages
        with self.assertRaises(ValueError):
            chunk_extracted_pages([{"page_number": 1, "text": "Text"}], chunk_size=100, chunk_overlap=100)

    def test_default_config_resolution(self):
        """Test that default chunk_size and chunk_overlap resolve to positive values."""
        self.assertGreater(get_chunk_size(), 0)
        self.assertGreaterEqual(get_chunk_overlap(), 0)
        self.assertLess(get_chunk_overlap(), get_chunk_size())

    def test_existing_caller_backward_compatibility(self):
        """Test that existing callers and legacy tests remain 100% compatible."""
        pages = [
            {"page_number": 1, "text": "A" * 1200, "char_count": 1200},
            {"page_number": 2, "text": "Page two text content.", "char_count": 22}
        ]
        # Legacy call signature
        chunks = chunk_extracted_pages(pages, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertEqual(chunks[-1]["page_number"], 2)

        # Check existing retrieval ranking helper still functions on these chunks
        retrieved = retrieve_relevant_chunks(chunks, query="text content", top_k=2)
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()
