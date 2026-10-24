import io
import os
import shutil
import tempfile
import unittest
import pypdf
from pdf_processing.extractor import extract_text_from_pdf


class PDFExtractorTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary working directory for test files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary test files
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_pdf_with_text(self, filepath: str):
        """Creates a minimal valid PDF file with known text content."""
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
        with open(filepath, "wb") as f:
            f.write(pdf_content)

    def test_successful_text_extraction(self):
        """Test extracting text from a PDF with actual text contents."""
        filepath = os.path.join(self.test_dir, "sample_text.pdf")
        self._create_pdf_with_text(filepath)

        result = extract_text_from_pdf(filepath)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_pages"], 1)
        self.assertGreater(result["total_characters"], 0)
        self.assertTrue(result["has_extractable_text"])
        self.assertEqual(len(result["pages"]), 1)
        self.assertEqual(result["pages"][0]["page_number"], 1)
        self.assertIn("Hello World from PDF", result["pages"][0]["text"])

    def test_empty_text_pdf(self):
        """Test a PDF containing blank pages (like scanned pages with no text layer)."""
        filepath = os.path.join(self.test_dir, "blank.pdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_blank_page(width=100, height=100)
        with open(filepath, "wb") as f:
            writer.write(f)

        result = extract_text_from_pdf(filepath)
        self.assertTrue(result["success"])
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["total_characters"], 0)
        self.assertFalse(result["has_extractable_text"])
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["pages"][0]["text"], "")

    def test_encrypted_pdf(self):
        """Test extracting from a password-protected PDF."""
        filepath = os.path.join(self.test_dir, "protected.pdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret_pass_123")
        with open(filepath, "wb") as f:
            writer.write(f)

        result = extract_text_from_pdf(filepath)
        self.assertFalse(result["success"])
        self.assertIn("password-protected", result["error"].lower())

    def test_corrupted_pdf(self):
        """Test handling a corrupted or truncated PDF."""
        filepath = os.path.join(self.test_dir, "corrupt.pdf")
        with open(filepath, "wb") as f:
            # Header looks like PDF, but body is corrupted truncated garbage
            f.write(b"%PDF-1.4\nSome broken truncated bytes that fail to parse")

        result = extract_text_from_pdf(filepath)
        self.assertFalse(result["success"])
        self.assertIn("corrupted or damaged", result["error"].lower())

    def test_nonexistent_file(self):
        """Test handling a file path that does not exist."""
        filepath = os.path.join(self.test_dir, "missing.pdf")
        result = extract_text_from_pdf(filepath)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
