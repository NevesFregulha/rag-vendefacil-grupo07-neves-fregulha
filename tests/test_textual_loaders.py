"""Testes dos loaders de JSONL, Markdown, PDF e TXT da TASK 03."""

import json
import tempfile
import unittest
from pathlib import Path

from src.loaders.jsonl_loader import load_jsonl_documents
from src.loaders.markdown_loader import (
    load_markdown_documents,
    load_markdown_file,
)
from src.loaders.pdf_loader import load_pdf_documents
from src.loaders.text_loader import load_text_documents, load_text_file
from src.metadata import validate_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


class JsonlLoaderTests(unittest.TestCase):
    def test_long_ticket_replicates_header_in_all_parts(self) -> None:
        ticket = {
            "ticket_id": "TCK-TESTE",
            "customer_id": "CUST001",
            "state": "MG",
            "module": "estoque",
            "title": "Falha de sincronização",
            "description": "Descrição longa. " * 80,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tickets.jsonl"
            path.write_text(json.dumps(ticket, ensure_ascii=False), encoding="utf-8")
            documents = load_jsonl_documents(path, chunk_size=250, chunk_overlap=30)

        self.assertGreater(len(documents), 1)
        self.assertTrue(all("Ticket: TCK-TESTE" in doc.page_content for doc in documents))
        self.assertTrue(all(doc.metadata["doc_type"] == "ticket" for doc in documents))
        validate_documents(documents)


class MarkdownLoaderTests(unittest.TestCase):
    def test_preserves_markdown_section_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual.md"
            path.write_text(
                "# Manual\n\n## Instalação\n\nPassos de instalação.\n",
                encoding="utf-8",
            )
            documents = load_markdown_file(path)

        self.assertTrue(any("Instalação" in doc.metadata["section"] for doc in documents))
        validate_documents(documents)


class TextLoaderTests(unittest.TestCase):
    def test_marks_customer_email_with_credentials_as_restricted(self) -> None:
        message = "De: cliente@empresa.com\n\nMinha senha temporaria e Segredo123."
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "customer_001.txt"
            path.write_text(message, encoding="utf-8")
            documents = load_text_file(path)

        self.assertTrue(documents)
        self.assertTrue(
            all(doc.metadata["sensitivity"] == "restrito" for doc in documents)
        )

    def test_separates_email_thread_before_size_split(self) -> None:
        thread = (
            "De: primeira@empresa.com\nData: 26/08/2026\n\nPrimeira mensagem.\n\n"
            "De: segunda@empresa.com\nData: 26/08/2026\n\nSegunda mensagem."
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "thread.txt"
            path.write_text(thread, encoding="utf-8")
            documents = load_text_file(path)

        self.assertEqual(len(documents), 2)
        self.assertEqual(
            {doc.metadata["message_index"] for doc in documents},
            {1, 2},
        )
        validate_documents(documents)


class CorpusLoaderTests(unittest.TestCase):
    def test_loads_and_validates_complete_textual_corpus(self) -> None:
        documents = []
        documents.extend(
            load_jsonl_documents(DATA_DIR / "semi_structured" / "tickets.jsonl")
        )
        documents.extend(load_markdown_documents(DATA_DIR / "unstructured"))
        documents.extend(load_pdf_documents(DATA_DIR / "unstructured" / "policies"))
        documents.extend(load_text_documents(DATA_DIR / "unstructured" / "emails"))

        self.assertGreater(len(documents), 0)
        self.assertEqual(
            {"ticket", "manual", "ata", "policy", "email"},
            {doc.metadata["doc_type"] for doc in documents},
        )
        validate_documents(documents)


if __name__ == "__main__":
    unittest.main()
