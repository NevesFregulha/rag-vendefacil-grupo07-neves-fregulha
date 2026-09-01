"""Testes de integracao do fluxo analyzer, filtros, BM25 e RRF."""

import unittest

from langchain_core.documents import Document

from src.retrieval_check import COMPARISON_QUESTIONS, comparison_report
from src.retrieve import hybrid_search


class _Docstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents[document_id]


class _Vectorstore:
    def __init__(self, documents):
        self.index_to_docstore_id = {index: index for index in range(len(documents))}
        self.docstore = _Docstore(documents)

    def similarity_search(self, question, *, k, fetch_k, filter):
        documents = list(self.docstore.documents)
        if filter:
            documents = [
                document
                for document in documents
                if all(document.metadata.get(key) == value for key, value in filter.items())
            ]
        return documents[:k]


class RetrievalIntegrationTests(unittest.TestCase):
    def setUp(self):
        combinations = [
            ("MG", "estoque"), ("SP", "pdv"), ("RJ", "pay"), ("BA", "analytics")
        ]
        self.documents = [
            Document(
                page_content=f"Ticket de {state} no modulo {module}",
                metadata={
                    "chunk_id": f"ticket-{state}-{module}",
                    "source_file": "tickets.jsonl",
                    "doc_type": "ticket",
                    "state": state,
                    "module": module,
                },
            )
            for state, module in combinations
        ]
        self.store = _Vectorstore(self.documents)

    def test_three_specific_queries_only_return_matching_metadata(self):
        expected = [("MG", "estoque"), ("SP", "pdv"), ("RJ", "pay")]
        for question, (state, module) in zip(COMPARISON_QUESTIONS, expected):
            with self.subTest(question=question):
                results = hybrid_search(self.store, question, filters=None)
                self.assertTrue(results)
                self.assertTrue(
                    all(doc.metadata["state"] == state and doc.metadata["module"] == module for doc in results)
                )

    def test_report_contains_filtered_and_unfiltered_columns(self):
        report = comparison_report(self.store, k=2)
        self.assertIn("| Sem filtro | Com filtro |", report)
        self.assertEqual(report.count("## "), 3)


if __name__ == "__main__":
    unittest.main()
