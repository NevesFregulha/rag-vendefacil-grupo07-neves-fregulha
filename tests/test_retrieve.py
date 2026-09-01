"""Testes da recuperacao densa com filtro."""

import unittest

from langchain_core.documents import Document

from src.retrieve import (
    bm25_search,
    dense_search,
    hybrid_search,
    matches_filters,
    reciprocal_rank_fusion,
)


class _FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents[document_id]


class _FakeVectorstore:
    def __init__(self, documents):
        self.index_to_docstore_id = {position: position for position in range(len(documents))}
        self.docstore = _FakeDocstore(documents)
        self.last_call = None

    def similarity_search(self, question, **kwargs):
        self.last_call = (question, kwargs)
        selected = list(self.docstore.documents)
        if kwargs.get("filter"):
            selected = [doc for doc in selected if matches_filters(doc, kwargs["filter"])]
        return selected[: kwargs["k"]]


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="MG estoque", metadata={"state": "MG", "module": "estoque"}),
            Document(page_content="SP estoque", metadata={"state": "SP", "module": "estoque"}),
        ]

    def test_applies_analyzed_filter_and_fetch_k(self):
        store = _FakeVectorstore(self.documents)
        result = dense_search(store, "tickets de Minas Gerais sobre estoque", k=1, fetch_k=500)
        self.assertEqual(result[0].page_content, "MG estoque")
        self.assertEqual(store.last_call[1]["filter"], {"state": "MG", "module": "estoque"})
        self.assertEqual(store.last_call[1]["fetch_k"], 500)

    def test_ignores_filter_value_that_does_not_exist(self):
        store = _FakeVectorstore(self.documents)
        dense_search(store, "consulta", filters={"state": "AC"})
        self.assertIsNone(store.last_call[1]["filter"])


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="Falha generica no pagamento", metadata={"chunk_id": "dense", "state": "MG"}),
            Document(page_content="Erro E-PDV-042 no caixa", metadata={"chunk_id": "exact", "state": "MG"}),
            Document(page_content="Erro E-PDV-042 fora do estado", metadata={"chunk_id": "sp", "state": "SP"}),
        ]

    def test_bm25_finds_exact_error_code_and_prefilters(self):
        result = bm25_search(
            self.documents, "E-PDV-042", k=2, filters={"state": "MG"}
        )
        self.assertEqual(result[0].metadata["chunk_id"], "exact")
        self.assertTrue(all(doc.metadata["state"] == "MG" for doc in result))

    def test_rrf_rewards_document_present_in_both_rankings(self):
        first, second, third = self.documents
        result = reciprocal_rank_fusion([[first, second], [second, third]], limit=3)
        self.assertEqual(result[0].metadata["chunk_id"], "exact")

    def test_hybrid_search_returns_fused_ranking(self):
        store = _FakeVectorstore(self.documents)
        result = hybrid_search(store, "E-PDV-042 em MG", k=2)
        self.assertEqual(
            {doc.metadata["chunk_id"] for doc in result}, {"dense", "exact"}
        )
        self.assertTrue(all(doc.metadata["state"] == "MG" for doc in result))


if __name__ == "__main__":
    unittest.main()
