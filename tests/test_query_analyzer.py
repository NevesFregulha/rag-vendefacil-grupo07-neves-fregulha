"""Testes do analisador de perguntas e extração de filtros."""

import unittest

from langchain_core.documents import Document

from src.query_analyzer import extract_filters, metadata_vocabulary, validate_filters


class QueryAnalyzerTests(unittest.TestCase):
    def test_extracts_doc_type(self) -> None:
        filters = extract_filters("Quais tickets estao abertos?")
        self.assertEqual(filters["doc_type"], "ticket")

    def test_extracts_state_from_full_name(self) -> None:
        filters = extract_filters("Quais lojas estão localizadas em Minas Gerais?")
        self.assertEqual(filters, {"doc_type": "store", "state": "MG"})

    def test_extracts_state_from_uf(self) -> None:
        filters = extract_filters("Existem tickets abertos no estado de SP?")
        self.assertEqual(filters["state"], "SP")

    def test_extracts_module_from_synonym(self) -> None:
        filters = extract_filters("Quais erros ocorreram no módulo de PDV?")
        self.assertEqual(filters["module"], "pdv")

    def test_extracts_module_ecommerce_synonym(self) -> None:
        filters = extract_filters("Teve algum problema na loja online dos clientes?")
        self.assertEqual(filters["module"], "ecommerce")

    def test_extracts_customer_id(self) -> None:
        filters = extract_filters("Quais tickets o cliente CUST001 abriu?")
        self.assertEqual(filters["customer_id"], "CUST001")

    def test_extracts_customer_id_case_insensitive(self) -> None:
        filters = extract_filters("Histórico do cliente cust0042")
        self.assertEqual(filters["customer_id"], "CUST0042")

    def test_extracts_priority_with_accent_variation(self) -> None:
        filters = extract_filters("Quais tickets tem prioridade critica?")
        self.assertEqual(filters["priority"], "Crítica")

    def test_extracts_priority_media(self) -> None:
        filters = extract_filters("Liste os chamados de prioridade média")
        self.assertEqual(filters["priority"], "Média")

    def test_extracts_multiple_filters_together(self) -> None:
        filters = extract_filters(
            "Quais tickets de prioridade alta do módulo de estoque em Minas Gerais "
            "para o cliente CUST010?"
        )
        self.assertEqual(
            filters,
            {
                "doc_type": "ticket",
                "state": "MG",
                "module": "estoque",
                "customer_id": "CUST010",
                "priority": "Alta",
            },
        )

    def test_extracts_product_doc_type_without_other_filters(self) -> None:
        filters = extract_filters("Quais produtos a VendeFácil oferece?")
        self.assertEqual(filters, {"doc_type": "product"})

    def test_returns_empty_dict_for_blank_question(self) -> None:
        self.assertEqual(extract_filters(""), {})
        self.assertEqual(extract_filters("   "), {})

    def test_validates_filters_against_corpus_values(self) -> None:
        documents = [
            Document(page_content="ticket", metadata={"state": "MG", "module": "estoque"})
        ]
        vocabulary = metadata_vocabulary(documents)
        self.assertEqual(
            validate_filters({"state": "mg", "module": "Estoque"}, vocabulary),
            {"state": "MG", "module": "estoque"},
        )

    def test_discards_value_absent_from_corpus(self) -> None:
        vocabulary = {"state": {"MG"}}
        self.assertEqual(validate_filters({"state": "AC"}, vocabulary), {})


if __name__ == "__main__":
    unittest.main()
