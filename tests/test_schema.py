"""Testes do esquema Pydantic da resposta estruturada do RAG."""

import unittest

from pydantic import ValidationError

from src.schema import RAGResponse, SourceEvidence


def build_evidence(**overrides) -> SourceEvidence:
    data = {
        "filepath": "data/semi_structured/tickets.jsonl",
        "chunk_id": "tickets-0001",
        "quotation": "Cliente relatou atraso na entrega.",
        "doc_type": "ticket",
    }
    data.update(overrides)
    return SourceEvidence(**data)


def build_response(**overrides) -> RAGResponse:
    data = {
        "answer": "O pedido foi entregue com 3 dias de atraso.",
        "confidence_level": "Alta",
        "sources_used": [build_evidence()],
        "reasoning": "O ticket indica atraso na entrega.",
    }
    data.update(overrides)
    return RAGResponse(**data)


class SourceEvidenceTests(unittest.TestCase):
    def test_accepts_valid_evidence(self) -> None:
        evidence = build_evidence()

        self.assertEqual(evidence.chunk_id, "tickets-0001")
        self.assertEqual(evidence.doc_type, "ticket")

    def test_rejects_quotation_only_whitespace(self) -> None:
        with self.assertRaises(ValidationError):
            build_evidence(quotation="   ")

    def test_strips_quotation_whitespace(self) -> None:
        evidence = build_evidence(quotation="  texto com espaços  ")

        self.assertEqual(evidence.quotation, "texto com espaços")

    def test_rejects_missing_chunk_id(self) -> None:
        with self.assertRaises(ValidationError):
            SourceEvidence(
                filepath="data/semi_structured/tickets.jsonl",
                quotation="Cliente relatou atraso na entrega.",
            )

    def test_rejects_invalid_doc_type(self) -> None:
        with self.assertRaises(ValidationError):
            build_evidence(doc_type="invalido")


class RAGResponseTests(unittest.TestCase):
    def test_accepts_valid_non_refusal_response(self) -> None:
        response = build_response()

        self.assertFalse(response.is_refusal)
        self.assertEqual(response.confidence_level, "Alta")

    def test_accepts_valid_refusal_response(self) -> None:
        response = build_response(
            confidence_level="Recusado",
            sources_used=[],
            is_refusal=True,
            refusal_reason="OUT_OF_DOMAIN",
        )

        self.assertTrue(response.is_refusal)
        self.assertEqual(response.refusal_reason, "OUT_OF_DOMAIN")

    def test_rejects_non_refusal_without_sources(self) -> None:
        with self.assertRaisesRegex(ValidationError, "ao menos uma evidência"):
            build_response(sources_used=[])

    def test_rejects_refusal_with_sources(self) -> None:
        with self.assertRaisesRegex(ValidationError, "não deve citar sources_used"):
            build_response(
                confidence_level="Recusado",
                is_refusal=True,
                refusal_reason="LGPD_PROTECTION",
            )

    def test_rejects_refusal_without_reason(self) -> None:
        with self.assertRaisesRegex(ValidationError, "refusal_reason"):
            build_response(
                confidence_level="Recusado",
                sources_used=[],
                is_refusal=True,
            )

    def test_rejects_refusal_with_wrong_confidence(self) -> None:
        with self.assertRaisesRegex(ValidationError, "confidence_level='Recusado'"):
            build_response(
                sources_used=[],
                is_refusal=True,
                refusal_reason="LGPD_PROTECTION",
            )

    def test_rejects_recusado_confidence_without_refusal(self) -> None:
        with self.assertRaisesRegex(ValidationError, "só é permitido quando is_refusal=True"):
            build_response(confidence_level="Recusado")

    def test_rejects_refusal_reason_without_refusal(self) -> None:
        with self.assertRaisesRegex(ValidationError, "refusal_reason só é permitido"):
            build_response(refusal_reason="OUT_OF_DOMAIN")


if __name__ == "__main__":
    unittest.main()