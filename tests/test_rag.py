"""Testes da geracao estruturada, evidencias e retry das TASKS 03/04."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

from src.rag import (
    EvidenceValidationError,
    NoRelevantDocumentsError,
    StructuredGenerationError,
    generate_rag_response,
)
from src.schema import RAGResponse


class _StructuredRunnable:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class _FakeLLM:
    def __init__(self, outputs):
        self.runnable = _StructuredRunnable(outputs)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.runnable


def _document(*, sensitivity="interno"):
    return Document(
        page_content="O SLA para chamados de prioridade Alta e de 4 horas.",
        metadata={
            "source_file": "politica_suporte.md",
            "chunk_id": "policy-0001",
            "doc_type": "policy",
            "sensitivity": sensitivity,
        },
    )


def _valid_output(**overrides):
    output = {
        "answer": "O SLA e de 4 horas.",
        "confidence_level": "Alta",
        "sources_used": [
            {
                "filepath": "politica_suporte.md",
                "chunk_id": "policy-0001",
                "quotation": "O SLA para chamados de prioridade Alta e de 4 horas.",
                "doc_type": "policy",
            }
        ],
        "reasoning": "A politica recuperada informa diretamente o prazo.",
        "is_refusal": False,
        "refusal_reason": None,
    }
    output.update(overrides)
    return output


class StructuredGenerationTests(unittest.TestCase):
    @patch("src.rag.hybrid_search")
    def test_integrates_hybrid_search_and_returns_rag_response(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM([_valid_output()])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA dos chamados de prioridade Alta?", k=3
        )

        self.assertIsInstance(result, RAGResponse)
        self.assertEqual(result.sources_used[0].chunk_id, "policy-0001")
        self.assertIs(llm.schema, RAGResponse)
        search.assert_called_once_with(
            unittest.mock.ANY,
            "Qual e o SLA dos chamados de prioridade Alta?",
            k=3,
            filters=None,
        )

    @patch("src.rag.hybrid_search")
    def test_retries_after_schema_validation_failure(self, search):
        search.return_value = [_document()]
        invalid = _valid_output(sources_used=[])
        llm = _FakeLLM([invalid, _valid_output()])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=2
        )

        self.assertEqual(result.confidence_level, "Alta")
        self.assertEqual(len(llm.runnable.calls), 2)
        second_prompt = llm.runnable.calls[1][1][1]
        self.assertIn("tentativa anterior foi invalida", second_prompt)

    @patch("src.rag.hybrid_search")
    def test_retries_after_provider_output_parser_failure(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM(
            [OutputParserException("JSON invalido"), _valid_output()]
        )

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=2
        )

        self.assertEqual(result.answer, "O SLA e de 4 horas.")
        self.assertEqual(len(llm.runnable.calls), 2)

    @patch("src.rag.hybrid_search")
    def test_retries_when_quotation_is_not_literal(self, search):
        search.return_value = [_document()]
        invented = _valid_output()
        invented["sources_used"][0]["quotation"] = "Trecho inventado"
        llm = _FakeLLM([invented, _valid_output()])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=2
        )

        self.assertEqual(result.sources_used[0].filepath, "politica_suporte.md")
        self.assertEqual(len(llm.runnable.calls), 2)

    @patch("src.rag.hybrid_search")
    def test_retries_when_filepath_chunk_or_doc_type_do_not_match(self, search):
        search.return_value = [_document()]

        invalid_evidences = [
            {"filepath": "inventado.md"},
            {"chunk_id": "chunk-inventado"},
            {"doc_type": "ticket"},
        ]
        for changed_fields in invalid_evidences:
            with self.subTest(changed_fields=changed_fields):
                invalid = _valid_output()
                invalid["sources_used"][0].update(changed_fields)
                llm = _FakeLLM([invalid, _valid_output()])

                result = generate_rag_response(
                    object(), llm, "Qual e o SLA de atendimento?", max_attempts=2
                )

                self.assertEqual(result.sources_used[0].chunk_id, "policy-0001")
                self.assertEqual(len(llm.runnable.calls), 2)

    @patch("src.rag.hybrid_search", return_value=[])
    def test_fails_without_documents_instead_of_generating_without_source(self, _search):
        with self.assertRaises(NoRelevantDocumentsError):
            generate_rag_response(
                object(), _FakeLLM([]), "Qual e o SLA de atendimento?"
            )

    @patch("src.rag.hybrid_search")
    def test_reports_error_after_exhausting_retries(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM([{"answer": "incompleta"}, {"answer": "incompleta"}])

        with self.assertRaises(StructuredGenerationError):
            generate_rag_response(
                object(), llm, "Qual e o SLA de atendimento?", max_attempts=2
            )

    @patch("src.rag.hybrid_search")
    def test_preserves_evidence_error_as_cause_after_last_attempt(self, search):
        search.return_value = [_document()]
        invented = _valid_output()
        invented["sources_used"][0]["quotation"] = "Trecho inventado"
        llm = _FakeLLM([invented])

        with self.assertRaises(StructuredGenerationError) as raised:
            generate_rag_response(
                object(), llm, "Qual e o SLA de atendimento?", max_attempts=1
            )

        self.assertIsInstance(raised.exception.__cause__, EvidenceValidationError)

    @patch("src.rag.hybrid_search")
    def test_refusal_does_not_call_search_or_llm(self, search):
        llm = _FakeLLM([])

        result = generate_rag_response(
            object(), llm, "Qual e a capital da Franca?"
        )

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.refusal_reason, "OUT_OF_DOMAIN")
        search.assert_not_called()
        self.assertEqual(llm.runnable.calls, [])


if __name__ == "__main__":
    unittest.main()
