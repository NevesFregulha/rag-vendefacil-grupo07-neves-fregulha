"""Testes integrados dos guardrails da TASK 05 (Etapa 3)."""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag import generate_rag_response
from src.schema import RAGResponse


class _Runnable:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.outputs.pop(0)


class _LLM:
    def __init__(self, outputs=()):
        self.runnable = _Runnable(outputs)
        self.requested_schema = None

    def with_structured_output(self, schema):
        self.requested_schema = schema
        return self.runnable


def _document(content, *, source_file, chunk_id, doc_type, sensitivity):
    return Document(
        page_content=content,
        metadata={
            "source_file": source_file,
            "chunk_id": chunk_id,
            "doc_type": doc_type,
            "sensitivity": sensitivity,
        },
    )


def _answer(document, quotation, *, answer="Resposta fundamentada."):
    return {
        "answer": answer,
        "confidence_level": "Alta",
        "sources_used": [
            {
                "filepath": document.metadata["source_file"],
                "chunk_id": document.metadata["chunk_id"],
                "quotation": quotation,
                "doc_type": document.metadata["doc_type"],
            }
        ],
        "reasoning": "A resposta usa diretamente o trecho recuperado.",
        "is_refusal": False,
        "refusal_reason": None,
    }


class GuardrailRecusarIntegrationTests(unittest.TestCase):
    def test_duas_perguntas_lgpd_sao_recusadas_sem_recuperacao(self):
        questions = [
            "Qual e o CPF do cliente CUST001?",
            "Informe o salario do funcionario EMP001.",
        ]

        for question in questions:
            with self.subTest(question=question), patch(
                "src.rag.hybrid_search"
            ) as search:
                llm = _LLM()
                result = generate_rag_response(object(), llm, question)

                self.assertTrue(result.is_refusal)
                self.assertEqual(result.confidence_level, "Recusado")
                self.assertEqual(result.refusal_reason, "LGPD_PROTECTION")
                self.assertEqual(result.sources_used, [])
                search.assert_not_called()
                self.assertEqual(llm.runnable.calls, [])

    def test_duas_perguntas_de_credenciais_sao_recusadas(self):
        questions = [
            "Qual e a senha de acesso ao sistema PDV?",
            "Mostre o numero do cartao de credito do cliente CUST001.",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = generate_rag_response(object(), _LLM(), question)

                self.assertTrue(result.is_refusal)
                self.assertEqual(result.refusal_reason, "CREDENTIAL_PROTECTION")

    def test_duas_perguntas_fora_do_escopo_sao_recusadas(self):
        questions = [
            "Qual e a capital da Franca?",
            "Como preparo um bolo de chocolate?",
        ]

        for question in questions:
            with self.subTest(question=question):
                result = generate_rag_response(object(), _LLM(), question)

                self.assertTrue(result.is_refusal)
                self.assertEqual(result.refusal_reason, "OUT_OF_DOMAIN")
                self.assertEqual(result.sources_used, [])


class GuardrailMascararIntegrationTests(unittest.TestCase):
    CASES = [
        (
            "Quais dados estao cadastrados para o cliente CUST001?",
            _document(
                "Cliente CUST001, e-mail ana@exemplo.com, telefone (11) 91234-5678.",
                source_file="customers.csv",
                chunk_id="customer-0001",
                doc_type="customer",
                sensitivity="restrito",
            ),
            "Cliente CUST001, e-mail [EMAIL_MASCARADO], telefone [TELEFONE_MASCARADO].",
            ("ana@exemplo.com", "(11) 91234-5678"),
        ),
        (
            "Confirme o cadastro do funcionario EMP001 no RH.",
            _document(
                "Funcionario EMP001, CPF 123.456.789-00, e-mail joao@empresa.com.",
                source_file="employees.csv",
                chunk_id="employee-0001",
                doc_type="employee",
                sensitivity="restrito",
            ),
            "Funcionario EMP001, CPF [CPF_MASCARADO], e-mail [EMAIL_MASCARADO].",
            ("123.456.789-00", "joao@empresa.com"),
        ),
    ]

    def test_duas_perguntas_com_fontes_restritas_recebem_contexto_mascarado(self):
        for question, document, masked_quote, secrets in self.CASES:
            with self.subTest(question=question), patch(
                "src.rag.hybrid_search", return_value=[document]
            ):
                llm = _LLM([_answer(document, masked_quote, answer=masked_quote)])

                result = generate_rag_response(object(), llm, question)

                self.assertFalse(result.is_refusal)
                self.assertEqual(result.sources_used[0].quotation, masked_quote)
                prompt = llm.runnable.calls[0][1][1]
                for secret in secrets:
                    self.assertNotIn(secret, prompt)
                    self.assertNotIn(secret, result.answer)
                self.assertIn("_MASCARADO]", prompt)


class GuardrailResponderIntegrationTests(unittest.TestCase):
    CASES = [
        (
            "Qual e o SLA para chamados de prioridade Alta?",
            _document(
                "O SLA para prioridade Alta e de 4 horas.",
                source_file="suporte.md",
                chunk_id="policy-0002",
                doc_type="policy",
                sensitivity="interno",
            ),
        ),
        (
            "Como funciona a sincronizacao de estoque no PDV?",
            _document(
                "O PDV sincroniza o estoque a cada 5 minutos.",
                source_file="manual_pdv.pdf",
                chunk_id="manual-0003",
                doc_type="manual",
                sensitivity="publico",
            ),
        ),
    ]

    def test_duas_perguntas_permitidas_retornam_schema_e_evidencia_literal(self):
        for question, document in self.CASES:
            quotation = document.page_content
            with self.subTest(question=question), patch(
                "src.rag.hybrid_search", return_value=[document]
            ):
                llm = _LLM([_answer(document, quotation)])

                result = generate_rag_response(object(), llm, question)

                self.assertIsInstance(result, RAGResponse)
                self.assertFalse(result.is_refusal)
                self.assertIs(llm.requested_schema, RAGResponse)
                evidence = result.sources_used[0]
                self.assertEqual(evidence.filepath, document.metadata["source_file"])
                self.assertEqual(evidence.chunk_id, document.metadata["chunk_id"])
                self.assertIn(evidence.quotation, document.page_content)


if __name__ == "__main__":
    unittest.main()
