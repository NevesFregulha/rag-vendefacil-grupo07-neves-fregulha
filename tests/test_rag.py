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
            # o pipeline aceita perder pureza de filtro para alcancar a fonte
            complementar_sem_filtro=True,
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


class CitacaoToleraEspacamentoTests(unittest.TestCase):
    """Regressao: Q03, Q09 e Q17 zeravam por diferenca de espaco em branco.

    O modelo citava o chunk certo e o texto certo, mas ao copiar um trecho que
    atravessa uma quebra de linha ele normaliza o espacamento - e a comparacao
    exata reprovava a resposta correta.
    """

    @patch("src.rag.hybrid_search")
    def test_aceita_citacao_com_quebra_de_linha_normalizada(self, search):
        chunk = Document(
            page_content="## Politica\nO SLA para chamados de prioridade Alta\ne de 4 horas.",
            metadata={
                "source_file": "politica_suporte.md",
                "chunk_id": "policy-0001",
                "doc_type": "policy",
                "sensitivity": "interno",
            },
        )
        search.return_value = [chunk]
        # o modelo devolve o mesmo trecho, com a quebra virando espaco
        citacao = "O SLA para chamados de prioridade Alta e de 4 horas."
        llm = _FakeLLM([_valid_output(sources_used=[{
            "filepath": "politica_suporte.md",
            "chunk_id": "policy-0001",
            "quotation": citacao,
            "doc_type": "policy",
        }])])

        result = generate_rag_response(object(), llm, "Qual e o SLA de atendimento?")

        self.assertFalse(result.is_refusal)
        self.assertEqual(len(llm.runnable.calls), 1, "nao deve precisar de retry")

    @patch("src.rag.hybrid_search")
    def test_continua_rejeitando_citacao_inventada(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM(
            [
                _valid_output(sources_used=[{
                    "filepath": "politica_suporte.md",
                    "chunk_id": "policy-0001",
                    "quotation": "O SLA e de 24 horas conforme contrato premium.",
                    "doc_type": "policy",
                }])
            ]
        )

        with self.assertRaises(StructuredGenerationError) as raised:
            generate_rag_response(
                object(), llm, "Qual e o SLA de atendimento?", max_attempts=1
            )

        self.assertIsInstance(raised.exception.__cause__, EvidenceValidationError)


class RecusaSemEvidenciaComContextoTests(unittest.TestCase):
    """Recusa SEM_EVIDENCIA com contexto recuperado deve ser aceita.

    Ja tentamos rejeita-la no validador, supondo que Q04, Q12 e Q21 recusavam
    com a resposta no contexto. A medicao refutou: chegava o arquivo certo, mas
    so o titulo ou a secao errada. Como a busca nunca devolve contexto vazio a
    este ponto, a rejeicao transformava toda recusa honesta em erro.
    """

    RECUSA = {
        "answer": "Nao encontrei essa informacao no contexto recuperado.",
        "confidence_level": "Recusado",
        "sources_used": [],
        "reasoning": "O contexto nao traz a resposta.",
        "is_refusal": True,
        "refusal_reason": "SEM_EVIDENCIA",
    }

    @patch("src.rag.hybrid_search")
    def test_aceita_recusa_honesta_mesmo_com_contexto(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM([self.RECUSA])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=3
        )

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.refusal_reason, "SEM_EVIDENCIA")
        self.assertEqual(len(llm.runnable.calls), 1, "nao deve forcar nova tentativa")

    @patch("src.rag.hybrid_search")
    def test_recusa_de_guardrail_nao_e_afetada(self, search):
        """Recusa por LGPD acontece antes da busca e nao passa por aqui."""
        llm = _FakeLLM([])

        result = generate_rag_response(object(), llm, "Qual e a capital da Franca?")

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.refusal_reason, "OUT_OF_DOMAIN")
        search.assert_not_called()


class _ProviderError(Exception):
    """Imita a rejeicao server-side de schema feita por provedores como a Groq.

    O ponto do teste esta na hierarquia: erros assim NAO herdam de ValueError,
    entao escapavam do `except` do laco de retry.
    """

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


class RejeicaoDeSchemaPeloProvedorTests(unittest.TestCase):
    """Regressao: o retry nao rodava quando o provedor validava no servidor.

    Detectado no benchmark da Etapa 4: 5 das 10 falhas eram HTTP 400 de
    validacao de tool call. Como `groq.BadRequestError` nao herda de
    `ValueError`, a excecao escapava do laco e as 3 tentativas configuradas
    nunca aconteciam.
    """

    ERRO_400 = (
        "Error code: 400 - {'error': {'message': 'Tool call validation failed: "
        "parameters for tool RAGResponse did not match schema: errors: "
        "[`/answer`: length must be >= 1, but got 0]'}}"
    )

    @patch("src.rag.hybrid_search")
    def test_retry_roda_apos_rejeicao_server_side(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM([_ProviderError(self.ERRO_400), _valid_output()])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=3
        )

        self.assertFalse(result.is_refusal)
        self.assertEqual(len(llm.runnable.calls), 2, "a segunda tentativa deve acontecer")
        segundo_prompt = llm.runnable.calls[1][1][1]
        self.assertIn("tentativa anterior foi invalida", segundo_prompt)

    @patch("src.rag.hybrid_search")
    def test_retry_roda_apos_variante_parsing_failed(self, search):
        """Regressao: esta redacao do mesmo erro nao acionava o retry (Q11)."""
        search.return_value = [_document()]
        erro = (
            "Error code: 400 - {'error': {'message': \"Parsing failed. The model "
            'generated output that could not be parsed."}}'
        )
        llm = _FakeLLM([_ProviderError(erro), _valid_output()])

        result = generate_rag_response(
            object(), llm, "Qual e o SLA de atendimento?", max_attempts=3
        )

        self.assertFalse(result.is_refusal)
        self.assertEqual(len(llm.runnable.calls), 2)

    @patch("src.rag.hybrid_search")
    def test_erro_nao_relacionado_a_schema_continua_propagando(self, search):
        search.return_value = [_document()]
        llm = _FakeLLM([_ProviderError("Error code: 401 - invalid api key", status_code=401)])

        with self.assertRaises(_ProviderError):
            generate_rag_response(
                object(), llm, "Qual e o SLA de atendimento?", max_attempts=3
            )

        self.assertEqual(
            len(llm.runnable.calls), 1, "nao faz sentido repetir erro de autenticacao"
        )


if __name__ == "__main__":
    unittest.main()
