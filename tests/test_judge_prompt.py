"""Testes do calculo da RAG Triad (eval/judge_prompt.py)."""

import unittest

from eval.judge_prompt import (
    JudgeVerdict,
    context_relevance,
    score_question,
    summarize_triad,
)


class _Runnable:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, messages):
        return self.outputs.pop(0)


class _LLM:
    def __init__(self, outputs=()):
        self.runnable = _Runnable(outputs)

    def with_structured_output(self, schema):
        return self.runnable


class ContextRelevanceTests(unittest.TestCase):
    def test_pergunta_esperando_recusa_nao_e_avaliada(self):
        item = {
            "expected_metadata": {"sensitive": True},
            "expected_sources": ["data/structured/employees.csv"],
            "retrieved_sources": [],
        }
        self.assertIsNone(context_relevance(item))

    def test_pergunta_fora_de_escopo_nao_e_avaliada(self):
        item = {
            "expected_metadata": {"out_of_domain": True},
            "expected_sources": [],
            "retrieved_sources": [],
        }
        self.assertIsNone(context_relevance(item))

    def test_todas_as_fontes_esperadas_recuperadas_vale_1(self):
        item = {
            "expected_metadata": {},
            "expected_sources": ["data/structured/products.json"],
            "retrieved_sources": ["products.json", "stores.json"],
        }
        self.assertEqual(context_relevance(item), 1.0)

    def test_metade_das_fontes_esperadas_recuperadas_vale_0_5(self):
        item = {
            "expected_metadata": {},
            "expected_sources": [
                "data/unstructured/emails/customer_001.txt",
                "data/semi_structured/tickets.jsonl",
            ],
            "retrieved_sources": ["tickets.jsonl"],
        }
        self.assertEqual(context_relevance(item), 0.5)

    def test_nenhuma_fonte_esperada_recuperada_vale_0(self):
        item = {
            "expected_metadata": {},
            "expected_sources": ["data/structured/products.json"],
            "retrieved_sources": ["stores.json"],
        }
        self.assertEqual(context_relevance(item), 0.0)


class ScoreQuestionTests(unittest.TestCase):
    def test_pergunta_com_erro_pula_o_julgamento_por_llm(self):
        item = {
            "id": "Q99",
            "category": "Teste",
            "expected_metadata": {},
            "expected_sources": [],
            "retrieved_sources": [],
            "error": "NoRelevantDocumentsError: sem contexto",
        }

        scored = score_question(_LLM(), item)

        self.assertIsNone(scored["answer_relevance"])
        self.assertIsNone(scored["groundedness"])
        self.assertIn("Pulado", scored["judge_justification"])

    def test_pergunta_valida_e_julgada_pelo_llm(self):
        item = {
            "id": "Q01",
            "category": "Fácil (RAG Básico)",
            "question": "Quais os produtos da VendeFacil?",
            "ground_truth_answer": "5 produtos.",
            "key_points_for_evaluation": ["Mencionar os 5 produtos"],
            "expected_metadata": {},
            "expected_sources": ["data/structured/products.json"],
            "retrieved_sources": ["products.json"],
            "answer": "A VendeFacil tem 5 produtos.",
            "is_refusal": False,
            "sources_used": [],
            "error": None,
        }
        verdict = JudgeVerdict(
            answer_relevance="Alta",
            groundedness="Alta",
            justification="Resposta cobre os pontos esperados.",
        )

        scored = score_question(_LLM([verdict]), item)

        self.assertEqual(scored["context_relevance"], 1.0)
        self.assertEqual(scored["answer_relevance"], 1.0)
        self.assertEqual(scored["groundedness"], 1.0)


class SummarizeTriadTests(unittest.TestCase):
    def test_media_ignora_metricas_nao_aplicaveis(self):
        scores = [
            {"context_relevance": 1.0, "answer_relevance": 1.0, "groundedness": 1.0},
            {"context_relevance": None, "answer_relevance": 1.0, "groundedness": 0.5},
            {"context_relevance": 0.0, "answer_relevance": None, "groundedness": None},
        ]

        summary = summarize_triad(scores)

        self.assertAlmostEqual(summary["context_relevance"], 0.5)
        self.assertAlmostEqual(summary["answer_relevance"], 1.0)
        self.assertAlmostEqual(summary["groundedness"], 0.75)


if __name__ == "__main__":
    unittest.main()
