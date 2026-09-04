"""Calcula a RAG Triad (Context Relevance, Answer Relevance, Groundedness) da Etapa 4.

Le eval/results.json (gerado por eval/run_benchmark.py) e calcula:

- Context Relevance: deterministica, sem LLM - compara as fontes recuperadas pela
  busca hibrida contra as expected_sources do gabarito. Perguntas que o gabarito
  espera que sejam recusadas (LGPD ou fora de escopo) sao marcadas como nao
  aplicavel, ja que a politica recusa antes de consultar o indice - a ausencia de
  contexto ali e o comportamento correto, nao uma falha de recuperacao.
- Answer Relevance e Groundedness: LLM-as-judge, comparando a resposta gerada com
  a ground_truth_answer e os key_points_for_evaluation do gabarito.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

import config

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
TRIAD_PATH = Path(__file__).resolve().parent / "triad_scores.json"

JudgeLevel = Literal["Alta", "Media", "Baixa"]

_LEVEL_TO_SCORE = {"Alta": 1.0, "Media": 0.5, "Baixa": 0.0}


class JudgeVerdict(BaseModel):
    """Veredito estruturado do LLM-as-judge para uma pergunta do benchmark."""

    answer_relevance: JudgeLevel = Field(
        description="O quanto a resposta atende ao que foi perguntado, comparada ao gabarito."
    )
    groundedness: JudgeLevel = Field(
        description="O quanto a resposta se apoia apenas no contexto/evidencia citada, sem alucinar."
    )
    justification: str = Field(
        min_length=1,
        description="Breve justificativa objetiva para os dois niveis atribuidos.",
    )


JUDGE_SYSTEM_PROMPT = (
    "Voce e um avaliador (LLM-as-judge) do assistente RAG da VendeFacil. Compare a "
    "resposta gerada pelo assistente com a resposta de referencia (ground truth) e "
    "os pontos-chave esperados. Atribua Alta, Media ou Baixa para answer_relevance "
    "(a resposta atende ao que foi perguntado?) e para groundedness (a resposta se "
    "apoia apenas na evidencia citada, sem inventar fatos?). Uma recusa correta, "
    "quando o gabarito tambem espera recusa, deve receber Alta em ambos os campos."
)


def _is_expected_refusal(item: dict[str, Any]) -> bool:
    expected_metadata = item.get("expected_metadata", {})
    return bool(expected_metadata.get("sensitive")) or bool(
        expected_metadata.get("out_of_domain")
    )


def context_relevance(item: dict[str, Any]) -> float | None:
    """Fracao das expected_sources do gabarito presentes nas fontes recuperadas.

    Retorna None quando o gabarito espera uma recusa (a politica nao consulta o
    indice nesses casos, entao nao ha contexto recuperado para avaliar).
    """
    if _is_expected_refusal(item):
        return None

    expected = {Path(source).name for source in item.get("expected_sources", [])}
    if not expected:
        return None

    retrieved = {Path(source).name for source in item.get("retrieved_sources", [])}
    return len(expected & retrieved) / len(expected)


def build_judge_messages(item: dict[str, Any]) -> list[tuple[str, str]]:
    user_prompt = (
        f"Pergunta:\n{item['question']}\n\n"
        f"Resposta de referencia (ground truth):\n{item.get('ground_truth_answer', '')}\n\n"
        f"Pontos-chave esperados:\n"
        + "\n".join(f"- {point}" for point in item.get("key_points_for_evaluation", []))
        + "\n\nResposta gerada pelo assistente:\n"
        f"{item.get('answer') or '(sem resposta - erro no pipeline)'}\n\n"
        f"is_refusal: {item.get('is_refusal')}\n"
        f"Fontes citadas pelo assistente: {item.get('sources_used', [])}"
    )
    return [("system", JUDGE_SYSTEM_PROMPT), ("human", user_prompt)]


def judge_answer(llm: Any, item: dict[str, Any]) -> JudgeVerdict:
    """Julga Answer Relevance e Groundedness de uma pergunta ja executada no benchmark."""
    structured_llm = llm.with_structured_output(JudgeVerdict)
    messages = build_judge_messages(item)
    verdict = structured_llm.invoke(messages)
    if isinstance(verdict, JudgeVerdict):
        return verdict
    return JudgeVerdict.model_validate(verdict)


def score_question(llm: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Calcula as tres metricas da RAG Triad para uma unica pergunta."""
    scored: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "context_relevance": context_relevance(item),
    }

    if item.get("error"):
        scored["answer_relevance"] = None
        scored["groundedness"] = None
        scored["judge_justification"] = f"Pulado: erro no pipeline ({item['error']})."
        return scored

    verdict = judge_answer(llm, item)
    scored["answer_relevance"] = _LEVEL_TO_SCORE[verdict.answer_relevance]
    scored["groundedness"] = _LEVEL_TO_SCORE[verdict.groundedness]
    scored["judge_justification"] = verdict.justification
    return scored


def _average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def summarize_triad(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    """Media geral das tres metricas, ignorando perguntas onde a metrica nao se aplica."""
    return {
        "context_relevance": _average([s["context_relevance"] for s in scores]),
        "answer_relevance": _average([s["answer_relevance"] for s in scores]),
        "groundedness": _average([s["groundedness"] for s in scores]),
    }


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    llm = config.get_llm()

    scores = [score_question(llm, item) for item in results]
    summary = summarize_triad(scores)

    TRIAD_PATH.write_text(
        json.dumps(
            {"scores": scores, "summary": summary}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("RAG Triad - medias gerais:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.2f}" if value is not None else f"  {metric}: N/A")
    print(f"\nDetalhes por pergunta salvos em: {TRIAD_PATH}")


if __name__ == "__main__":
    main()
