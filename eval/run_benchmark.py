"""Executa o benchmark da Etapa 4 contra o pipeline RAG e salva os resultados brutos.

Le todas as perguntas de benchmark/questions_and_ground_truth.json (sem assumir uma
quantidade fixa), roda cada uma pelo pipeline de src/rag.py e grava eval/results.json
com a resposta, a citacao de evidencia e as fontes efetivamente recuperadas para cada
pergunta. O calculo da RAG Triad (Context Relevance, Answer Relevance, Groundedness)
fica em eval/judge_prompt.py, que le este results.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from src.policy import decide_policy
from src.rag import generate_rag_response
from src.retrieve import hybrid_search
from src.vectorstore import load_vectorstore

# O tier gratuito da Groq limita tokens por minuto, e o benchmark envia 24
# prompts com 5 chunks de contexto cada. Sem espera, a rajada estoura o limite
# no meio da execucao e perguntas boas viram erro - contaminando a medicao com
# um problema de cota, nao do pipeline.
RATE_LIMIT_MAX_ATTEMPTS = 4
RATE_LIMIT_BACKOFF_SECONDS = 25.0
PAUSE_BETWEEN_QUESTIONS_SECONDS = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "questions_and_ground_truth.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_benchmark_questions(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Carrega as perguntas do benchmark, sem exigir uma quantidade fixa."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def _retrieved_sources(vectorstore: Any, question: str) -> list[str]:
    """Fontes realmente recuperadas pela busca hibrida, usadas na Context Relevance.

    So consulta o indice quando a politica permite (perguntas recusadas nao chegam
    a buscar contexto, entao nao ha o que comparar).
    """
    if decide_policy(question).level == "recusar":
        return []
    documents = hybrid_search(
        vectorstore, question, k=config.RETRIEVAL_K, complementar_sem_filtro=True
    )
    return sorted({str(document.metadata.get("source_file", "")) for document in documents})


def _is_rate_limit(error: Exception) -> bool:
    """Distingue estouro de cota (transitorio) de defeito do pipeline."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 429:
        return True
    return "rate limit" in str(error).lower()


def run_question(vectorstore: Any, llm: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Roda uma pergunta do benchmark pelo pipeline RAG completo."""
    question = item["question"]
    entry: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected_sources": item.get("expected_sources", []),
        "expected_metadata": item.get("expected_metadata", {}),
        "ground_truth_answer": item.get("ground_truth_answer", ""),
        "key_points_for_evaluation": item.get("key_points_for_evaluation", []),
        "retrieved_sources": _retrieved_sources(vectorstore, question),
    }

    for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
        try:
            response = generate_rag_response(
                vectorstore,
                llm,
                question,
                k=config.RETRIEVAL_K,
                max_attempts=config.MAX_GENERATION_ATTEMPTS,
            )
        except Exception as error:  # noqa: BLE001 - benchmark nao para por 1 pergunta ruim
            if _is_rate_limit(error) and attempt < RATE_LIMIT_MAX_ATTEMPTS:
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS * attempt)
                continue
            entry.update(
                {
                    "answer": None,
                    "confidence_level": None,
                    "is_refusal": None,
                    "refusal_reason": None,
                    "reasoning": None,
                    "sources_used": [],
                    "error": str(error),
                }
            )
            return entry

        entry.update(
            {
                "answer": response.answer,
                "confidence_level": response.confidence_level,
                "is_refusal": response.is_refusal,
                "refusal_reason": response.refusal_reason,
                "reasoning": response.reasoning,
                "sources_used": [
                    source.model_dump() for source in response.sources_used
                ],
                "error": None,
            }
        )
        return entry

    raise AssertionError("inalcancavel")


def summarize_by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Agrega total, recusas e erros por categoria, para a tabela resumo."""
    summary: dict[str, dict[str, int]] = {}
    for result in results:
        stats = summary.setdefault(
            result["category"], {"total": 0, "recusas": 0, "erros": 0}
        )
        stats["total"] += 1
        if result.get("is_refusal"):
            stats["recusas"] += 1
        if result.get("error"):
            stats["erros"] += 1
    return summary


def _print_summary_table(summary: dict[str, dict[str, int]]) -> None:
    header = f"{'Categoria':<35} {'Total':>6} {'Recusas':>8} {'Erros':>6}"
    print(header)
    print("-" * len(header))
    for category, stats in summary.items():
        print(
            f"{category:<35} {stats['total']:>6} {stats['recusas']:>8} {stats['erros']:>6}"
        )


def _save_results(results: list[dict[str, Any]], *, archive_path: Path | None = None) -> None:
    """Grava o resultado corrente e, opcionalmente, uma copia datada.

    A copia existe porque uma reexecucao sobrescreve `results.json`: uma medicao
    boa ja foi perdida assim, substituida por uma execucao contaminada por
    estouro de cota. Manter o historico torna cada medicao reproduzivel e
    comparavel.
    """
    payload = json.dumps(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # Sem registrar provedor e modelo, duas execucoes viram numeros sem
            # contexto: ja aconteceu de comparar medicoes feitas em modelos
            # diferentes e atribuir a diferenca a uma correcao de codigo.
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL,
            "retrieval_k": config.RETRIEVAL_K,
            "total_questions": len(results),
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
    )
    RESULTS_PATH.write_text(payload, encoding="utf-8")
    if archive_path is not None:
        archive_path.write_text(payload, encoding="utf-8")


def main() -> None:
    questions = load_benchmark_questions()
    vectorstore = load_vectorstore()
    llm = config.get_llm()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = RUNS_DIR / f"results-{carimbo}.json"

    results: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {item['id']} - {item['question'][:60]}...")
        results.append(run_question(vectorstore, llm, item))
        # salva a cada pergunta, para nao perder progresso, e guarda copia datada
        _save_results(results, archive_path=archive_path)
        time.sleep(PAUSE_BETWEEN_QUESTIONS_SECONDS)  # respeita o limite de tokens/minuto

    print(f"\nTotal de perguntas executadas: {len(results)}\n")
    _print_summary_table(summarize_by_category(results))
    print(f"\nResultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
