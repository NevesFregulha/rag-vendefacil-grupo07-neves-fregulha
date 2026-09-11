"""Mede a qualidade da recuperacao sem usar LLM nenhum.

Roda a busca hibrida para cada pergunta do benchmark e compara as fontes
recuperadas com as `expected_sources` do gabarito. Como nao depende do
LLM-as-judge nem de cota de API, e a medida mais confiavel e reproduzivel do
projeto - e a unica que pode ser usada para validar mudancas na recuperacao sem
gastar tokens.

    python -m eval.retrieval_score
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import config
from src.policy import decide_policy
from src.retrieve import hybrid_search
from src.vectorstore import load_vectorstore

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "questions_and_ground_truth.json"


def _espera_recusa(item: dict[str, Any]) -> bool:
    metadata = item.get("expected_metadata", {})
    return bool(metadata.get("sensitive")) or bool(metadata.get("out_of_domain"))


def avaliar_pergunta(vectorstore: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Recupera contexto para uma pergunta e compara com o gabarito."""
    pergunta = item["question"]
    esperadas = {Path(s).name for s in item.get("expected_sources", [])}

    if _espera_recusa(item) or not esperadas:
        return {"id": item["id"], "aplicavel": False, "recall": None, "achou_alguma": None}

    if decide_policy(pergunta).level == "recusar":
        return {"id": item["id"], "aplicavel": False, "recall": None, "achou_alguma": None}

    # mesma configuracao usada pelo pipeline de geracao, para medir o que ele ve
    documentos = hybrid_search(
        vectorstore, pergunta, k=config.RETRIEVAL_K, complementar_sem_filtro=True
    )
    recuperadas = {str(d.metadata.get("source_file", "")) for d in documentos}
    encontradas = esperadas & recuperadas

    return {
        "id": item["id"],
        "categoria": item["category"],
        "aplicavel": True,
        "esperadas": sorted(esperadas),
        "recuperadas": sorted(recuperadas),
        "recall": len(encontradas) / len(esperadas),
        "achou_alguma": bool(encontradas),
    }


def main() -> None:
    questions = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))["questions"]
    vectorstore = load_vectorstore()

    linhas = [avaliar_pergunta(vectorstore, item) for item in questions]
    aplicaveis = [linha for linha in linhas if linha["aplicavel"]]

    print("{:<6}{:<32}{:>9}  {}".format("id", "categoria", "recall", "achou fonte esperada?"))
    print("-" * 78)
    for linha in aplicaveis:
        print(
            "{:<6}{:<32}{:>9.2f}  {}".format(
                linha["id"],
                linha["categoria"][:30],
                linha["recall"],
                "sim" if linha["achou_alguma"] else "NAO",
            )
        )

    recall_medio = mean(linha["recall"] for linha in aplicaveis)
    acertos = sum(1 for linha in aplicaveis if linha["achou_alguma"])

    print()
    print("Perguntas avaliadas (excluidas as que esperam recusa): {}".format(len(aplicaveis)))
    print("Context Relevance (recall medio das fontes esperadas): {:.3f}".format(recall_medio))
    print("Perguntas que acharam ao menos uma fonte esperada    : {}/{}".format(acertos, len(aplicaveis)))


if __name__ == "__main__":
    main()
