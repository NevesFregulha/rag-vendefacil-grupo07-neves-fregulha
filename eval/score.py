"""Aplica a rubrica de pontuacao do guia aos resultados do benchmark.

Rubrica (1,0 ponto por questao):
  0,5 - resposta correta, ou recusa correta nas questoes de recusa;
  0,3 - a citacao aponta o arquivo certo;
  0,2 - `confidence_level` e `is_refusal` coerentes com a resposta dada.

Uma questao NAO AVALIADA nao e uma questao zerada. Quando o LLM-as-judge falha
(por exemplo, por estouro de cota), `answer_relevance` fica vazio - e tratar
esse vazio como zero ja produziu uma nota falsamente baixa neste projeto.
Aqui essas questoes sao contabilizadas a parte, e a pontuacao e reportada como
intervalo: o piso conta as nao avaliadas como zero, o teto conta como acerto
total. Se o intervalo for largo, a medicao nao esta pronta para ser reportada.

    python -m eval.score
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results.json"
TRIAD_PATH = EVAL_DIR / "triad_scores.json"

PESO_RESPOSTA = 0.5
PESO_CITACAO = 0.3
PESO_COERENCIA = 0.2


def espera_recusa(item: dict[str, Any]) -> bool:
    metadata = item.get("expected_metadata", {})
    return bool(metadata.get("sensitive")) or bool(metadata.get("out_of_domain"))


def pontuar_questao(item: dict[str, Any], veredito: dict[str, Any]) -> dict[str, Any]:
    """Pontua uma questao. `avaliada=False` significa sem nota do juiz."""
    base = {"id": item["id"], "categoria": item["category"], "avaliada": True}

    if item.get("error"):
        # Erro de pipeline e falha objetiva: nao produziu resposta alguma.
        return {**base, "resposta": 0.0, "citacao": 0.0, "coerencia": 0.0, "total": 0.0}

    if espera_recusa(item):
        resposta = PESO_RESPOSTA if item.get("is_refusal") else 0.0
        citacao = PESO_CITACAO if not item.get("sources_used") else 0.0
        coerente = item.get("is_refusal") and item.get("confidence_level") == "Recusado"
        coerencia = PESO_COERENCIA if coerente else 0.0
        return {**base, "resposta": resposta, "citacao": citacao, "coerencia": coerencia,
                "total": resposta + citacao + coerencia}

    relevancia = veredito.get("answer_relevance")
    if relevancia is None:
        # O juiz nao avaliou. Citacao e coerencia sao objetivas e continuam
        # valendo; so a correcao do conteudo fica em aberto.
        citacao = _pontuar_citacao(item)
        coerencia = _pontuar_coerencia(item)
        return {**base, "avaliada": False, "resposta": None, "citacao": citacao,
                "coerencia": coerencia, "total": None}

    return {
        **base,
        "resposta": PESO_RESPOSTA * relevancia,
        "citacao": _pontuar_citacao(item),
        "coerencia": _pontuar_coerencia(item),
        "total": PESO_RESPOSTA * relevancia + _pontuar_citacao(item) + _pontuar_coerencia(item),
    }


def _pontuar_citacao(item: dict[str, Any]) -> float:
    esperadas = {Path(s).name for s in item.get("expected_sources", [])}
    citadas = {Path(s["filepath"]).name for s in item.get("sources_used", [])}
    return PESO_CITACAO if esperadas & citadas else 0.0


def _pontuar_coerencia(item: dict[str, Any]) -> float:
    coerente = (
        not item.get("is_refusal")
        and bool(item.get("sources_used"))
        and item.get("confidence_level") != "Recusado"
    )
    return PESO_COERENCIA if coerente else 0.0


def pontuar(results: list[dict[str, Any]], scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vereditos = {s["id"]: s for s in scores}
    return [pontuar_questao(item, vereditos.get(item["id"], {})) for item in results]


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    scores = json.loads(TRIAD_PATH.read_text(encoding="utf-8"))["scores"]
    linhas = pontuar(results, scores)

    total_questoes = len(linhas)
    avaliadas = [linha for linha in linhas if linha["avaliada"]]
    pendentes = [linha for linha in linhas if not linha["avaliada"]]

    piso = sum(linha["total"] for linha in avaliadas) + sum(
        linha["citacao"] + linha["coerencia"] for linha in pendentes
    )
    teto = piso + sum(PESO_RESPOSTA for _ in pendentes)

    print("{:<6}{:<32}{:>9}{:>9}{:>10}{:>9}".format(
        "id", "categoria", "resposta", "citacao", "coerencia", "total"))
    print("-" * 76)
    for linha in linhas:
        marca = "  (nao avaliada)" if not linha["avaliada"] else ""
        total = "-" if linha["total"] is None else "{:.2f}".format(linha["total"])
        resposta = "-" if linha["resposta"] is None else "{:.2f}".format(linha["resposta"])
        print("{:<6}{:<32}{:>9}{:>9.2f}{:>10.2f}{:>9}{}".format(
            linha["id"], linha["categoria"][:30], resposta,
            linha["citacao"], linha["coerencia"], total, marca))

    print()
    if pendentes:
        print("ATENCAO: {} de {} questoes nao foram avaliadas pelo juiz: {}".format(
            len(pendentes), total_questoes, ", ".join(linha["id"] for linha in pendentes)))
        print("A pontuacao abaixo e um INTERVALO. Rode eval.judge_prompt novamente")
        print("com cota disponivel para obter um numero unico.")
        print()
        print("PONTUACAO: entre {:.2f} e {:.2f} de {} ({:.1f}% a {:.1f}%)".format(
            piso, teto, total_questoes, 100 * piso / total_questoes, 100 * teto / total_questoes))
    else:
        print("PONTUACAO: {:.2f} / {} = {:.1f}%".format(
            piso, total_questoes, 100 * piso / total_questoes))

    print()
    print("=== por categoria ===")
    grupos: dict[str, dict[str, float]] = collections.OrderedDict()
    for linha in linhas:
        g = grupos.setdefault(linha["categoria"], {"n": 0, "piso": 0.0, "pendentes": 0})
        g["n"] += 1
        g["piso"] += linha["total"] if linha["total"] is not None else (
            linha["citacao"] + linha["coerencia"])
        g["pendentes"] += 0 if linha["avaliada"] else 1
    for categoria, g in grupos.items():
        pend = "" if not g["pendentes"] else "  ({} nao avaliada(s))".format(int(g["pendentes"]))
        print("{:<32} {:>5.2f}/{:<3} ({:>5.1f}%){}".format(
            categoria[:30], g["piso"], int(g["n"]), 100 * g["piso"] / g["n"], pend))


if __name__ == "__main__":
    main()
