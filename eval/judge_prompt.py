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

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

import config

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
TRIAD_PATH = Path(__file__).resolve().parent / "triad_scores.json"
RUNS_DIR = Path(__file__).resolve().parent / "runs"

JUDGE_MAX_ATTEMPTS = 3
JUDGE_RETRY_BACKOFF_SECONDS = 20.0

# Quantas perguntas sao julgadas ao mesmo tempo. As avaliacoes sao
# independentes, entao o unico limite real e a cota por minuto do provedor.
JUDGE_MAX_WORKERS = int(os.getenv("JUDGE_MAX_WORKERS", "4"))

# Reaproveita vereditos ja existentes em triad_scores.json. Permite interromper
# a execucao e retomar sem pagar de novo pelo que ja foi avaliado.
REUSAR_VEREDITOS = os.getenv("JUDGE_REUSAR", "1") != "0"

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


def score_question(
    llm: Any, item: dict[str, Any], *, max_attempts: int = JUDGE_MAX_ATTEMPTS
) -> dict[str, Any]:
    """Calcula as tres metricas da RAG Triad para uma unica pergunta.

    O julgamento por LLM tenta novamente com espera crescente, porque o tier
    gratuito da Groq devolve 429 (rate limit por tokens/minuto) sob rajada.
    Se todas as tentativas falharem, registra o erro e segue - uma pergunta
    ruim nao pode derrubar a avaliacao inteira.
    """
    scored: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "context_relevance": context_relevance(item),
        # amarra o veredito a resposta que ele julgou
        "resposta_avaliada": impressao_da_resposta(item),
    }

    if item.get("error"):
        scored["answer_relevance"] = None
        scored["groundedness"] = None
        scored["judge_justification"] = f"Pulado: erro no pipeline ({item['error']})."
        return scored

    for attempt in range(1, max_attempts + 1):
        try:
            verdict = judge_answer(llm, item)
        except Exception as error:  # noqa: BLE001 - fronteira com a API do provedor
            if attempt == max_attempts:
                scored["answer_relevance"] = None
                scored["groundedness"] = None
                scored["judge_justification"] = f"Falha ao julgar: {error}"
                return scored
            time.sleep(JUDGE_RETRY_BACKOFF_SECONDS * attempt)
            continue

        scored["answer_relevance"] = _LEVEL_TO_SCORE[verdict.answer_relevance]
        scored["groundedness"] = _LEVEL_TO_SCORE[verdict.groundedness]
        scored["judge_justification"] = verdict.justification
        return scored

    raise AssertionError("inalcancavel")


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


def _save(scores: list[dict[str, Any]], *, archive_path: Path | None = None) -> None:
    """Grava a avaliacao corrente e, opcionalmente, uma copia datada.

    A copia existe pelo mesmo motivo do arquivamento em run_benchmark.py: uma
    reexecucao sobrescreve o arquivo, e medicoes boas ja foram perdidas assim.
    """
    payload = json.dumps(
        {"scores": scores, "summary": summarize_triad(scores)},
        indent=2,
        ensure_ascii=False,
    )
    TRIAD_PATH.write_text(payload, encoding="utf-8")
    if archive_path is not None:
        archive_path.write_text(payload, encoding="utf-8")


def impressao_da_resposta(item: dict[str, Any]) -> str:
    """Identifica a resposta julgada, para nao reusar veredito de outra saida."""
    conteudo = json.dumps(
        {
            "answer": item.get("answer"),
            "is_refusal": item.get("is_refusal"),
            "refusal_reason": item.get("refusal_reason"),
            "sources_used": item.get("sources_used"),
            "error": item.get("error"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()[:16]


def carregar_vereditos_existentes(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Le avaliacoes ja feitas, para nao pagar de novo pelo que ja foi julgado.

    Duas condicoes para reaproveitar. O veredito precisa estar completo - uma
    pergunta que ficou sem nota porque o juiz falhou e reavaliada. E a resposta
    precisa ser a mesma: o veredito guarda a impressao digital da saida que
    julgou, entao reexecutar o benchmark invalida automaticamente os vereditos
    das respostas que mudaram. Sem isso, o reaproveitamento silenciosamente
    misturaria a nota de uma resposta com o texto de outra.
    """
    if not TRIAD_PATH.exists():
        return {}
    try:
        anteriores = json.loads(TRIAD_PATH.read_text(encoding="utf-8"))["scores"]
    except (json.JSONDecodeError, KeyError):
        return {}

    impressoes = {item["id"]: impressao_da_resposta(item) for item in results}
    aproveitaveis = {}
    for s in anteriores:
        completo = s.get("answer_relevance") is not None or str(
            s.get("judge_justification", "")
        ).startswith("Pulado")
        mesma_resposta = s.get("resposta_avaliada") == impressoes.get(s["id"])
        if completo and mesma_resposta:
            aproveitaveis[s["id"]] = s
    return aproveitaveis


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    llm = config.get_llm()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = RUNS_DIR / f"triad-{carimbo}.json"

    aproveitados = carregar_vereditos_existentes(results) if REUSAR_VEREDITOS else {}
    pendentes = [item for item in results if item["id"] not in aproveitados]
    if aproveitados:
        print(
            f"Reaproveitando {len(aproveitados)} veredito(s) ja existente(s); "
            f"julgando {len(pendentes)}."
        )

    # As perguntas sao independentes entre si, entao sao julgadas em paralelo.
    # A execucao serial com pausa fixa levava ~4 min no melhor caso e passava de
    # 20 min quando a cota estava apertada. O numero de trabalhadores e baixo de
    # proposito: paralelismo demais estoura o limite de tokens por minuto e o
    # backoff devolve o tempo economizado.
    julgados: dict[str, dict[str, Any]] = {}
    total = len(pendentes)
    if total:
        with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as executor:
            futuros = {
                executor.submit(score_question, llm, item): item for item in pendentes
            }
            for concluidos, futuro in enumerate(as_completed(futuros), start=1):
                item = futuros[futuro]
                julgados[item["id"]] = futuro.result()
                print(f"[{concluidos}/{total}] julgado {item['id']}")
                # salva o parcial na ordem original, para nao perder progresso
                _save(
                    [
                        julgados.get(r["id"]) or aproveitados.get(r["id"])
                        for r in results
                        if r["id"] in julgados or r["id"] in aproveitados
                    ],
                    archive_path=archive_path,
                )

    scores = [
        julgados.get(item["id"]) or aproveitados[item["id"]]
        for item in results
        if item["id"] in julgados or item["id"] in aproveitados
    ]
    _save(scores, archive_path=archive_path)

    nao_avaliadas = [s["id"] for s in scores if s["answer_relevance"] is None and not str(
        s.get("judge_justification", "")
    ).startswith("Pulado")]
    if nao_avaliadas:
        print(
            "\nATENCAO: o juiz nao conseguiu avaliar {} pergunta(s): {}".format(
                len(nao_avaliadas), ", ".join(nao_avaliadas)
            )
        )
        print("A pontuacao calculada sobre esta avaliacao sera parcial.")

    summary = summarize_triad(scores)

    print("\nRAG Triad - medias gerais:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.2f}" if value is not None else f"  {metric}: N/A")
    print(f"\nDetalhes por pergunta salvos em: {TRIAD_PATH}")


if __name__ == "__main__":
    main()
