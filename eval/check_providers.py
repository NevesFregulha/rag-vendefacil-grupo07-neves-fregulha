"""Descobre quais modelos estao disponiveis agora, em cada provedor.

Rodar antes de iniciar um benchmark evita comecar uma medicao que vai morrer no
meio por falta de cota - o que ja aconteceu neste projeto e contaminou os
resultados. Cada modelo e testado em duas etapas:

1. chamada do tamanho de uma pergunta real do benchmark, que revela se ha cota
   suficiente - e nao apenas cota para uma chamada curta;
2. saida estruturada, que e o que `src/rag.py` e `eval/judge_prompt.py` exigem.

Um modelo que passa em (1) e falha em (2) nao serve para este projeto.

Atencao: "PRONTO" significa que cabe UMA pergunta, nao as 48 chamadas de um
ciclo completo (24 do benchmark + 24 do juiz). Na Groq o limite e por dia e por
modelo, entao trocar de modelo destrava; no OpenRouter o limite e da conta.

    python -m eval.check_providers

Nao consome cota relevante: sao duas chamadas curtas por modelo.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import config


class _Sonda(BaseModel):
    """Schema minimo, so para verificar se o modelo devolve JSON valido."""

    resposta: str = Field(description="Uma palavra qualquer.")


# Na Groq cada modelo tem cota diaria propria, entao trocar de modelo destrava.
# No OpenRouter o limite e da conta e vale para todos os gratuitos somados.
CANDIDATOS: dict[str, tuple[str, ...]] = {
    "groq": (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "groq/compound",
        "groq/compound-mini",
    ),
    "openrouter": (
        "nex-agi/nex-n2.5-pro:free",
        "dots-studio/dots-3-note-preview:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ),
}


def _construir(provedor: str, modelo: str) -> Any:
    if provedor == "groq":
        from langchain_groq import ChatGroq

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY ausente")
        return ChatGroq(model=modelo, api_key=config.GROQ_API_KEY)

    if provedor == "openrouter":
        import httpx
        from langchain_openai import ChatOpenAI

        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY ausente")
        return ChatOpenAI(
            model=modelo,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            http_client=httpx.Client(timeout=config.OPENROUTER_TIMEOUT_SECONDS),
            max_tokens=config.OPENROUTER_MAX_TOKENS,
        )

    raise ValueError(f"provedor desconhecido: {provedor}")


def _resumir_erro(error: Exception) -> str:
    msg = str(error)
    if "free-models-per-day" in msg:
        return "sem cota (limite diario da CONTA, vale para todos os gratuitos)"
    if "tokens per day" in msg or "TPD" in msg:
        return "sem cota (limite diario deste modelo)"
    if "tokens per minute" in msg or "TPM" in msg:
        return "limite por minuto - tente de novo em instantes"
    if "429" in msg:
        return "429 - limite de uso"
    if "does not exist" in msg or "model_not_found" in msg:
        return "modelo nao existe mais no catalogo"
    return f"{type(error).__name__}: {msg[:70]}"


# Uma pergunta do benchmark envia o contexto recuperado inteiro - cerca de
# 3.700 tokens. Sondar com uma chamada curta responde "tem cota?" para uma
# chamada curta, e nao para um ciclo completo: a primeira versao desta
# ferramenta deu sinal verde e o benchmark morreu na segunda pergunta. O
# preenchimento abaixo aproxima o tamanho real de uma chamada do pipeline.
TOKENS_POR_PERGUNTA = 3700
_ENCHIMENTO = "contexto recuperado de exemplo. " * 620


def testar(provedor: str, modelo: str) -> tuple[str, str]:
    """Devolve (situacao, detalhe) para um modelo."""
    try:
        llm = _construir(provedor, modelo)
    except Exception as error:  # noqa: BLE001
        return "ERRO", _resumir_erro(error)

    try:
        llm.invoke(
            [
                (
                    "human",
                    f"{_ENCHIMENTO}\n\nIgnore o texto acima e responda apenas: ok",
                )
            ]
        )
    except Exception as error:  # noqa: BLE001
        return "INDISPONIVEL", _resumir_erro(error)

    try:
        llm.with_structured_output(_Sonda).invoke(
            [("human", "Devolva o campo resposta com a palavra ok.")]
        )
    except Exception as error:  # noqa: BLE001
        return "PARCIAL", "responde, mas falhou na saida estruturada: " + _resumir_erro(error)

    return "PRONTO", "cota disponivel e saida estruturada funcionando"


def main() -> None:
    print(f"Provedor configurado agora: {config.LLM_PROVIDER} / {config.LLM_MODEL}\n")
    print("{:<14}{:<42}{:<14}{}".format("provedor", "modelo", "situacao", "detalhe"))
    print("-" * 118)

    prontos: list[tuple[str, str]] = []
    for provedor, modelos in CANDIDATOS.items():
        for modelo in modelos:
            situacao, detalhe = testar(provedor, modelo)
            if situacao == "PRONTO":
                prontos.append((provedor, modelo))
            print("{:<14}{:<42}{:<14}{}".format(provedor, modelo, situacao, detalhe))

    print()
    if prontos:
        print("Use qualquer um destes no .env:")
        for provedor, modelo in prontos:
            print(f"  LLM_PROVIDER={provedor}")
            print(f"  LLM_MODEL={modelo}")
            print()
    else:
        print("Nenhum modelo disponivel agora. Aguarde a renovacao da cota.")


if __name__ == "__main__":
    main()
