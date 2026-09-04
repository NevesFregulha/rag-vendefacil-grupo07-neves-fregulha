"""Configuracao centralizada de credenciais e parametros do projeto.

Carrega variaveis de .env (nunca commitado - ver .env.example) e expoe os
valores usados pelo pipeline RAG: provedor e chave de API do LLM, modelo
padrao e parametros de geracao/avaliacao.

LLM_PROVIDER escolhe o provedor ("groq" ou "anthropic"). Padrao "groq" porque
tem tier gratuito sem cartao de credito, o que evita bloquear o benchmark da
Etapa 4 por falta de credito de API.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-5",
}

LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))


def get_llm():
    """Cria o chat model usado pelo pipeline RAG (src/rag.py) e pelo benchmark.

    Escolhe o provedor pela variavel LLM_PROVIDER ("groq" ou "anthropic").
    Mantido como factory para que os testes continuem usando um LLM fake, sem
    depender desta funcao.
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY nao definido. Copie .env.example para .env e "
                "preencha a chave antes de rodar o pipeline com um LLM real."
            )
        return ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY nao definido. Copie .env.example para .env e "
                "preencha a chave antes de rodar o pipeline com um LLM real."
            )
        return ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY)

    raise ValueError(
        f"LLM_PROVIDER invalido: {LLM_PROVIDER!r}. Use 'groq' ou 'anthropic'."
    )
