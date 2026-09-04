"""Configuracao centralizada de credenciais e parametros do projeto.

Carrega variaveis de .env (nunca commitado - ver .env.example) e expoe os
valores usados pelo pipeline RAG: chave de API do LLM, modelo padrao e
parametros de geracao/avaliacao.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))


def get_llm():
    """Cria o chat model usado pelo pipeline RAG (src/rag.py).

    Requer ANTHROPIC_API_KEY definido em .env. Mantido como factory para que
    os testes continuem usando um LLM fake, sem depender desta funcao.
    """
    from langchain_anthropic import ChatAnthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY nao definido. Copie .env.example para .env e "
            "preencha a chave antes de rodar o pipeline com um LLM real."
        )

    return ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY)