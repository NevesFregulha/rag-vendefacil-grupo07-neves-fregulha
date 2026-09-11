"""Configuracao centralizada de credenciais e parametros do projeto.

Carrega variaveis de .env (nunca commitado - ver .env.example) e expoe os
valores usados pelo pipeline RAG: provedor e chave de API do LLM, modelo
padrao e parametros de geracao/avaliacao.

LLM_PROVIDER escolhe o provedor: "openrouter" (padrao), "groq" ou "anthropic".
O padrao e o OpenRouter, com um modelo gratuito fixo que nao exige cartao de
credito. A Groq segue disponivel como alternativa, e foi o provedor usado nas
medicoes registradas no RELATORIO.md.

Trocar de provedor e uma mudanca de variavel de ambiente: o pipeline em
src/rag.py recebe o modelo ja pronto e nao conhece nenhum SDK especifico.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# O OpenRouter expoe uma API compativel com a da OpenAI, entao e alcancado pelo
# cliente ChatOpenAI apontado para esta base.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "8192"))

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-5",
    # Modelo gratuito FIXO, e nao o roteador `openrouter/free`. O pipeline
    # depende de `with_structured_output`, e o roteador escolhe um modelo
    # diferente a cada chamada - alguns nao fazem tool calling, outros sao de
    # raciocinio e esgotam o limite de tokens antes de escrever o JSON.
    # Medido em 3 tentativas de saida estruturada:
    #   openrouter/free                          1/3
    #   nvidia/nemotron-3-super-120b-a12b:free   2/3
    #   nex-agi/nex-n2.5-pro:free                3/3  <- escolhido
    #   dots-studio/dots-3-note-preview:free     3/3
    # Para conferir outros: https://openrouter.ai/api/v1/models, campo
    # `supported_parameters` deve conter "tools".
    "openrouter": "nex-agi/nex-n2.5-pro:free",
}

LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))


def get_llm():
    """Cria o chat model usado pelo pipeline RAG (src/rag.py) e pelo benchmark.

    Escolhe o provedor pela variavel LLM_PROVIDER: "openrouter" (padrao),
    "groq" ou "anthropic". Mantido como factory para que os testes continuem
    usando um LLM fake, sem depender desta funcao.
    """
    if LLM_PROVIDER == "openrouter":
        import httpx
        from langchain_openai import ChatOpenAI

        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY nao definido. Copie .env.example para .env e "
                "preencha a chave antes de rodar o pipeline com um LLM real."
            )
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            # Cliente HTTP explicito de proposito. O SDK da OpenAI monta o dele
            # sozinho, e essa construcao entra em recursao infinita quando o
            # pacote pip-system-certs esta instalado - ele faz monkeypatch no
            # modulo ssl para usar o repositorio de certificados do Windows, o
            # que e necessario aqui por causa da interceptacao de TLS do
            # antivirus. O sintoma e um "Connection error" generico que esconde
            # um RecursionError. Passar o cliente pronto contorna o problema.
            http_client=httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS),
            # Modelos gratuitos costumam ser de raciocinio e gastam tokens antes
            # de escrever a resposta; um limite baixo devolve conteudo vazio.
            max_tokens=OPENROUTER_MAX_TOKENS,
        )

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
        f"LLM_PROVIDER invalido: {LLM_PROVIDER!r}. "
        "Use 'openrouter', 'groq' ou 'anthropic'."
    )
