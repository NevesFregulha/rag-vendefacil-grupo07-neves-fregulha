"""Interface de demonstracao do assistente RAG VendeFacil (Streamlit).

Ligada ao pipeline real da Etapa 3: recuperacao hibrida (src/retrieve.py),
politica de LGPD (src/policy.py) e geracao estruturada validada por Pydantic
(src/rag.py + src/schema.py).

Executar:
    streamlit run app.py

Requer o indice FAISS ja criado (python -m src.vectorstore) e o .env
preenchido com a chave do provedor de LLM.
"""

from __future__ import annotations

import streamlit as st

import config
from src.rag import generate_rag_response
from src.vectorstore import load_vectorstore

CONFIDENCE_COLORS = {
    "Alta": "🟢",
    "Média": "🟡",
    "Baixa": "🟠",
    "Recusado": "🔴",
}

REFUSAL_LABELS = {
    "LGPD_PROTECTION": "Dado pessoal protegido pela LGPD",
    "CREDENTIAL_PROTECTION": "Credencial ou dado de acesso",
    "OUT_OF_DOMAIN": "Fora do escopo da VendeFácil",
}

PERGUNTAS_EXEMPLO = [
    "Quais são os produtos oferecidos pela VendeFácil?",
    "Quais tickets de Minas Gerais estão relacionados ao módulo de estoque?",
    "Qual é o prazo de arrependimento para reembolso integral?",
    "Qual é o salário da funcionária Ana Souza?",
]

st.set_page_config(page_title="VendeFácil RAG", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Carregando índice FAISS e modelo de embeddings...")
def carregar_pipeline():
    """Carrega indice e LLM uma unica vez por sessao do servidor.

    O cache e essencial na demo: recarregar o FAISS e o modelo de embeddings a
    cada pergunta levaria dezenas de segundos.
    """
    return load_vectorstore(), config.get_llm()


def renderar_fontes(sources: list[dict]) -> None:
    """Mostra as evidencias citadas, com arquivo, chunk_id e trecho literal."""
    if not sources:
        return
    with st.expander(f"📎 Evidências citadas ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['filepath']}** · `{source['chunk_id']}`")
            st.caption(f"„{source['quotation']}”")


def renderar_resposta(payload: dict) -> None:
    """Renderiza uma resposta ja normalizada em dicionario."""
    if payload.get("erro"):
        st.error(payload["erro"])
        return

    if payload["is_refusal"]:
        motivo = REFUSAL_LABELS.get(payload["refusal_reason"], payload["refusal_reason"])
        st.warning(f"**Solicitação recusada** — {motivo}")

    st.markdown(payload["answer"])

    icone = CONFIDENCE_COLORS.get(payload["confidence_level"], "⚪")
    st.caption(f"{icone} Confiança: **{payload['confidence_level']}**")

    renderar_fontes(payload["sources_used"])

    if payload.get("reasoning"):
        with st.expander("🧠 Raciocínio"):
            st.caption(payload["reasoning"])


def responder(pergunta: str) -> dict:
    """Chama o pipeline RAG real e normaliza o resultado para exibicao."""
    vectorstore, llm = carregar_pipeline()
    try:
        resposta = generate_rag_response(
            vectorstore,
            llm,
            pergunta,
            k=config.RETRIEVAL_K,
            max_attempts=config.MAX_GENERATION_ATTEMPTS,
        )
    except Exception as error:  # noqa: BLE001 - a demo nao pode quebrar na tela
        return {"erro": f"Não foi possível responder: {error}"}

    return {
        "answer": resposta.answer,
        "confidence_level": resposta.confidence_level,
        "is_refusal": resposta.is_refusal,
        "refusal_reason": resposta.refusal_reason,
        "reasoning": resposta.reasoning,
        "sources_used": [fonte.model_dump() for fonte in resposta.sources_used],
    }


# ---------------------------------------------------------------- barra lateral

with st.sidebar:
    st.header("Configuração")
    st.markdown(
        f"""
        - **Provedor:** `{config.LLM_PROVIDER}`
        - **Modelo:** `{config.LLM_MODEL}`
        - **Chunks recuperados:** `{config.RETRIEVAL_K}`
        - **Tentativas de geração:** `{config.MAX_GENERATION_ATTEMPTS}`
        """
    )
    st.divider()
    st.subheader("Perguntas de exemplo")
    for exemplo in PERGUNTAS_EXEMPLO:
        if st.button(exemplo, use_container_width=True):
            st.session_state.pergunta_pendente = exemplo
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()

# ------------------------------------------------------------------ area central

st.title("🔎 VendeFácil — Assistente RAG")
st.caption(
    "Consulta a base interna da VendeFácil com busca híbrida, citação de evidência "
    "e guardrails de LGPD. Toda resposta cita o arquivo e o chunk de origem."
)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        if mensagem["role"] == "user":
            st.markdown(mensagem["content"])
        else:
            renderar_resposta(mensagem["content"])

pergunta = st.chat_input("Pergunte algo sobre a base da VendeFácil...")

if not pergunta and "pergunta_pendente" in st.session_state:
    pergunta = st.session_state.pop("pergunta_pendente")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Recuperando evidências e gerando resposta..."):
            payload = responder(pergunta)
        renderar_resposta(payload)

    st.session_state.mensagens.append({"role": "assistant", "content": payload})
