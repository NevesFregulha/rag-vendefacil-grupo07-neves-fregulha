"""Pipeline de recuperacao e geracao estruturada do RAG VendeFacil."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from src.policy import decide_policy, mask_sensitive_text
from src.retrieve import hybrid_search
from src.schema import RAGResponse


DEFAULT_MAX_ATTEMPTS = 3


class StructuredGenerationError(RuntimeError):
    """Indica que o LLM nao produziu uma resposta valida apos os retries."""


class NoRelevantDocumentsError(RuntimeError):
    """Indica que a busca nao encontrou contexto que permita responder."""


class EvidenceValidationError(ValueError):
    """Indica que uma evidencia nao corresponde aos chunks recuperados."""


def _refusal(reason: str, explanation: str) -> RAGResponse:
    return RAGResponse(
        answer="Nao posso atender a esta solicitacao.",
        confidence_level="Recusado",
        sources_used=[],
        reasoning=explanation,
        is_refusal=True,
        refusal_reason=reason,
    )


def _masked_documents(documents: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=mask_sensitive_text(document.page_content),
            metadata=dict(document.metadata),
        )
        for document in documents
    ]


def _context_from_documents(documents: list[Document]) -> str:
    blocks: list[str] = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    "<fonte>",
                    f"filepath: {metadata.get('source_file', '')}",
                    f"chunk_id: {metadata.get('chunk_id', '')}",
                    f"doc_type: {metadata.get('doc_type', '')}",
                    "conteudo:",
                    document.page_content,
                    "</fonte>",
                ]
            )
        )
    return "\n\n".join(blocks)


def _normalizar_espacos(texto: str) -> str:
    """Colapsa qualquer sequencia de espaco em branco num unico espaco.

    A citacao precisa vir do chunk, nao da imaginacao do modelo - essa e a
    garantia que impede evidencia inventada. Mas exigir correspondencia
    caractere a caractere confunde "trecho diferente" com "mesmo trecho,
    espacamento diferente": os chunks tem quebras de linha e marcacao Markdown,
    e ao copiar um trecho que atravessa uma quebra o modelo normaliza o espaco.
    Tres perguntas do benchmark (Q03, Q09 e Q17) zeravam por isso, citando o
    chunk certo e o texto certo. Comparar com o espaco normalizado preserva a
    garantia e para de reprovar resposta correta por formatacao.
    """
    return " ".join(texto.split())


def _validate_evidence(response: RAGResponse, documents: list[Document]) -> None:
    """Confirma que cada citacao veio de um chunk recuperado.

    A comparacao ignora diferencas de espaco em branco - ver
    ``_normalizar_espacos``.
    """
    available: dict[tuple[str, str], Document] = {}
    for document in documents:
        metadata = document.metadata
        key = (str(metadata.get("source_file", "")), str(metadata.get("chunk_id", "")))
        available[key] = document

    # Nao rejeitamos recusas SEM_EVIDENCIA quando ha documentos no contexto.
    # Essa regra foi testada contra Q04, Q12 e Q21 e refutada: o arquivo certo
    # chegava, mas so com o titulo ou com a secao errada - a recusa do modelo era
    # honesta. Como a busca nunca entrega contexto vazio a este ponto, a regra
    # tornava toda recusa SEM_EVIDENCIA impossivel e convertia recusa em erro.
    for evidence in response.sources_used:
        key = (evidence.filepath, evidence.chunk_id)
        document = available.get(key)
        if document is None:
            raise EvidenceValidationError(
                "Fonte citada nao pertence aos chunks recuperados: "
                f"filepath={evidence.filepath!r}, chunk_id={evidence.chunk_id!r}."
            )
        if _normalizar_espacos(evidence.quotation) not in _normalizar_espacos(
            document.page_content
        ):
            raise EvidenceValidationError(
                f"A quotation do chunk {evidence.chunk_id!r} nao aparece no conteudo "
                "do chunk (comparacao ignora espacos em branco)."
            )
        source_doc_type = document.metadata.get("doc_type")
        if evidence.doc_type is not None and evidence.doc_type != source_doc_type:
            raise EvidenceValidationError(
                f"doc_type incorreto para o chunk {evidence.chunk_id!r}."
            )


# Marcadores de rejeicao de schema feita pelo PROVEDOR, e nao pelo Pydantic.
# Alguns provedores (a Groq, por exemplo) validam o JSON Schema da ferramenta no
# servidor e devolvem HTTP 400 antes de entregar qualquer conteudo. Esse erro nao
# herda de ValueError, entao escapava do laco de retry e a mensagem corretiva -
# mecanismo central desta etapa - nunca era enviada. Traduzimos essas rejeicoes
# na fronteira para que o retry existente volte a funcionar, sem acoplar o
# pipeline a um SDK especifico.
SCHEMA_REJECTION_MARKERS = (
    "tool call validation failed",
    "tool_use_failed",
    "failed to parse tool call",
    "did not match schema",
    # Variante de redacao da mesma rejeicao, observada no benchmark da Etapa 4:
    # "Parsing failed. The model generated output that could not be parsed".
    "parsing failed",
    "could not be parsed",
)


def _correction_message(error: Exception) -> str:
    return (
        "\n\nA tentativa anterior foi invalida. Corrija a resposta sem "
        "alterar os fatos. Erro de validacao: " + str(error)
    )


def _is_schema_rejection(error: Exception) -> bool:
    """Indica se o provedor recusou a saida por violar o schema da ferramenta.

    Erros de outra natureza (autenticacao, rate limit, rede) nao sao tratados
    aqui: repetir a chamada com uma mensagem corretiva nao os resolveria, entao
    eles continuam propagando para o chamador.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None and status != 400:
        return False

    message = str(error).lower()
    return any(marker in message for marker in SCHEMA_REJECTION_MARKERS)


def _coerce_response(value: Any) -> RAGResponse:
    if isinstance(value, RAGResponse):
        return value
    if isinstance(value, Mapping):
        return RAGResponse.model_validate(value)
    raise TypeError(
        "O LLM deve retornar RAGResponse ou um mapping compativel com o schema."
    )


def generate_rag_response(
    vectorstore: Any,
    llm: Any,
    question: str,
    *,
    k: int = 5,
    filters: Mapping[str, str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> RAGResponse:
    """Recupera contexto, aplica a politica e gera uma ``RAGResponse`` valida.

    ``llm`` deve implementar ``with_structured_output(schema)``; isso permite usar
    qualquer chat model LangChain compativel e facilita testes sem chamadas reais.
    """
    if not question or not question.strip():
        raise ValueError("Pergunta vazia.")
    if max_attempts < 1:
        raise ValueError("max_attempts deve ser maior ou igual a 1.")

    initial_policy = decide_policy(question)
    if initial_policy.level == "recusar":
        return _refusal(initial_policy.refusal_reason, initial_policy.reason)

    # complementar_sem_filtro: o objetivo aqui e responder, entao vale trocar a
    # pureza do filtro inferido pela chance de alcancar a fonte certa. Ver
    # src/retrieve.hybrid_search.
    documents = hybrid_search(
        vectorstore, question, k=k, filters=filters, complementar_sem_filtro=True
    )
    if not documents:
        raise NoRelevantDocumentsError(
            "A busca hibrida nao encontrou documentos para fundamentar a resposta."
        )

    final_policy = decide_policy(question, [document.metadata for document in documents])
    effective_documents = (
        _masked_documents(documents)
        if final_policy.level == "mascarar"
        else documents
    )
    context = _context_from_documents(effective_documents)
    structured_llm = llm.with_structured_output(RAGResponse)

    system_prompt = (
        "Voce e o assistente RAG da VendeFacil. Responda somente com fatos do "
        "contexto recuperado e no schema solicitado. Toda resposta deve citar ao "
        "menos uma fonte, copiando uma quotation literal e informando exatamente "
        "filepath, chunk_id e doc_type exibidos. Nao invente fontes nem dados.\n"
        "Se o contexto recuperado nao contiver a resposta, recuse com "
        "refusal_reason='SEM_EVIDENCIA'. Use 'OUT_OF_DOMAIN' apenas quando a "
        "pergunta nao tiver relacao com a operacao da VendeFacil - uma pergunta "
        "sobre produtos, clientes, modulos ou politicas da empresa esta no "
        "dominio mesmo que o contexto recuperado nao a responda."
    )
    correction = ""
    last_error: Exception | None = None
    # Guarda o motivo de cada tentativa. Sem isso, uma falha depois de N
    # tentativas so informava "nao gerou resposta valida", e o erro real ficava
    # no __cause__, que o benchmark descartava - deixando quatro perguntas sem
    # diagnostico possivel.
    tentativas_falhas: list[str] = []

    for _attempt in range(1, max_attempts + 1):
        user_prompt = (
            f"Pergunta:\n{question}\n\nContexto recuperado:\n{context}{correction}"
        )
        try:
            raw_response = structured_llm.invoke(
                [("system", system_prompt), ("human", user_prompt)]
            )
            response = _coerce_response(raw_response)
            _validate_evidence(response, effective_documents)
            return response
        except (OutputParserException, ValidationError, TypeError, ValueError) as error:
            last_error = error
            correction = _correction_message(error)
        except Exception as error:  # noqa: BLE001 - fronteira com o SDK do provedor
            if not _is_schema_rejection(error):
                raise
            last_error = error
            correction = _correction_message(error)
        tentativas_falhas.append(f"tentativa {_attempt}: {type(last_error).__name__}: {last_error}")

    detalhe = " | ".join(tentativas_falhas)
    raise StructuredGenerationError(
        f"O LLM nao gerou uma RAGResponse valida em {max_attempts} tentativa(s). {detalhe}"
    ) from last_error


__all__ = [
    "EvidenceValidationError",
    "NoRelevantDocumentsError",
    "StructuredGenerationError",
    "generate_rag_response",
]
