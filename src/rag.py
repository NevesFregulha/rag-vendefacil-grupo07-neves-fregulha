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


def _validate_evidence(response: RAGResponse, documents: list[Document]) -> None:
    """Confirma que cada citacao veio literalmente de um chunk recuperado."""
    available: dict[tuple[str, str], Document] = {}
    for document in documents:
        metadata = document.metadata
        key = (str(metadata.get("source_file", "")), str(metadata.get("chunk_id", "")))
        available[key] = document

    for evidence in response.sources_used:
        key = (evidence.filepath, evidence.chunk_id)
        document = available.get(key)
        if document is None:
            raise EvidenceValidationError(
                "Fonte citada nao pertence aos chunks recuperados: "
                f"filepath={evidence.filepath!r}, chunk_id={evidence.chunk_id!r}."
            )
        if evidence.quotation not in document.page_content:
            raise EvidenceValidationError(
                f"A quotation do chunk {evidence.chunk_id!r} nao e um trecho literal."
            )
        source_doc_type = document.metadata.get("doc_type")
        if evidence.doc_type is not None and evidence.doc_type != source_doc_type:
            raise EvidenceValidationError(
                f"doc_type incorreto para o chunk {evidence.chunk_id!r}."
            )


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

    documents = hybrid_search(vectorstore, question, k=k, filters=filters)
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
        "filepath, chunk_id e doc_type exibidos. Nao invente fontes nem dados."
    )
    correction = ""
    last_error: Exception | None = None

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
            correction = (
                "\n\nA tentativa anterior foi invalida. Corrija a resposta sem "
                "alterar os fatos. Erro de validacao: " + str(error)
            )

    raise StructuredGenerationError(
        f"O LLM nao gerou uma RAGResponse valida em {max_attempts} tentativa(s)."
    ) from last_error


__all__ = [
    "EvidenceValidationError",
    "NoRelevantDocumentsError",
    "StructuredGenerationError",
    "generate_rag_response",
]
