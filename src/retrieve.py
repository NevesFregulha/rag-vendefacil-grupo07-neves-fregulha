"""Recuperacao densa com filtros de metadados validados."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.documents import Document

from src.query_analyzer import extract_filters, metadata_vocabulary, validate_filters


def documents_from_vectorstore(vectorstore: Any) -> list[Document]:
    """Retorna os documentos guardados no docstore do FAISS."""
    index_to_id = getattr(vectorstore, "index_to_docstore_id", {})
    documents: list[Document] = []
    for position in sorted(index_to_id):
        document = vectorstore.docstore.search(index_to_id[position])
        if isinstance(document, Document):
            documents.append(document)
    return documents


def matches_filters(document: Document, filters: Mapping[str, str]) -> bool:
    """Verifica conjuncao exata dos filtros nos metadados do documento."""
    return all(str(document.metadata.get(key, "")) == str(value) for key, value in filters.items())


def analyze_query(question: str, documents: Iterable[Document]) -> dict[str, str]:
    """Extrai filtros e os valida contra o vocabulario real dos documentos."""
    document_list = list(documents)
    return validate_filters(extract_filters(question), metadata_vocabulary(document_list))


def dense_search(
    vectorstore: Any,
    question: str,
    *,
    k: int = 5,
    fetch_k: int = 500,
    filters: Mapping[str, str] | None = None,
) -> list[Document]:
    """Busca no FAISS ampliando o conjunto candidato antes de filtrar."""
    if k < 1 or fetch_k < k:
        raise ValueError("Use k >= 1 e fetch_k >= k.")

    documents = documents_from_vectorstore(vectorstore)
    selected_filters = (
        validate_filters(filters, metadata_vocabulary(documents))
        if filters is not None
        else analyze_query(question, documents)
    )
    search_filter = dict(selected_filters) or None
    return vectorstore.similarity_search(
        question,
        k=k,
        fetch_k=max(fetch_k, k),
        filter=search_filter,
    )
