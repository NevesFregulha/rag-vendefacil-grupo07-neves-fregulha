"""Recuperacao densa com filtros de metadados validados."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Plus

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


def _has_invalid_filters(
    requested: Mapping[str, str], validated: Mapping[str, str]
) -> bool:
    """Impede que filtro invalido seja removido e amplie silenciosamente a busca."""
    return bool(requested) and set(requested) != set(validated)


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
    requested_filters = dict(filters) if filters is not None else extract_filters(question)
    selected_filters = validate_filters(
        requested_filters, metadata_vocabulary(documents)
    )
    if _has_invalid_filters(requested_filters, selected_filters):
        return []
    search_filter = dict(selected_filters) or None
    return vectorstore.similarity_search(
        question,
        k=k,
        fetch_k=max(fetch_k, k),
        filter=search_filter,
    )


def tokenize(text: str) -> list[str]:
    """Normaliza texto em tokens adequados ao BM25 em portugues."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", without_accents)


def bm25_search(
    documents: Iterable[Document],
    question: str,
    *,
    k: int = 5,
    filters: Mapping[str, str] | None = None,
) -> list[Document]:
    """Executa busca esparsa, pre-filtrando o corpus quando necessario."""
    if k < 1:
        raise ValueError("Use k >= 1.")
    corpus = list(documents)
    requested_filters = dict(filters or {})
    selected_filters = validate_filters(requested_filters, metadata_vocabulary(corpus))
    if _has_invalid_filters(requested_filters, selected_filters):
        return []
    if selected_filters:
        corpus = [document for document in corpus if matches_filters(document, selected_filters)]
    if not corpus:
        return []

    tokenized_corpus = [tokenize(document.page_content) for document in corpus]
    # BM25Plus evita IDF zero em subconjuntos pequenos gerados por filtros.
    bm25 = BM25Plus(tokenized_corpus)
    scores = bm25.get_scores(tokenize(question))
    ranking = sorted(range(len(corpus)), key=lambda index: (-scores[index], index))
    return [corpus[index] for index in ranking[:k]]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Document]],
    *,
    k: int = 60,
    limit: int = 5,
) -> list[Document]:
    """Funde rankings incomparaveis pela posicao, usando RRF."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, document in enumerate(ranking, start=1):
            chunk_id = str(document.metadata.get("chunk_id", id(document)))
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            documents[chunk_id] = document
            first_seen.setdefault(chunk_id, order)
            order += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    ordered_ids = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return [documents[chunk_id] for chunk_id in ordered_ids[:limit]]


def hybrid_search(
    vectorstore: Any,
    question: str,
    *,
    k: int = 5,
    fetch_k: int = 500,
    filters: Mapping[str, str] | None = None,
    rrf_k: int = 60,
) -> list[Document]:
    """Combina resultados densos e BM25 com Reciprocal Rank Fusion."""
    documents = documents_from_vectorstore(vectorstore)
    requested_filters = dict(filters) if filters is not None else extract_filters(question)
    selected_filters = validate_filters(
        requested_filters, metadata_vocabulary(documents)
    )
    if _has_invalid_filters(requested_filters, selected_filters):
        return []
    candidate_k = max(k * 4, k)
    dense_results = dense_search(
        vectorstore,
        question,
        k=candidate_k,
        fetch_k=max(fetch_k, candidate_k),
        filters=selected_filters,
    )
    sparse_results = bm25_search(
        documents,
        question,
        k=candidate_k,
        filters=selected_filters,
    )
    return reciprocal_rank_fusion(
        [dense_results, sparse_results], k=rrf_k, limit=k
    )
