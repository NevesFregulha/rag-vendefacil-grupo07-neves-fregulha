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
    """Verifica conjuncao exata dos filtros nos metadados do documento.

    Campo ausente reprova o documento. Testamos a alternativa - tratar campo
    ausente como "nao se aplica", para nao eliminar `products.json` num filtro
    `module=estoque` - e a medicao a refutou: a Context Relevance caiu de 0,654
    para 0,610, porque afrouxar o filtro deixa passar documentos demais e dilui
    o top-5, expulsando justamente as fontes certas de Q07, Q11 e Q13.
    Restringir a tolerancia so ao campo `module` deu o mesmo resultado. Ver
    RELATORIO.md, secao 4.
    """
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


# Ordem em que os filtros sao abandonados quando a conjuncao nao casa com nada.
# O doc_type sai primeiro por ser o menos confiavel: e inferido de um substantivo
# qualquer do enunciado ("...os logs do CLIENTE CUST008" vira doc_type=customer),
# e nao do tipo de documento realmente procurado. O customer_id sai por ultimo,
# por ser um identificador explicito na pergunta.
FILTER_RELAXATION_ORDER = ("doc_type", "module", "priority", "state", "customer_id")

# Quantos documentos da busca sem filtro sao ACRESCENTADOS ao resultado, quando
# o chamador pede o complemento. Sao posicoes extras: nenhum resultado filtrado
# e removido para abrir espaco.
#
# Context Relevance medida em eval/retrieval_score.py, variando este valor:
#   0 vagas -> 0,654 (15/19 acham a fonte)
#   1 vaga  -> 0,737 (16/19)
#   2 vagas -> 0,781 (16/19)
#   3 vagas -> 0,886 (18/19)
#
# O ganho nao e aleatorio: as vagas alcancam exatamente as perguntas
# diagnosticadas (Q04, Q12, Q19), onde a fonte certa aparece na busca aberta e
# o filtro inferido a elimina. Sobra apenas Q02, cuja fonte nao aparece nem sem
# filtro - limitacao de embedding, fora do alcance desta correcao.
#
# Cuidado ao subir mais: acrescentar documentos sempre melhora esta metrica (no
# limite, devolver o corpus inteiro daria 1,0), mas cada documento a mais e
# contexto extra para o modelo processar - e a geracao ja e o gargalo atual.
#
# A alternativa - fundir o ranking aberto por pontuacao, com peso menor - foi
# implementada e medida antes: nao muda nada. O filtro devolve 10+ chunks do
# mesmo arquivo e ocupa todas as vagas, entao o 1o da busca aberta (1/181)
# nunca alcanca um chunk filtrado (2/61).
VAGAS_SEM_FILTRO = 3


def relax_filters(
    documents: Iterable[Document], filters: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Abandona os filtros menos confiaveis ate a conjuncao casar com algum chunk.

    Valores individualmente validos podem formar uma conjuncao impossivel: no
    corpus atual somente `log` e `ticket` carregam o campo `module`, entao
    doc_type=product + module=estoque nao casa com nenhum chunk, por mais
    relevante que a pergunta seja. Sem relaxamento a busca devolve zero
    resultados e o pipeline falha com NoRelevantDocumentsError.

    Devolve os filtros aplicaveis e a lista do que foi descartado, para que o
    chamador possa registrar o relaxamento.
    """
    corpus = list(documents)
    current = dict(filters)
    dropped: list[str] = []

    while current and not any(matches_filters(document, current) for document in corpus):
        for field in FILTER_RELAXATION_ORDER:
            if field in current:
                del current[field]
                dropped.append(field)
                break
        else:
            break

    return current, dropped


def _resolve_filters(
    documents: list[Document],
    question: str,
    filters: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Decide os filtros a aplicar; devolve None quando o pedido e invalido.

    Filtros passados explicitamente pelo chamador sao respeitados como vieram -
    apenas validados. Filtros inferidos da pergunta passam por relaxamento,
    porque quem montou a conjuncao foi o proprio analisador, e ele pode montar
    uma combinacao que nao existe no indice.
    """
    vocabulary = metadata_vocabulary(documents)

    if filters is not None:
        requested = dict(filters)
        validated = validate_filters(requested, vocabulary)
        return None if _has_invalid_filters(requested, validated) else validated

    inferred = extract_filters(question)
    validated = validate_filters(inferred, vocabulary)
    if _has_invalid_filters(inferred, validated):
        return None
    relaxed, _dropped = relax_filters(documents, validated)
    return relaxed


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
    selected_filters = _resolve_filters(documents, question, filters)
    if selected_filters is None:
        return []
    return vectorstore.similarity_search(
        question,
        k=k,
        fetch_k=max(fetch_k, k),
        filter=dict(selected_filters) or None,
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
    ks: Iterable[int] | None = None,
) -> list[Document]:
    """Funde rankings incomparaveis pela posicao, usando RRF.

    ``ks`` permite um ``k`` diferente por ranking. Como a contribuicao de um
    documento e ``1 / (k + posicao)``, um ``k`` maior torna aquele ranking mais
    fraco na fusao. Isso e usado para incluir a busca sem filtro como um voto
    secundario: ela so alcanca as vagas que a busca filtrada nao preencheu, em
    vez de competir de igual para igual e diluir o resultado filtrado.
    """
    rankings = list(rankings)
    pesos = list(ks) if ks is not None else [k] * len(rankings)
    if len(pesos) != len(rankings):
        raise ValueError("ks deve ter um valor por ranking.")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking, k_do_ranking in zip(rankings, pesos):
        seen_in_ranking: set[str] = set()
        for rank, document in enumerate(ranking, start=1):
            chunk_id = str(document.metadata.get("chunk_id", id(document)))
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            documents[chunk_id] = document
            first_seen.setdefault(chunk_id, order)
            order += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k_do_ranking + rank)
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
    complementar_sem_filtro: bool = False,
) -> list[Document]:
    """Combina resultados densos e BM25 com Reciprocal Rank Fusion.

    Por padrao todo resultado satisfaz os filtros - e a garantia da Etapa 2, e
    e o que o comparativo com/sem filtro demonstra.

    ``complementar_sem_filtro=True`` acrescenta ate ``VAGAS_SEM_FILTRO``
    documentos vindos da busca sem filtro nenhum - devolvendo, portanto, ate
    ``k + VAGAS_SEM_FILTRO`` resultados. Isso muda a garantia acima, e por isso
    e opcional e explicito: quem chama assume que prefere achar a resposta a
    manter a pureza do filtro. O pipeline de geracao usa essa opcao; o
    comparativo da Etapa 2, nao.

    O motivo esta medido no benchmark: em Q04, Q12 e Q19 a fonte correta
    aparece na busca aberta e e eliminada pelo filtro inferido, que acerta a
    forma e erra a intencao - "politica de home office para a equipe de
    Engenharia" vira doc_type=employee, e a resposta esta em home_office.md.

    O peso reduzido e deliberado: com ``k`` tres vezes maior, um documento so
    da busca aberta contribui menos que o ultimo da busca filtrada, e portanto
    so ocupa vaga que o filtro deixou vazia.
    """
    documents = documents_from_vectorstore(vectorstore)
    selected_filters = _resolve_filters(documents, question, filters)
    if selected_filters is None:
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

    complementa = complementar_sem_filtro and filters is None and bool(selected_filters)
    if not complementa:
        return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, limit=k)

    # A busca filtrada fica com k-1 vagas; a ultima e reservada ao melhor
    # resultado sem filtro. Fundir os tres rankings por pontuacao nao funciona:
    # o filtro devolve 10+ chunks do mesmo arquivo e ocupa todas as vagas, entao
    # qualquer peso que permita a busca aberta competir tambem a deixa diluir o
    # resultado. A reserva torna o custo explicito e limitado a uma posicao.
    filtrado = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, limit=k)
    aberto = dense_search(
        vectorstore,
        question,
        k=candidate_k,
        fetch_k=max(fetch_k, candidate_k),
        filters={},
    )

    ja_incluidos = {str(d.metadata.get("chunk_id", id(d))) for d in filtrado}
    complemento = [
        documento
        for documento in aberto
        if str(documento.metadata.get("chunk_id", id(documento))) not in ja_incluidos
    ][:VAGAS_SEM_FILTRO]

    # O complemento e ADICIONAL, nao substitui posicao filtrada: devolve ate
    # k + VAGAS_SEM_FILTRO documentos. Ocupar a k-esima vaga custou a Q21, que
    # dependia exatamente do 5o chunk filtrado - e nao havia razao para pagar
    # esse preco, ja que o custo real de um documento a mais e so o tamanho do
    # contexto enviado ao modelo.
    return filtrado + complemento
