"""Analisador de perguntas: extração de filtros de metadados a partir de texto livre."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

FILTER_FIELDS = ("doc_type", "state", "module", "customer_id", "priority")

DOC_TYPE_ALIASES: dict[str, str] = {
    "tickets": "ticket",
    "ticket": "ticket",
    "chamados": "ticket",
    "chamado": "ticket",
    "clientes": "customer",
    "cliente": "customer",
    "funcionarios": "employee",
    "funcionario": "employee",
    "produtos": "product",
    "produto": "product",
    "lojas": "store",
    "vendas": "sale",
    "logs": "log",
    "manuais": "manual",
    "manual": "manual",
    "politicas": "policy",
    "politica": "policy",
    "emails": "email",
    "atas": "ata",
}

# Estados presentes no corpus (customers.csv, stores.json, tickets.jsonl, system_logs.csv).
STATE_NAMES: dict[str, str] = {
    "bahia": "BA",
    "ceara": "CE",
    "ceará": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "espírito santo": "ES",
    "goias": "GO",
    "goiás": "GO",
    "minas gerais": "MG",
    "pernambuco": "PE",
    "parana": "PR",
    "paraná": "PR",
    "rio de janeiro": "RJ",
    "rio grande do sul": "RS",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "são paulo": "SP",
}

KNOWN_UFS = frozenset(STATE_NAMES.values())

# Módulos presentes em tickets.jsonl e system_logs.csv, com sinônimos usados em perguntas.
# "loja" sozinho não é mapeado: colide com "lojas" (doc_type=store), então só
# variações que deixam claro tratar-se do módulo e-commerce são reconhecidas.
MODULE_ALIASES: dict[str, str] = {
    "pdv": "pdv",
    "frente de caixa": "pdv",
    "ponto de venda": "pdv",
    "caixa": "pdv",
    "estoque": "estoque",
    "inventario": "estoque",
    "inventário": "estoque",
    "nf-e": "estoque",
    "nfe": "estoque",
    "loja online": "ecommerce",
    "lojas online": "ecommerce",
    "e-commerce": "ecommerce",
    "ecommerce": "ecommerce",
    "catalogo": "ecommerce",
    "catálogo": "ecommerce",
    "analytics": "analytics",
    "dashboard": "analytics",
    "dre": "analytics",
    "curva abc": "analytics",
    "pay": "pay",
    "pagamento": "pay",
    "pagamentos": "pay",
    "tef": "pay",
    "pix": "pay",
}

# Prioridades usadas em tickets.jsonl, com variação sem acento.
PRIORITY_ALIASES: dict[str, str] = {
    "alta": "Alta",
    "baixa": "Baixa",
    "media": "Média",
    "média": "Média",
    "critica": "Crítica",
    "crítica": "Crítica",
}

CUSTOMER_ID_PATTERN = re.compile(r"\bCUST-?\d+\b", flags=re.IGNORECASE)
UF_PATTERN = re.compile(r"\b([A-Za-z]{2})\b")


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _extract_doc_type(question_lower: str) -> str | None:
    normalized = _strip_accents(question_lower)
    for alias, doc_type in DOC_TYPE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return doc_type
    return None


def _extract_state(question: str, question_lower: str) -> str | None:
    for name, uf in STATE_NAMES.items():
        if name in question_lower:
            return uf

    for match in UF_PATTERN.finditer(question):
        candidate = match.group(1).upper()
        if candidate in KNOWN_UFS:
            return candidate

    return None


def _extract_module(question_lower: str) -> str | None:
    normalized = _strip_accents(question_lower)
    for alias, module in MODULE_ALIASES.items():
        if _strip_accents(alias) in normalized:
            return module
    return None


def _extract_customer_id(question: str) -> str | None:
    match = CUSTOMER_ID_PATTERN.search(question)
    if not match:
        return None

    digits = re.sub(r"\D", "", match.group(0))
    return f"CUST{digits}"


def _extract_priority(question_lower: str) -> str | None:
    normalized = _strip_accents(question_lower)
    for alias, priority in PRIORITY_ALIASES.items():
        if _strip_accents(alias) in normalized:
            return priority
    return None


def extract_filters(question: str) -> dict[str, str]:
    """Extrai os filtros de metadados reconhecíveis em uma pergunta em texto livre.

    Retorna um dicionário apenas com as chaves detectadas, dentre
    ``state``, ``module``, ``customer_id`` e ``priority``.
    """
    if not question or not question.strip():
        return {}

    question_lower = question.lower()
    filters: dict[str, str] = {}

    doc_type = _extract_doc_type(question_lower)
    if doc_type:
        filters["doc_type"] = doc_type

    state = _extract_state(question, question_lower)
    if state:
        filters["state"] = state

    module = _extract_module(question_lower)
    if module:
        filters["module"] = module

    customer_id = _extract_customer_id(question)
    if customer_id:
        filters["customer_id"] = customer_id

    priority = _extract_priority(question_lower)
    if priority:
        filters["priority"] = priority

    return filters


def metadata_vocabulary(
    documents: Iterable[Any],
    fields: Iterable[str] = FILTER_FIELDS,
) -> dict[str, set[str]]:
    """Coleta os valores realmente existentes para cada campo filtravel."""
    vocabulary = {field: set() for field in fields}
    for document in documents:
        metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        for field in vocabulary:
            value = metadata.get(field)
            if value is not None and str(value).strip():
                vocabulary[field].add(str(value).strip())
    return vocabulary


def validate_filters(
    filters: Mapping[str, str],
    vocabulary: Mapping[str, set[str]],
) -> dict[str, str]:
    """Mantem apenas filtros cujo campo e valor existem no corpus indexado."""
    validated: dict[str, str] = {}
    for field, value in filters.items():
        if field not in FILTER_FIELDS:
            continue
        allowed_values = vocabulary.get(field, set())
        match = next(
            (
                candidate
                for candidate in allowed_values
                if _strip_accents(candidate.casefold())
                == _strip_accents(str(value).strip().casefold())
            ),
            None,
        )
        if match is not None:
            validated[field] = match
    return validated
