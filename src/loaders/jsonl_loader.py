"""Carregamento adaptativo dos tickets armazenados em JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.metadata import build_metadata


def _ticket_header(ticket: dict[str, Any]) -> str:
    """Serializa os campos que identificam o ticket e devem ser replicados."""
    fields = (
        ("Ticket", ticket.get("ticket_id")),
        ("Cliente", ticket.get("customer_id")),
        ("Nome do cliente", ticket.get("customer_name")),
        ("Estado", ticket.get("state")),
        ("Módulo", ticket.get("module")),
        ("Título", ticket.get("title")),
        ("Prioridade", ticket.get("priority")),
        ("Status", ticket.get("status")),
        ("Data de abertura", ticket.get("created_at")),
        ("Categoria", ticket.get("category")),
    )
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def _ticket_body(ticket: dict[str, Any]) -> str:
    """Serializa apenas o corpo, que pode ser dividido quando for longo."""
    sections = []
    if ticket.get("description"):
        sections.append(f"Descrição: {ticket['description']}")
    if ticket.get("resolution"):
        sections.append(f"Resolução: {ticket['resolution']}")
    if ticket.get("sentiment"):
        sections.append(f"Sentimento: {ticket['sentiment']}")
    return "\n\n".join(sections)


def load_jsonl_documents(
    file_path: str | Path,
    *,
    chunk_size: int = 1_200,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Lê tickets JSONL, mantendo um ticket por chunk sempre que possível."""
    path = Path(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                ticket = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido em {path.name}, linha {line_number}: {error.msg}."
                ) from error

            ticket_id = str(ticket.get("ticket_id") or f"linha-{line_number}")
            header = _ticket_header(ticket)
            body = _ticket_body(ticket)
            body_parts = splitter.split_text(body) if body else [""]

            for part_index, body_part in enumerate(body_parts, start=1):
                content = f"{header}\n\n{body_part}".strip()
                chunk_id = f"tickets-{ticket_id}"
                if len(body_parts) > 1:
                    chunk_id = f"{chunk_id}-parte-{part_index}"

                metadata = build_metadata(
                    source_file=path,
                    doc_type="ticket",
                    chunk_id=chunk_id,
                    sensitivity="interno",
                    customer_id=ticket.get("customer_id"),
                    state=ticket.get("state"),
                    module=ticket.get("module"),
                    priority=ticket.get("priority"),
                    status=ticket.get("status"),
                    date=ticket.get("created_at"),
                    ticket_id=ticket.get("ticket_id"),
                )
                documents.append(Document(page_content=content, metadata=metadata))

    return documents
