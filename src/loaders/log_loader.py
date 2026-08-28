"""Carregamento de logs CSV, com um documento por registro."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from langchain_core.documents import Document

from src.metadata import build_metadata


RESTRICTED_PATTERNS = re.compile(
    r"\b(senha|password|credencia(?:l|is)|token|chave\s+(?:de\s+)?api|api[_\s-]?key)\b",
    flags=re.IGNORECASE,
)


def _log_sensitivity(row: dict[str, str]) -> str:
    """Classifica logs com credenciais como restritos."""
    text = " ".join(
        str(row.get(field, ""))
        for field in ("event", "error_code", "message")
    )

    if RESTRICTED_PATTERNS.search(text):
        return "restrito"

    return "interno"


def load_log_documents(file_path: str | Path) -> list[Document]:
    """Lê logs CSV e gera um Document para cada linha."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de logs não encontrado: {path}")

    documents: list[Document] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for line_number, row in enumerate(reader, start=1):
            timestamp = row.get("timestamp", "").strip()
            level = row.get("level", "").strip()
            service = row.get("service", "").strip()
            module = row.get("module", "").strip()
            customer_id = row.get("customer_id", "").strip()
            event = row.get("event", "").strip()
            error_code = row.get("error_code", "").strip()
            message = row.get("message", "").strip()

            page_content = (
                f"Log registrado em {timestamp}. "
                f"Nível: {level}. "
                f"Serviço: {service}. "
                f"Módulo: {module}. "
                f"Cliente: {customer_id}. "
                f"Evento: {event}. "
                f"Código de erro: {error_code}. "
                f"Mensagem: {message}."
            )

            metadata = build_metadata(
                source_file=path,
                doc_type="log",
                chunk_id=f"log-{path.stem}-{timestamp}-{line_number}",
                sensitivity=_log_sensitivity(row),
                timestamp=timestamp,
                date=timestamp[:10],
                level=level,
                service=service,
                module=module,
                customer_id=customer_id,
                event=event,
                error_code=error_code,
            )

            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )

    return documents