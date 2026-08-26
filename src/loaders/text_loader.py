"""Carregamento de e-mails TXT por mensagem do thread."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.metadata import build_metadata


MESSAGE_START = re.compile(r"(?=^De:\s*)", flags=re.MULTILINE | re.IGNORECASE)
RESTRICTED_PATTERNS = re.compile(
    r"\b(senha|credencia(?:l|is)|chave\s+(?:de\s+)?api|token|cpf|pix|dados\s+bancários)\b",
    flags=re.IGNORECASE,
)
CUSTOMER_ID_PATTERN = re.compile(r"\bCUST\d+\b", flags=re.IGNORECASE)
DATE_PATTERN = re.compile(r"^Data:\s*(.+)$", flags=re.MULTILINE | re.IGNORECASE)


def _split_messages(text: str) -> list[str]:
    messages = [part.strip() for part in MESSAGE_START.split(text) if part.strip()]
    return messages or ([text.strip()] if text.strip() else [])


def _sensitivity(path: Path, message: str) -> str:
    if path.stem.lower().startswith("internal_") and RESTRICTED_PATTERNS.search(message):
        return "restrito"
    return "interno"


def load_text_file(
    file_path: str | Path,
    *,
    chunk_size: int = 1_200,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Separa o thread antes de dividir mensagens que excedem o limite."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []

    for message_index, message in enumerate(_split_messages(text), start=1):
        customer_match = CUSTOMER_ID_PATTERN.search(message)
        date_match = DATE_PATTERN.search(message)
        parts = splitter.split_text(message)

        for part_index, content in enumerate(parts, start=1):
            metadata = build_metadata(
                source_file=path,
                doc_type="email",
                chunk_id=(
                    f"email-{path.stem}-mensagem-{message_index}-parte-{part_index}"
                ),
                sensitivity=_sensitivity(path, message),
                customer_id=(customer_match.group(0).upper() if customer_match else None),
                date=(date_match.group(1).strip() if date_match else None),
                message_index=message_index,
            )
            documents.append(Document(page_content=content, metadata=metadata))

    return documents


def load_text_documents(root_dir: str | Path) -> list[Document]:
    """Carrega recursivamente todos os arquivos TXT de uma pasta."""
    root = Path(root_dir)
    documents: list[Document] = []
    for path in sorted(root.rglob("*.txt")):
        documents.extend(load_text_file(path))
    return documents
