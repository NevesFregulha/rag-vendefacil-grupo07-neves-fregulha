"""Extração e chunking de políticas em PDF."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.metadata import build_metadata


def load_pdf_file(
    file_path: str | Path,
    *,
    chunk_size: int = 1_000,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Extrai cada página e divide parágrafos ou cláusulas longas com overlap."""
    path = Path(file_path)
    reader = PdfReader(str(path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for part_index, content in enumerate(splitter.split_text(text), start=1):
            metadata = build_metadata(
                source_file=path,
                doc_type="policy",
                chunk_id=f"pdf-{path.stem}-pagina-{page_number}-parte-{part_index}",
                sensitivity="interno",
                page=page_number,
            )
            documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise ValueError(f"Nenhum texto pôde ser extraído do PDF {path.name}.")
    return documents


def load_pdf_documents(root_dir: str | Path) -> list[Document]:
    """Carrega recursivamente todos os PDFs de uma pasta."""
    root = Path(root_dir)
    documents: list[Document] = []
    for path in sorted(root.rglob("*.pdf")):
        documents.extend(load_pdf_file(path))
    return documents
