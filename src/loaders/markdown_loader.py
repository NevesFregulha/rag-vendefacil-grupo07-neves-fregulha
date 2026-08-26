"""Carregamento de Markdown por seções e subseções."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.metadata import build_metadata


HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]


def _classification(path: Path) -> tuple[str, str]:
    parts = {part.lower() for part in path.parts}
    if "meetings" in parts:
        return "ata", "interno"
    if "policies" in parts:
        return "policy", "interno"
    return "manual", "publico"


def _section_name(metadata: dict[str, str]) -> str:
    titles = [metadata.get(f"header_{level}") for level in range(1, 5)]
    return " > ".join(title for title in titles if title) or "documento"


def load_markdown_file(
    file_path: str | Path,
    *,
    chunk_size: int = 1_200,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Divide um Markdown por cabeçalhos, com fallback para seções longas."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc_type, sensitivity = _classification(path)
    sections = header_splitter.split_text(text)
    documents: list[Document] = []

    for section_index, section in enumerate(sections, start=1):
        section_name = _section_name(section.metadata)
        parts = size_splitter.split_text(section.page_content)
        for part_index, content in enumerate(parts, start=1):
            chunk_id = f"md-{path.stem}-secao-{section_index}-parte-{part_index}"
            metadata = build_metadata(
                source_file=path,
                doc_type=doc_type,
                chunk_id=chunk_id,
                sensitivity=sensitivity,
                section=section_name,
                section_index=section_index,
            )
            documents.append(Document(page_content=content, metadata=metadata))

    return documents


def load_markdown_documents(root_dir: str | Path) -> list[Document]:
    """Carrega recursivamente todos os arquivos Markdown de uma pasta."""
    root = Path(root_dir)
    documents: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        documents.extend(load_markdown_file(path))
    return documents
