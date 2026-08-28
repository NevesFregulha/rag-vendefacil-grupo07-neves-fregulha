"""Criação, persistência e recarga do índice FAISS."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.metadata import validate_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def create_embeddings() -> HuggingFaceEmbeddings:
    """Cria o modelo local de embeddings multilíngue."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_and_save_vectorstore(
    documents: list[Document],
    index_dir: str | Path | None = None,
) -> FAISS:
    """Vetoriza documentos e persiste o índice FAISS em disco."""
    if not documents:
        raise ValueError("Não é possível criar um índice sem documentos.")

    validate_documents(documents)

    destination = (
        Path(index_dir)
        if index_dir is not None
        else DEFAULT_INDEX_DIR
    )
    destination.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()
    chunk_ids = [
        str(document.metadata["chunk_id"])
        for document in documents
    ]

    vectorstore = FAISS.from_documents(
        documents,
        embeddings,
        ids=chunk_ids,
    )

    vectorstore.save_local(str(destination))

    return vectorstore


def load_vectorstore(
    index_dir: str | Path | None = None,
) -> FAISS:
    """Recarrega o índice FAISS persistido, sem reindexar o corpus."""
    source = Path(index_dir) if index_dir is not None else DEFAULT_INDEX_DIR

    if not source.exists():
        raise FileNotFoundError(
            f"Índice FAISS não encontrado em: {source}"
        )

    embeddings = create_embeddings()

    return FAISS.load_local(
        str(source),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main() -> None:
    """Gera e salva o índice a partir do corpus completo."""
    from src.ingest import load_all_documents

    documents = load_all_documents()
    build_and_save_vectorstore(documents)

    print(f"Índice FAISS salvo em: {DEFAULT_INDEX_DIR}")
    print(f"Documentos indexados: {len(documents)}")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()