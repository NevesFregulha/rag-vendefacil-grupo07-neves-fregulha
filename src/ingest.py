"""Orquestração da ingestão heterogênea da Etapa 1."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from src.loaders.jsonl_loader import load_jsonl_documents
from src.loaders.log_loader import load_log_documents
from src.loaders.markdown_loader import load_markdown_documents
from src.loaders.pdf_loader import load_pdf_documents
from src.loaders.structured import load_structured_documents
from src.loaders.text_loader import load_text_documents
from src.metadata import validate_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _validate_required_files(data_dir: Path) -> None:
    """Garante que todas as fontes obrigatórias estejam disponíveis."""
    required_paths = [
        data_dir / "structured" / "customers.csv",
        data_dir / "structured" / "employees.csv",
        data_dir / "structured" / "sales.csv",
        data_dir / "structured" / "products.json",
        data_dir / "structured" / "stores.json",
        data_dir / "semi_structured" / "tickets.jsonl",
        data_dir / "semi_structured" / "system_logs.csv",
        data_dir / "unstructured",
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Fontes obrigatórias não encontradas:\n{missing}"
        )


def load_all_documents(
    data_dir: str | Path | None = None,
) -> list[Document]:
    """Carrega e valida todos os documentos do corpus VendeFácil."""
    base_data_dir = (
        Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    )

    _validate_required_files(base_data_dir)

    documents: list[Document] = []

    documents.extend(
        load_structured_documents(base_data_dir / "structured")
    )

    documents.extend(
        load_log_documents(
            base_data_dir / "semi_structured" / "system_logs.csv"
        )
    )

    documents.extend(
        load_jsonl_documents(
            base_data_dir / "semi_structured" / "tickets.jsonl"
        )
    )

    documents.extend(
        load_markdown_documents(base_data_dir / "unstructured")
    )

    documents.extend(
        load_pdf_documents(
            base_data_dir / "unstructured" / "policies"
        )
    )

    documents.extend(
        load_text_documents(
            base_data_dir / "unstructured" / "emails"
        )
    )

    validate_documents(documents)

    return documents


def main() -> None:
    """Executa a ingestão e imprime um resumo do corpus."""
    documents = load_all_documents()
    distribution = Counter(
        document.metadata["doc_type"] for document in documents
    )

    print(f"Total de chunks: {len(documents)}")
    print("Distribuição por doc_type:")

    for doc_type, quantity in sorted(distribution.items()):
        print(f"- {doc_type}: {quantity}")


if __name__ == "__main__":
    main()