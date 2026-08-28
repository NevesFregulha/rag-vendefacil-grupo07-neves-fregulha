import csv
import json
from pathlib import Path

from langchain_core.documents import Document


def load_csv_file(
    file_path: str,
    doc_type: str,
    sensitivity: str,
) -> list[Document]:
    """Lê um arquivo CSV e transforma cada linha em um Document do LangChain."""
    documents = []
    path = Path(file_path)

    if not path.exists():
        return documents

    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for index, row in enumerate(reader):
            raw_id = (
                row.get("customer_id")
                or row.get("employee_id")
                or row.get("sale_id")
                or row.get("id")
                or f"row-{index + 1}"
            )

            content_parts = [f"{key}: {value}" for key, value in row.items() if value]
            page_content = (
                f"Registro do tipo {doc_type} ({path.name}): "
                + ", ".join(content_parts)
            )

            chunk_id = f"{path.stem}-{raw_id}-{index + 1}"

            metadata = {
                "source_file": path.name,
                "doc_type": doc_type,
                "chunk_id": chunk_id,
                "sensitivity": sensitivity,
            }

            for key in [
                "customer_id",
                "state",
                "module",
                "status",
                "date",
                "employee_id",
                "sale_id",
            ]:
                if key in row:
                    metadata[key] = row[key]

            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )

    return documents


def load_json_file(
    file_path: str,
    doc_type: str,
    sensitivity: str,
    records_key: str | None = None,
) -> list[Document]:
    """Lê registros JSON e transforma cada registro em um Document."""
    documents = []
    path = Path(file_path)

    if not path.exists():
        return documents

    with open(path, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data
    elif records_key is not None:
        records = data.get(records_key)

        if not isinstance(records, list):
            raise ValueError(
                f"Esperada uma lista na chave {records_key!r} de {path.name}."
            )
    else:
        raise ValueError(
            f"Esperada uma lista de registros em {path.name}."
        )

    for index, row in enumerate(records):
        raw_id = (
            row.get("product_id")
            or row.get("store_id")
            or row.get("id")
            or f"item-{index + 1}"
        )

        content_parts = [f"{key}: {value}" for key, value in row.items() if value]
        page_content = (
            f"Registro do tipo {doc_type} ({path.name}): "
            + ", ".join(content_parts)
        )

        chunk_id = f"{path.stem}-{raw_id}-{index + 1}"

        metadata = {
            "source_file": path.name,
            "doc_type": doc_type,
            "chunk_id": chunk_id,
            "sensitivity": sensitivity,
        }

        for key in [
            "product_id",
            "store_id",
            "customer_id",
            "state",
            "category",
            "city",
        ]:
            if key in row:
                metadata[key] = str(row[key])

        documents.append(
            Document(page_content=page_content, metadata=metadata)
        )

    return documents


def load_structured_documents(
    data_dir: str = "data/structured",
) -> list[Document]:
    """Carrega todos os arquivos estruturados do diretório informado."""
    base_path = Path(data_dir)
    all_documents = []

    configs = [
        {
            "file": "customers.csv",
            "type": "customer",
            "sensitivity": "interno",
            "format": "csv",
        },
        {
            "file": "employees.csv",
            "type": "employee",
            "sensitivity": "restrito",
            "format": "csv",
        },
        {
            "file": "sales.csv",
            "type": "sale",
            "sensitivity": "interno",
            "format": "csv",
        },
        {
            "file": "products.json",
            "type": "product",
            "sensitivity": "publico",
            "format": "json",
            "records_key": "products",
        },
        {
            "file": "stores.json",
            "type": "store",
            "sensitivity": "publico",
            "format": "json",
            "records_key": "network_stores",
        },
    ]

    for config in configs:
        file_path = base_path / config["file"]

        if config["format"] == "csv":
            documents = load_csv_file(
                str(file_path),
                config["type"],
                config["sensitivity"],
            )
        else:
            documents = load_json_file(
                str(file_path),
                config["type"],
                config["sensitivity"],
                config.get("records_key"),
            )

        all_documents.extend(documents)

    return all_documents