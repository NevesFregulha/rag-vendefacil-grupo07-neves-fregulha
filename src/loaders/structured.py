import json
import csv
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

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            # Tenta pegar um ID real do registro, mas usa o índice para garantir unicidade
            raw_id = (
                row.get("customer_id")
                or row.get("employee_id")
                or row.get("sale_id")
                or row.get("id")
                or f"row-{index + 1}"
            )
            
            content_parts = [f"{k}: {v}" for k, v in row.items() if v]
            page_content = f"Registro do tipo {doc_type} ({path.name}): " + ", ".join(content_parts)

            # Garante que o chunk_id combine o nome do arquivo, o ID e a linha para evitar duplicatas
            chunk_id = f"{path.stem}-{raw_id}-{index + 1}"

            metadata = {
                "source_file": path.name,
                "doc_type": doc_type,
                "chunk_id": chunk_id,
                "sensitivity": sensitivity,
            }

            for key in ["customer_id", "state", "module", "status", "date", "employee_id", "sale_id"]:
                if key in row:
                    metadata[key] = row[key]

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def load_json_file(
    file_path: str,
    doc_type: str,
    sensitivity: str,
) -> list[Document]:
    """Lê um arquivo JSON contendo uma lista de objetos e transforma cada um em um Document."""
    documents = []
    path = Path(file_path)

    if not path.exists():
        return documents

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]

    for index, row in enumerate(records):
        raw_id = (
            row.get("product_id")
            or row.get("store_id")
            or row.get("id")
            or f"item-{index + 1}"
        )

        content_parts = [f"{k}: {v}" for k, v in row.items() if v]
        page_content = f"Registro do tipo {doc_type} ({path.name}): " + ", ".join(content_parts)

        # Garante unicidade combinando nome do arquivo, ID e índice
        chunk_id = f"{path.stem}-{raw_id}-{index + 1}"

        metadata = {
            "source_file": path.name,
            "doc_type": doc_type,
            "chunk_id": chunk_id,
            "sensitivity": sensitivity,
        }

        for key in ["product_id", "store_id", "category", "city"]:
            if key in row:
                metadata[key] = str(row[key])

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def load_structured_documents(data_dir: str = "data/structured") -> list[Document]:
    """Carrega todos os arquivos estruturados (CSV e JSON) do diretório informado."""
    base_path = Path(data_dir)
    all_documents = []

    configs = [
        {"file": "customers.csv", "type": "customer", "sensitivity": "interno", "format": "csv"},
        {"file": "employees.csv", "type": "employee", "sensitivity": "restrito", "format": "csv"},
        {"file": "sales.csv", "type": "sale", "sensitivity": "interno", "format": "csv"},
        {"file": "products.json", "type": "product", "sensitivity": "publico", "format": "json"},
        {"file": "stores.json", "type": "store", "sensitivity": "publico", "format": "json"},
    ]

    for cfg in configs:
        file_path = base_path / cfg["file"]
        if cfg["format"] == "csv":
            docs = load_csv_file(str(file_path), cfg["type"], cfg["sensitivity"])
        else:
            docs = load_json_file(str(file_path), cfg["type"], cfg["sensitivity"])
        
        all_documents.extend(docs)

    return all_documents