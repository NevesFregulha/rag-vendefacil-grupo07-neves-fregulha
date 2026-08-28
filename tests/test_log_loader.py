from pathlib import Path

from src.loaders.log_loader import load_log_documents
from src.metadata import validate_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_FILE = PROJECT_ROOT / "data" / "semi_structured" / "system_logs.csv"


def test_loads_all_system_logs():
    documents = load_log_documents(LOGS_FILE)

    assert len(documents) == 450
    assert {doc.metadata["doc_type"] for doc in documents} == {"log"}

    validate_documents(documents)


def test_logs_include_filterable_metadata():
    documents = load_log_documents(LOGS_FILE)

    required_fields = {
        "source_file",
        "doc_type",
        "chunk_id",
        "sensitivity",
        "timestamp",
        "date",
        "level",
        "service",
        "module",
        "customer_id",
        "event",
        "error_code",
    }

    for document in documents:
        assert required_fields.issubset(document.metadata)