from collections import Counter

from src.ingest import load_all_documents


EXPECTED_DISTRIBUTION = {
    "ata": 38,
    "customer": 2000,
    "email": 43,
    "employee": 10,
    "log": 450,
    "manual": 24,
    "policy": 23,
    "product": 5,
    "sale": 3000,
    "store": 50,
    "ticket": 75,
}


def test_loads_complete_corpus():
    documents = load_all_documents()

    assert len(documents) == 5718

    distribution = Counter(
        document.metadata["doc_type"] for document in documents
    )

    assert distribution == EXPECTED_DISTRIBUTION