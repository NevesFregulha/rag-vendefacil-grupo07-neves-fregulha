from pathlib import Path

import pytest

from src.vectorstore import load_vectorstore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "index"

pytestmark = pytest.mark.skipif(
    not (INDEX_DIR / "index.faiss").exists() or not (INDEX_DIR / "index.pkl").exists(),
    reason="indice local ausente; execute python -m src.vectorstore",
)


def test_loads_persisted_faiss_index():
    vectorstore = load_vectorstore(INDEX_DIR)

    assert (INDEX_DIR / "index.faiss").exists()
    assert (INDEX_DIR / "index.pkl").exists()
    assert vectorstore.index.ntotal == 5718


def test_searches_persisted_faiss_index():
    vectorstore = load_vectorstore(INDEX_DIR)

    documents = vectorstore.similarity_search(
        "Quais produtos a VendeFácil oferece?",
        k=5,
    )

    assert len(documents) == 5
    assert all(document.page_content.strip() for document in documents)
