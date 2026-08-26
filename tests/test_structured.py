"""Testes da ingestão estruturada, reservados para a TASK 02."""
import pytest
from src.loaders.structured import load_structured_documents, load_csv_file, load_json_file

def test_load_structured_returns_list():
    docs = load_structured_documents()
    assert isinstance(docs, list)

def test_documents_are_not_empty():
    docs = load_structured_documents()
    if docs: 
        for doc in docs:
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content.strip()) > 0

def test_mandatory_metadata():
    docs = load_structured_documents()
    mandatory_keys = {"source_file", "doc_type", "chunk_id", "sensitivity"}
    
    for doc in docs:
        for key in mandatory_keys:
            assert key in doc.metadata, f"Metadado obrigatório '{key}' ausente."

def test_unique_chunk_ids():
    docs = load_structured_documents()
    chunk_ids = [doc.metadata["chunk_id"] for doc in docs]
    
    # Verifica se não há IDs duplicados
    assert len(chunk_ids) == len(set(chunk_ids)), "Existem chunk_ids duplicados!"

def test_sensitivity_values():
    docs = load_structured_documents()
    valid_sensitivities = {"publico", "interno", "restrito"}
    
    for doc in docs:
        assert doc.metadata["sensitivity"] in valid_sensitivities