"""Testes do contrato compartilhado de metadados."""

import unittest
from types import SimpleNamespace

from src.metadata import (
    MetadataValidationError,
    build_metadata,
    validate_documents,
)


class MetadataTests(unittest.TestCase):
    def test_build_metadata_normalizes_shared_fields(self) -> None:
        metadata = build_metadata(
            source_file="data/structured/customers.csv",
            doc_type="customer",
            chunk_id="customers-CUST001",
            sensitivity="INTERNO",
            state="MG",
        )

        self.assertEqual(metadata["source_file"], "customers.csv")
        self.assertEqual(metadata["sensitivity"], "interno")
        self.assertEqual(metadata["state"], "MG")

    def test_rejects_missing_required_field(self) -> None:
        document = SimpleNamespace(
            metadata={
                "source_file": "customers.csv",
                "doc_type": "customer",
                "chunk_id": "customers-CUST001",
            }
        )

        with self.assertRaisesRegex(MetadataValidationError, "sensitivity"):
            validate_documents([document])

    def test_rejects_invalid_sensitivity(self) -> None:
        document = SimpleNamespace(
            metadata={
                "source_file": "customers.csv",
                "doc_type": "customer",
                "chunk_id": "customers-CUST001",
                "sensitivity": "secreto",
            }
        )

        with self.assertRaisesRegex(MetadataValidationError, "Sensibilidade inválida"):
            validate_documents([document])

    def test_rejects_duplicate_chunk_id(self) -> None:
        metadata = build_metadata(
            source_file="customers.csv",
            doc_type="customer",
            chunk_id="customers-CUST001",
            sensitivity="interno",
        )
        documents = [
            SimpleNamespace(metadata=metadata),
            SimpleNamespace(metadata=metadata.copy()),
        ]

        with self.assertRaisesRegex(MetadataValidationError, "chunk_id duplicado"):
            validate_documents(documents)


if __name__ == "__main__":
    unittest.main()
