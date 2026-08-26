"""Contrato e validação dos metadados usados na ingestão."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REQUIRED_METADATA = frozenset(
    {"source_file", "doc_type", "chunk_id", "sensitivity"}
)

ALLOWED_SENSITIVITIES = frozenset({"publico", "interno", "restrito"})


class MetadataValidationError(ValueError):
    """Indica que um ou mais documentos possuem metadados inválidos."""


def build_metadata(
    *,
    source_file: str | Path,
    doc_type: str,
    chunk_id: str,
    sensitivity: str,
    **additional_fields: Any,
) -> dict[str, Any]:
    """Cria metadados no formato compartilhado por todos os loaders."""
    metadata: dict[str, Any] = {
        "source_file": Path(source_file).name,
        "doc_type": doc_type.strip(),
        "chunk_id": chunk_id.strip(),
        "sensitivity": sensitivity.strip().lower(),
    }
    metadata.update(
        {key: value for key, value in additional_fields.items() if value is not None}
    )
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Valida os campos obrigatórios de um único chunk."""
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        fields = ", ".join(sorted(missing))
        raise MetadataValidationError(f"Metadados obrigatórios ausentes: {fields}.")

    empty = [field for field in REQUIRED_METADATA if not str(metadata[field]).strip()]
    if empty:
        fields = ", ".join(sorted(empty))
        raise MetadataValidationError(f"Metadados obrigatórios vazios: {fields}.")

    sensitivity = str(metadata["sensitivity"]).strip().lower()
    if sensitivity not in ALLOWED_SENSITIVITIES:
        allowed = ", ".join(sorted(ALLOWED_SENSITIVITIES))
        raise MetadataValidationError(
            f"Sensibilidade inválida: {metadata['sensitivity']!r}. "
            f"Valores permitidos: {allowed}."
        )


def validate_documents(documents: Iterable[Any]) -> None:
    """Valida os metadados e a unicidade dos IDs de uma coleção de Documents."""
    seen_chunk_ids: set[str] = set()

    for position, document in enumerate(documents):
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise MetadataValidationError(
                f"Documento na posição {position} não possui metadata válido."
            )

        try:
            validate_metadata(metadata)
        except MetadataValidationError as error:
            raise MetadataValidationError(
                f"Documento na posição {position}: {error}"
            ) from error

        chunk_id = str(metadata["chunk_id"]).strip()
        if chunk_id in seen_chunk_ids:
            raise MetadataValidationError(f"chunk_id duplicado: {chunk_id!r}.")
        seen_chunk_ids.add(chunk_id)
