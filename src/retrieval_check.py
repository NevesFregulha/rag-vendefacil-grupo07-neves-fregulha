"""Comparativo de busca hibrida com e sem filtros para a Etapa 2."""

from __future__ import annotations

from langchain_core.documents import Document

from src.query_analyzer import extract_filters
from src.retrieve import hybrid_search


COMPARISON_QUESTIONS = (
    "Quais tickets de Minas Gerais estao relacionados ao modulo de estoque?",
    "Quais tickets de Sao Paulo estao relacionados ao modulo de PDV?",
    "Quais tickets do Rio de Janeiro estao relacionados ao modulo pay?",
)


def _result_line(document: Document) -> str:
    metadata = document.metadata
    preview = " ".join(document.page_content.split())[:120]
    return (
        f"- {metadata.get('chunk_id')} | {metadata.get('source_file')} | "
        f"state={metadata.get('state', '-')} | module={metadata.get('module', '-')} | "
        f"{preview}"
    )


def comparison_report(vectorstore, *, k: int = 5) -> str:
    """Gera relatorio lado a lado para as tres consultas obrigatorias."""
    sections = ["# Comparativo de recuperacao da Etapa 2"]
    for question in COMPARISON_QUESTIONS:
        filters = extract_filters(question)
        without_filter = hybrid_search(vectorstore, question, k=k, filters={})
        with_filter = hybrid_search(vectorstore, question, k=k, filters=filters)
        sections.extend(
            [
                f"\n## {question}",
                f"Filtros: `{filters}`",
                "\n| Sem filtro | Com filtro |",
                "|---|---|",
            ]
        )
        for index in range(max(len(without_filter), len(with_filter))):
            left = _result_line(without_filter[index]) if index < len(without_filter) else "-"
            right = _result_line(with_filter[index]) if index < len(with_filter) else "-"
            sections.append(f"| {left} | {right} |")
    return "\n".join(sections)


def main() -> None:
    from src.vectorstore import load_vectorstore

    vectorstore = load_vectorstore()
    print(comparison_report(vectorstore))


if __name__ == "__main__":
    main()
