"""Verificações de sanidade do índice FAISS persistido."""

from __future__ import annotations

from collections import Counter

from src.vectorstore import load_vectorstore


TEST_QUESTIONS = [
    "Quais produtos a VendeFácil oferece?",
    "Quais lojas estão localizadas em Minas Gerais?",
    "Quais erros ocorreram no módulo de PDV?",
]


def main() -> None:
    """Imprime estatísticas e resultados de busca no índice persistido."""
    vectorstore = load_vectorstore()

    documents = list(vectorstore.docstore._dict.values())
    distribution = Counter(
        document.metadata["doc_type"] for document in documents
    )

    print(f"Total de chunks: {len(documents)}")
    print("\nDistribuição por doc_type:")

    for doc_type, quantity in sorted(distribution.items()):
        print(f"- {doc_type}: {quantity}")

    for question in TEST_QUESTIONS:
        print(f"\nPergunta: {question}")
        print("5 chunks mais similares:")

        results = vectorstore.similarity_search_with_score(
            question,
            k=5,
        )

        for position, (document, score) in enumerate(results, start=1):
            metadata = document.metadata
            excerpt = document.page_content.replace("\n", " ").strip()

            if len(excerpt) > 220:
                excerpt = f"{excerpt[:220]}..."

            print(
                f"{position}. score={score:.4f} | "
                f"chunk_id={metadata['chunk_id']} | "
                f"doc_type={metadata['doc_type']} | "
                f"source={metadata['source_file']}"
            )
            print(f"   {excerpt}")


if __name__ == "__main__":
    main()