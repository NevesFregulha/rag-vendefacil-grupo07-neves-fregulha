"""Testes da recuperacao densa com filtro."""

import unittest

from langchain_core.documents import Document

from src.retrieve import (
    VAGAS_SEM_FILTRO,
    bm25_search,
    dense_search,
    hybrid_search,
    matches_filters,
    reciprocal_rank_fusion,
)


class _FakeDocstore:
    def __init__(self, documents):
        self.documents = documents

    def search(self, document_id):
        return self.documents[document_id]


class _FakeVectorstore:
    def __init__(self, documents):
        self.index_to_docstore_id = {position: position for position in range(len(documents))}
        self.docstore = _FakeDocstore(documents)
        self.last_call = None

    def similarity_search(self, question, **kwargs):
        self.last_call = (question, kwargs)
        selected = list(self.docstore.documents)
        filtro = kwargs.get("filter")
        # O FAISS do LangChain aceita `Callable` alem de `dict`; o dublê precisa
        # suportar os dois para refletir a API real.
        if callable(filtro):
            selected = [doc for doc in selected if filtro(doc.metadata)]
        elif filtro:
            selected = [doc for doc in selected if matches_filters(doc, filtro)]
        return selected[: kwargs["k"]]


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="MG estoque", metadata={"doc_type": "ticket", "state": "MG", "module": "estoque"}),
            Document(page_content="SP estoque", metadata={"doc_type": "ticket", "state": "SP", "module": "estoque"}),
        ]

    def test_applies_analyzed_filter_and_fetch_k(self):
        store = _FakeVectorstore(self.documents)
        result = dense_search(store, "tickets de Minas Gerais sobre estoque", k=1, fetch_k=500)
        self.assertEqual(result[0].page_content, "MG estoque")
        self.assertEqual(
            store.last_call[1]["filter"],
            {"doc_type": "ticket", "state": "MG", "module": "estoque"},
        )
        self.assertEqual(store.last_call[1]["fetch_k"], 500)

    def test_ignores_filter_value_that_does_not_exist(self):
        store = _FakeVectorstore(self.documents)
        result = dense_search(store, "consulta", filters={"state": "AC"})
        self.assertEqual(result, [])
        self.assertIsNone(store.last_call)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="Falha generica no pagamento", metadata={"chunk_id": "dense", "state": "MG", "module": "pdv"}),
            Document(page_content="Erro E-PDV-042 no caixa", metadata={"chunk_id": "exact", "state": "MG", "module": "pdv"}),
            Document(page_content="Erro E-PDV-042 fora do estado", metadata={"chunk_id": "sp", "state": "SP", "module": "pdv"}),
        ]

    def test_bm25_finds_exact_error_code_and_prefilters(self):
        result = bm25_search(
            self.documents, "E-PDV-042", k=2, filters={"state": "MG"}
        )
        self.assertEqual(result[0].metadata["chunk_id"], "exact")
        self.assertTrue(all(doc.metadata["state"] == "MG" for doc in result))

    def test_rrf_rewards_document_present_in_both_rankings(self):
        first, second, third = self.documents
        result = reciprocal_rank_fusion([[first, second], [second, third]], limit=3)
        self.assertEqual(result[0].metadata["chunk_id"], "exact")

    def test_hybrid_search_returns_fused_ranking(self):
        store = _FakeVectorstore(self.documents)
        result = hybrid_search(store, "E-PDV-042 em MG", k=2)
        self.assertEqual(
            {doc.metadata["chunk_id"] for doc in result}, {"dense", "exact"}
        )
        self.assertTrue(all(doc.metadata["state"] == "MG" for doc in result))


class BuscaAbertaComplementaFiltroInferidoTests(unittest.TestCase):
    """Regressao: filtro inferido eliminava a fonte certa (Q04, Q12, Q19).

    Medido no benchmark: nessas tres perguntas o documento correto aparece na
    busca sem filtro e some na filtrada. O filtro acerta a forma e erra a
    intencao - "politica de home office para a equipe de Engenharia" vira
    doc_type=employee, e a resposta esta em home_office.md.
    """

    def setUp(self):
        self.documents = [
            Document(
                page_content="Cadastro do funcionario da equipe de engenharia.",
                metadata={"chunk_id": "employee", "doc_type": "employee"},
            ),
            Document(
                page_content="Politica de home office: modelo remoto first.",
                metadata={"chunk_id": "politica", "doc_type": "policy"},
            ),
        ]

    def test_traz_fonte_que_o_filtro_inferido_eliminaria(self):
        store = _FakeVectorstore(self.documents)

        resultado = hybrid_search(
            store, "politica de home office do funcionario", k=5, complementar_sem_filtro=True
        )
        ids = {d.metadata["chunk_id"] for d in resultado}

        self.assertIn("employee", ids, "o resultado filtrado continua presente")
        self.assertIn("politica", ids, "a busca aberta preenche a vaga que sobrou")

    def test_filtro_explicito_nao_recebe_complemento(self):
        store = _FakeVectorstore(self.documents)

        resultado = hybrid_search(
            store,
            "qualquer",
            k=5,
            filters={"doc_type": "employee"},
            complementar_sem_filtro=True,
        )
        ids = {d.metadata["chunk_id"] for d in resultado}

        self.assertEqual(ids, {"employee"}, "filtro pedido pelo chamador nao e complementado")

    def test_por_padrao_o_resultado_continua_respeitando_o_filtro(self):
        """Garantia da Etapa 2: sem opt-in, todo resultado satisfaz o filtro."""
        store = _FakeVectorstore(self.documents)

        resultado = hybrid_search(store, "politica de home office do funcionario", k=5)
        ids = {d.metadata["chunk_id"] for d in resultado}

        self.assertEqual(ids, {"employee"})


class RelaxamentoDeFiltroImpossivelTests(unittest.TestCase):
    """Regressao: conjuncao valida mas impossivel devolvia zero documentos.

    Detectado no benchmark da Etapa 4 (Q02, Q07, Q11, Q13). No corpus real
    somente `log` e `ticket` carregam `module`, entao qualquer combinacao de
    `module` com outro doc_type nunca casa - e a busca falhava por completo em
    vez de relaxar o filtro menos confiavel.
    """

    def setUp(self):
        self.documents = [
            Document(
                page_content="VendeFacil Estoque: gestao de inventario e entrada de NF-e.",
                metadata={"chunk_id": "produto", "doc_type": "product", "state": "MG"},
            ),
            Document(
                page_content="Ticket sobre sincronizacao de estoque na filial.",
                metadata={
                    "chunk_id": "ticket",
                    "doc_type": "ticket",
                    "module": "estoque",
                    "state": "MG",
                },
            ),
        ]

    def test_relax_filters_descarta_doc_type_antes_de_module(self):
        from src.retrieve import relax_filters

        resolvido, descartados = relax_filters(
            self.documents, {"doc_type": "product", "module": "estoque"}
        )

        self.assertEqual(descartados, ["doc_type"])
        self.assertEqual(resolvido, {"module": "estoque"})

    def test_relax_filters_preserva_conjuncao_que_ja_casa(self):
        from src.retrieve import relax_filters

        resolvido, descartados = relax_filters(
            self.documents, {"doc_type": "ticket", "module": "estoque"}
        )

        self.assertEqual(descartados, [])
        self.assertEqual(resolvido, {"doc_type": "ticket", "module": "estoque"})

    def test_hybrid_search_nao_devolve_vazio_para_conjuncao_impossivel(self):
        store = _FakeVectorstore(self.documents)

        resultado = hybrid_search(store, "produtos do modulo de estoque", k=2)

        self.assertTrue(resultado, "a busca nao pode devolver lista vazia por filtro impossivel")

    def test_filtro_explicito_do_chamador_nao_e_relaxado(self):
        store = _FakeVectorstore(self.documents)

        resultado = hybrid_search(
            store, "qualquer pergunta", k=2, filters={"doc_type": "product", "module": "estoque"}
        )

        self.assertEqual(resultado, [])


if __name__ == "__main__":
    unittest.main()
