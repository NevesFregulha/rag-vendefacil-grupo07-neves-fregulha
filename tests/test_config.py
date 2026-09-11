"""Testes da selecao de provedor de LLM em config.py.

Nenhum destes testes faz chamada de rede: verificam apenas qual cliente e
construido, com quais parametros, e que a ausencia de chave falha de forma
explicita em vez de silenciosa.
"""

import importlib
import unittest
from unittest.mock import patch


def recarregar_config(**variaveis):
    """Recarrega config.py com um ambiente controlado.

    O modulo le as variaveis no momento do import, entao trocar de provedor
    exige recarregar - e nao apenas alterar os.environ.
    """
    import config

    with patch.dict("os.environ", variaveis, clear=True), patch(
        "dotenv.load_dotenv", lambda *a, **k: False
    ):
        return importlib.reload(config)


class SelecaoDeProvedorTests(unittest.TestCase):
    def tearDown(self):
        # devolve o modulo ao estado do ambiente real, para nao contaminar
        # outros testes que importem config
        import config

        importlib.reload(config)

    def test_openrouter_e_o_padrao_quando_nao_ha_provedor_definido(self):
        cfg = recarregar_config(OPENROUTER_API_KEY="x")

        self.assertEqual(cfg.LLM_PROVIDER, "openrouter")
        self.assertEqual(cfg.LLM_MODEL, "nex-agi/nex-n2.5-pro:free")

    def test_groq_continua_disponivel_quando_pedido(self):
        cfg = recarregar_config(LLM_PROVIDER="groq", GROQ_API_KEY="x")

        self.assertEqual(cfg.LLM_MODEL, "openai/gpt-oss-120b")
        self.assertEqual(type(cfg.get_llm()).__name__, "ChatGroq")

    def test_openrouter_usa_cliente_compativel_com_openai(self):
        cfg = recarregar_config(
            LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="chave-de-teste"
        )

        self.assertEqual(cfg.LLM_MODEL, "nex-agi/nex-n2.5-pro:free")

        llm = cfg.get_llm()

        self.assertEqual(llm.model_name, "nex-agi/nex-n2.5-pro:free")
        self.assertIn("openrouter.ai", str(llm.openai_api_base))

    def test_openrouter_aceita_modelo_e_base_url_customizados(self):
        cfg = recarregar_config(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="chave-de-teste",
            LLM_MODEL="google/gemma-4-31b-it:free",
            OPENROUTER_BASE_URL="https://exemplo.invalido/v1",
        )

        llm = cfg.get_llm()

        self.assertEqual(llm.model_name, "google/gemma-4-31b-it:free")
        self.assertIn("exemplo.invalido", str(llm.openai_api_base))

    def test_openrouter_sem_chave_falha_com_mensagem_util(self):
        cfg = recarregar_config(LLM_PROVIDER="openrouter")

        with self.assertRaises(RuntimeError) as erro:
            cfg.get_llm()

        self.assertIn("OPENROUTER_API_KEY", str(erro.exception))

    def test_provedor_desconhecido_falha_explicitamente(self):
        cfg = recarregar_config(LLM_PROVIDER="inexistente")

        with self.assertRaises(ValueError) as erro:
            cfg.get_llm()

        self.assertIn("openrouter", str(erro.exception))

    def test_groq_sem_chave_falha_com_mensagem_util(self):
        cfg = recarregar_config(LLM_PROVIDER="groq")

        with self.assertRaises(RuntimeError) as erro:
            cfg.get_llm()

        self.assertIn("GROQ_API_KEY", str(erro.exception))


if __name__ == "__main__":
    unittest.main()
