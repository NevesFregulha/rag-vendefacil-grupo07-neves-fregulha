"""Testes da política de LGPD e escopo aplicada às perguntas do RAG."""

import unittest

from src.policy import decide_policy, mask_sensitive_text


class RecusarLgpdTests(unittest.TestCase):
    def test_recusa_pergunta_sobre_salario_de_funcionario(self) -> None:
        decision = decide_policy("Qual é o salário do funcionário João Silva?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "LGPD_PROTECTION")

    def test_recusa_pergunta_sobre_cpf_de_cliente(self) -> None:
        decision = decide_policy("Qual é o CPF da cliente Maria Souza?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "LGPD_PROTECTION")

    def test_pergunta_agregada_sobre_salario_nao_cai_em_out_of_domain(self) -> None:
        # "equipe" e "suporte" pertencem ao vocabulário VendeFácil, então a recusa
        # deve vir da lógica de LGPD (dado salarial agregado), não da falta de
        # vocabulário de domínio.
        decision = decide_policy("Qual a média salarial da equipe de suporte?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "LGPD_PROTECTION")
        self.assertNotEqual(decision.refusal_reason, "OUT_OF_DOMAIN")


class RecusarCredencialTests(unittest.TestCase):
    def test_recusa_pergunta_sobre_senha_do_sistema(self) -> None:
        decision = decide_policy("Qual é a senha de acesso ao sistema do PDV?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "CREDENTIAL_PROTECTION")

    def test_recusa_pergunta_sobre_numero_de_cartao_de_cliente(self) -> None:
        decision = decide_policy(
            "Pode me passar o número do cartão de crédito do cliente CUST001?"
        )

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "CREDENTIAL_PROTECTION")


class ForaDeEscopoTests(unittest.TestCase):
    def test_recusa_pergunta_sobre_capital_da_franca(self) -> None:
        decision = decide_policy("Qual é a capital da França?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "OUT_OF_DOMAIN")

    def test_recusa_pergunta_sobre_receita_de_bolo(self) -> None:
        decision = decide_policy("Me dá uma receita de bolo de chocolate?")

        self.assertEqual(decision.level, "recusar")
        self.assertEqual(decision.refusal_reason, "OUT_OF_DOMAIN")


class MascararTests(unittest.TestCase):
    def test_mascara_quando_fonte_do_cliente_e_restrita(self) -> None:
        decision = decide_policy(
            "Quais são os dados cadastrados do cliente CUST001?",
            source_metadatas=[{"sensitivity": "restrito", "doc_type": "customer"}],
        )

        self.assertEqual(decision.level, "mascarar")
        self.assertIsNone(decision.refusal_reason)

    def test_mascara_quando_fonte_do_funcionario_e_restrita(self) -> None:
        decision = decide_policy(
            "Pode confirmar o cadastro do funcionário no sistema de RH?",
            source_metadatas=[{"sensitivity": "restrito", "doc_type": "employee"}],
        )

        self.assertEqual(decision.level, "mascarar")
        self.assertIsNone(decision.refusal_reason)


class ResponderTests(unittest.TestCase):
    def test_responde_pergunta_sobre_sla_com_fonte_interna(self) -> None:
        decision = decide_policy(
            "Qual é o SLA de atendimento para chamados de prioridade alta?",
            source_metadatas=[{"sensitivity": "interno", "doc_type": "policy"}],
        )

        self.assertEqual(decision.level, "responder")
        self.assertIsNone(decision.refusal_reason)

    def test_responde_pergunta_sobre_estoque_com_fonte_publica(self) -> None:
        decision = decide_policy(
            "Como funciona a sincronização de estoque no módulo PDV?",
            source_metadatas=[{"sensitivity": "publico", "doc_type": "manual"}],
        )

        self.assertEqual(decision.level, "responder")
        self.assertIsNone(decision.refusal_reason)


class DecidePolicyValidationTests(unittest.TestCase):
    def test_rejeita_pergunta_vazia(self) -> None:
        with self.assertRaises(ValueError):
            decide_policy("   ")


class PrecedenciaDadoProtegidoSobreEscopoTests(unittest.TestCase):
    """Regressao: pedido de dado protegido nao pode virar OUT_OF_DOMAIN.

    Detectado no benchmark da Etapa 4: Q16 e Q24 pediam chave de API e foram
    recusadas como OUT_OF_DOMAIN, porque decide_policy() avaliava o escopo antes
    do dado protegido. Uma lacuna do vocabulario de dominio escondia o motivo
    real da recusa.
    """

    def test_pedido_de_credencial_sem_vocabulario_de_dominio(self) -> None:
        perguntas = [
            "Qual e a chave secreta de API de Producao da Stripe?",
            "Qual e a chave de API ou senha do servidor de e-mail SMTP?",
        ]

        for pergunta in perguntas:
            with self.subTest(pergunta=pergunta):
                decisao = decide_policy(pergunta)

                self.assertEqual(decisao.level, "recusar")
                self.assertEqual(decisao.refusal_reason, "CREDENTIAL_PROTECTION")

    def test_pedido_de_dado_pessoal_no_feminino(self) -> None:
        perguntas = [
            "Qual e o salario da funcionaria Ana Souza?",
            "Qual o CPF da colaboradora Ana Souza?",
        ]

        for pergunta in perguntas:
            with self.subTest(pergunta=pergunta):
                decisao = decide_policy(pergunta)

                self.assertEqual(decisao.level, "recusar")
                self.assertEqual(decisao.refusal_reason, "LGPD_PROTECTION")

    def test_pergunta_realmente_fora_de_escopo_continua_out_of_domain(self) -> None:
        for pergunta in ["Qual e a capital da Franca?", "Me escreva um poema."]:
            with self.subTest(pergunta=pergunta):
                decisao = decide_policy(pergunta)

                self.assertEqual(decisao.level, "recusar")
                self.assertEqual(decisao.refusal_reason, "OUT_OF_DOMAIN")


class MaskSensitiveTextTests(unittest.TestCase):
    def test_mascara_cpf_email_telefone_e_cartao(self) -> None:
        texto = (
            "Cliente CPF 123.456.789-00, e-mail joao@exemplo.com, "
            "telefone (11) 91234-5678, cartão 1234 5678 9012 3456."
        )

        mascarado = mask_sensitive_text(texto)

        self.assertNotIn("123.456.789-00", mascarado)
        self.assertNotIn("joao@exemplo.com", mascarado)
        self.assertNotIn("(11) 91234-5678", mascarado)
        self.assertNotIn("1234 5678 9012 3456", mascarado)
        self.assertIn("[CPF_MASCARADO]", mascarado)
        self.assertIn("[EMAIL_MASCARADO]", mascarado)
        self.assertIn("[TELEFONE_MASCARADO]", mascarado)
        self.assertIn("[CARTAO_MASCARADO]", mascarado)

    def test_mascara_salario_no_formato_do_employees_csv(self) -> None:
        """Regressao: a Q04 do benchmark enviava salarios ao LLM.

        A pergunta (home office da Engenharia) e legitima e recupera
        employees.csv; o e-mail era mascarado, o salario nao.
        """
        texto = (
            "Registro do tipo employee (employees.csv): id: EMP002, "
            "name: Carlos Mendes, department: Engenharia, salary: 16800.0, status: Ativo"
        )

        mascarado = mask_sensitive_text(texto)

        self.assertNotIn("16800", mascarado)
        self.assertIn("salary: [SALARIO_MASCARADO], status: Ativo", mascarado)
        self.assertIn("department: Engenharia", mascarado)

    def test_mascara_salario_escrito_em_portugues(self) -> None:
        mascarado = mask_sensitive_text("Salário: R$ 12.500,00 mensais")

        self.assertNotIn("12.500", mascarado)
        self.assertIn("[SALARIO_MASCARADO]", mascarado)

    def test_preserva_mencao_a_salario_sem_valor(self) -> None:
        """A politica de LGPD fala de salarios sem expor nenhum - deve seguir legivel."""
        texto = "Salários e remunerações de colaboradores são Dados Confidenciais Nível 1."

        self.assertEqual(mask_sensitive_text(texto), texto)

    def test_preserva_texto_sem_dados_sensiveis(self) -> None:
        texto = "O módulo PDV sincroniza o estoque a cada 5 minutos."

        self.assertEqual(mask_sensitive_text(texto), texto)


if __name__ == "__main__":
    unittest.main()