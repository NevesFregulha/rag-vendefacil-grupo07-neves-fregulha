"""Política de segurança/LGPD e escopo para respostas do assistente RAG.

Regras derivadas de `data/unstructured/policies/seguranca_lgpd.md` (itens 1.2-1.4):
perguntas sobre salário, CPF ou dados pessoais sensíveis de colaboradores/clientes
são recusadas (LGPD_PROTECTION); perguntas sobre senha, cartão ou credenciais são
recusadas (CREDENTIAL_PROTECTION); perguntas fora do domínio da VendeFácil também
são recusadas (OUT_OF_DOMAIN). Quando a pergunta é legítima mas alguma fonte
recuperada tem sensitivity="restrito" (ver `src/metadata.py`), a resposta é
permitida, mas os dados pessoais presentes no texto são mascarados antes de virar
evidência/resposta.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

PolicyLevel = Literal["recusar", "mascarar", "responder"]
RefusalReason = Literal["LGPD_PROTECTION", "OUT_OF_DOMAIN", "CREDENTIAL_PROTECTION"]


@dataclass(frozen=True)
class PolicyDecision:
    """Resultado da avaliação de política para uma pergunta."""

    level: PolicyLevel
    refusal_reason: RefusalReason | None = None
    reason: str = ""


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize(text: str) -> str:
    return _strip_accents(text.casefold())


def _contains_any(normalized_text: str, keywords: Iterable[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized_text) for keyword in keywords
    )


def _matches_any(normalized_text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, normalized_text) for pattern in patterns)


# Pedidos de credenciais/dados financeiros sensíveis (política item 1.2 e 1.4).
CREDENTIAL_KEYWORDS = (
    "senha",
    "senhas",
    "password",
    "cvv",
    "numero do cartao",
    "numero de cartao",
    "cartao de credito",
    "cartao de debito",
    "dados do cartao",
    "chave de api",
    "api key",
    "chave de acesso",
    "token de acesso",
    "token de api",
    "dados bancarios",
    "conta bancaria",
    "conta corrente",
)

# Alguns pedidos de credencial nao casam por palavra-chave fixa porque admitem
# adjetivos no meio ("chave SECRETA de API de Producao"). Estes padroes cobrem
# essas variacoes sem exigir a frase exata.
CREDENTIAL_PATTERNS = (
    r"\bchaves?\s+(?:\w+\s+){0,3}de\s+(?:api|acesso|producao|criptografia)\b",
    r"\bchaves?\s+(?:secreta|privada|publica)s?\b",
    r"\bsegredos?\s+(?:jwt|de\s+\w+)\b",
    r"\bjwt\b",
    r"\bapi[_\s-]?key\b",
    r"\bsecret\s+key\b",
    r"\bcredencia(?:l|is)\b",
)

# Dados pessoais confidenciais de colaboradores/clientes (política item 1.3 e 1.4).
LGPD_KEYWORDS = (
    "salario",
    "salarios",
    "salarial",
    "salariais",
    "remuneracao",
    "remuneracoes",
    "cpf",
    "endereco residencial",
    "endereco pessoal",
    "onde mora",
    "dados de saude",
    "condicao de saude",
    "biometria",
    "biometrico",
    "religiao",
)

# Vocabulário do domínio VendeFácil (módulos, entidades e documentos do corpus).
DOMAIN_KEYWORDS = (
    "vendefacil",
    "venda",
    "vendas",
    "cliente",
    "clientes",
    "produto",
    "produtos",
    "loja",
    "lojas",
    "estoque",
    "pedido",
    "pedidos",
    "ticket",
    "tickets",
    "chamado",
    "chamados",
    "pdv",
    "caixa",
    "frente de caixa",
    "ecommerce",
    "e-commerce",
    "catalogo",
    "pagamento",
    "pagamentos",
    "pix",
    "tef",
    "maquininha",
    "nota fiscal",
    "nf-e",
    "nfe",
    "funcionario",
    "funcionarios",
    "funcionaria",
    "funcionarias",
    "colaborador",
    "colaboradores",
    "colaboradora",
    "colaboradoras",
    "gerente",
    "gerentes",
    "diretor",
    "diretora",
    "vendedor",
    "vendedora",
    "equipe",
    "equipes",
    "suporte",
    "rh",
    "recursos humanos",
    "politica",
    "politicas",
    "manual",
    "relatorio",
    "relatorios",
    "dashboard",
    "indicador",
    "indicadores",
    "sla",
    "atendimento",
    "reembolso",
    "home office",
    "beneficio",
    "beneficios",
    "ferias",
    "codigo de conduta",
    "reuniao",
    "reunioes",
    "ata",
    "atas",
    "sistema",
    "modulo",
    "modulos",
    "integracao",
    "integracoes",
    "frete",
    "logistica",
    "inventario",
    "analytics",
    "compra",
    "compras",
    "fornecedor",
    "fornecedores",
)


def is_out_of_scope(question: str) -> bool:
    """Indica se a pergunta não toca em nenhum tema do domínio VendeFácil."""
    return not _contains_any(_normalize(question), DOMAIN_KEYWORDS)


def classify_restricted_request(question: str) -> RefusalReason | None:
    """Detecta se a própria pergunta solicita dado protegido pela política de LGPD."""
    normalized = _normalize(question)
    if _contains_any(normalized, CREDENTIAL_KEYWORDS) or _matches_any(
        normalized, CREDENTIAL_PATTERNS
    ):
        return "CREDENTIAL_PROTECTION"
    if _contains_any(normalized, LGPD_KEYWORDS):
        return "LGPD_PROTECTION"
    return None


def has_restricted_sources(metadatas: Iterable[Mapping] | None) -> bool:
    """Verifica se alguma fonte recuperada tem sensitivity='restrito'."""
    if not metadatas:
        return False
    return any(
        str(metadata.get("sensitivity", "")).strip().lower() == "restrito"
        for metadata in metadatas
        if isinstance(metadata, Mapping)
    )


def decide_policy(
    question: str,
    source_metadatas: Iterable[Mapping] | None = None,
) -> PolicyDecision:
    """Decide entre recusar, mascarar ou responder, conforme a política de LGPD e escopo."""
    if not question or not question.strip():
        raise ValueError("Pergunta vazia.")

    # A checagem de dado protegido vem ANTES da de escopo: um pedido de senha,
    # chave de API ou salário é um pedido de dado protegido mesmo que nenhuma
    # palavra do domínio apareça na pergunta. Na ordem inversa, uma lacuna do
    # vocabulário rebaixava a recusa para OUT_OF_DOMAIN e escondia o motivo real
    # (ex.: "Qual a chave secreta de API de Produção da Stripe?").
    restricted_reason = classify_restricted_request(question)
    if restricted_reason is not None:
        return PolicyDecision(
            "recusar",
            restricted_reason,
            "Pergunta solicita diretamente dado protegido pela política de LGPD.",
        )

    if is_out_of_scope(question):
        return PolicyDecision(
            "recusar",
            "OUT_OF_DOMAIN",
            "Pergunta não relacionada a nenhum tema do domínio VendeFácil.",
        )

    if has_restricted_sources(source_metadatas):
        return PolicyDecision(
            "mascarar",
            None,
            "Fontes recuperadas têm sensitivity='restrito'; dados pessoais serão mascarados.",
        )

    return PolicyDecision("responder", None, "Pergunta e fontes dentro da política.")


_CPF_PATTERN = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
_CVV_PATTERN = re.compile(r"\bcvv\s*:?\s*\d{3,4}\b", flags=re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_PATTERN = re.compile(r"\(\d{2}\)\s?\d{4,5}-\d{4}")
# Salario e dado Nivel 1 (politica item 1.3), mas nao tem formato proprio como
# CPF ou e-mail: e so um numero. Por isso o padrao exige o rotulo do campo e
# preserva o rotulo, escondendo apenas o valor. Sem esta regra, uma pergunta
# legitima que recuperava employees.csv (ex.: home office da Engenharia) enviava
# os salarios ao LLM, que chegou a copia-los numa citacao.
_SALARY_PATTERN = re.compile(
    # O valor aceita "." e "," so ENTRE digitos (16800.0, 12.500,00): uma versao
    # com [\d.,]* engolia a virgula que separa os campos do registro.
    r"\b(salary|sal[aá]rio|remunera[cç][aã]o)(\s*[:=]\s*)(?:R\$\s*)?\d+(?:[.,]\d+)*",
    flags=re.IGNORECASE,
)


def mask_sensitive_text(text: str) -> str:
    """Ofusca CPF, cartão, CVV, e-mail, telefone e salário presentes em um trecho de texto."""
    masked = _CPF_PATTERN.sub("[CPF_MASCARADO]", text)
    masked = _CARD_PATTERN.sub("[CARTAO_MASCARADO]", masked)
    masked = _CVV_PATTERN.sub("[CVV_MASCARADO]", masked)
    masked = _EMAIL_PATTERN.sub("[EMAIL_MASCARADO]", masked)
    masked = _PHONE_PATTERN.sub("[TELEFONE_MASCARADO]", masked)
    masked = _SALARY_PATTERN.sub(r"\1\2[SALARIO_MASCARADO]", masked)
    return masked