"""Esquema Pydantic da resposta estruturada do assistente RAG VendeFácil.

Baseado em `starter/schema.py`, com três reforços exigidos pela TASK 01 da Etapa 3:
os campos fechados por `Literal`, o `chunk_id` obrigatório em cada evidência (para
rastrear a fonte até o chunk exato do índice) e um validador que impede respostas
estruturalmente inconsistentes (ex.: recusa com fontes, ou resposta sem evidência).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DocType = Literal[
    "ata",
    "customer",
    "email",
    "employee",
    "log",
    "manual",
    "policy",
    "product",
    "sale",
    "store",
    "ticket",
]

ConfidenceLevel = Literal["Alta", "Média", "Baixa", "Recusado"]

RefusalReason = Literal[
    "LGPD_PROTECTION",
    "OUT_OF_DOMAIN",
    "CREDENTIAL_PROTECTION",
    # Previsto no schema do guia e inicialmente omitido por nos. Sem ele, uma
    # pergunta legitima cuja resposta nao foi recuperada era rotulada como
    # OUT_OF_DOMAIN - factualmente falso, e confundia falha de recuperacao com
    # pergunta fora de escopo no diagnostico.
    "SEM_EVIDENCIA",
]

CITATION_MAX_LENGTH = 400


class SourceEvidence(BaseModel):
    """Trecho de evidência extraído das fontes recuperadas pelo RAG."""

    filepath: str = Field(
        min_length=1,
        description="Caminho relativo do arquivo de onde a informação foi extraída (ex: data/semi_structured/tickets.jsonl)",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Identificador único e estável do chunk usado como evidência (igual ao `chunk_id` dos metadados do índice).",
    )
    quotation: str = Field(
        min_length=1,
        description="Trecho exato do texto ou dado utilizado para fundamentar a afirmação (até "
        f"{CITATION_MAX_LENGTH} caracteres; trechos maiores são truncados).",
    )
    doc_type: Optional[DocType] = Field(
        default=None,
        description="Tipo de documento de origem do chunk.",
    )

    @field_validator("quotation")
    @classmethod
    def quotation_limpa_e_truncada(cls, value: str) -> str:
        """Normaliza a citação, truncando em vez de rejeitar trechos longos.

        O limite é aplicado aqui, e não como `max_length` no `Field`, de
        propósito: `max_length` entraria no JSON Schema enviado ao provedor, e
        provedores que validam a ferramenta no servidor (Groq) rejeitam a
        chamada inteira com HTTP 400 quando o modelo cita um trecho longo. Uma
        regra de apresentação derrubava a resposta toda. Truncando no cliente, o
        objetivo do limite é preservado sem transformar verbosidade do modelo em
        falha; o `chunk_id` continua permitindo recuperar o trecho completo.

        O truncamento não acrescenta reticências porque `_validate_evidence()`
        exige que a citação seja subtrecho literal do chunk de origem - qualquer
        caractere extra invalidaria essa verificação.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("quotation não pode ser vazia ou conter apenas espaços.")
        return stripped[:CITATION_MAX_LENGTH]


class RAGResponse(BaseModel):
    """Resposta estruturada final produzida pelo assistente VendeFácil RAG."""

    answer: str = Field(
        min_length=1,
        description="Resposta em linguagem natural, clara, objetiva e estritamente fundamentada no contexto recuperado.",
    )
    confidence_level: ConfidenceLevel = Field(
        description="Nível de confiança da resposta com base nas evidências encontradas."
    )
    sources_used: List[SourceEvidence] = Field(
        default_factory=list,
        description="Lista de fontes e trechos específicos que comprovam a resposta gerada.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Breve explicação do raciocínio lógico utilizado para construir a resposta a partir do contexto.",
    )
    is_refusal: bool = Field(
        default=False,
        description="True se o assistente recusou responder a pergunta (LGPD, segurança ou fora do escopo).",
    )
    refusal_reason: Optional[RefusalReason] = Field(
        default=None,
        description="Motivo da recusa quando is_refusal é True.",
    )

    @model_validator(mode="after")
    def consistencia_entre_recusa_confianca_e_fontes(self) -> "RAGResponse":
        if self.is_refusal:
            if self.confidence_level != "Recusado":
                raise ValueError(
                    "Toda resposta recusada (is_refusal=True) deve ter confidence_level='Recusado'."
                )
            if self.refusal_reason is None:
                raise ValueError(
                    "Toda resposta recusada (is_refusal=True) deve informar refusal_reason."
                )
            if self.sources_used:
                raise ValueError(
                    "Uma resposta recusada (is_refusal=True) não deve citar sources_used."
                )
        else:
            if self.confidence_level == "Recusado":
                raise ValueError(
                    "confidence_level='Recusado' só é permitido quando is_refusal=True."
                )
            if self.refusal_reason is not None:
                raise ValueError(
                    "refusal_reason só é permitido quando is_refusal=True."
                )
            if not self.sources_used:
                raise ValueError(
                    "Toda resposta não recusada deve citar ao menos uma evidência em sources_used."
                )

        return self
