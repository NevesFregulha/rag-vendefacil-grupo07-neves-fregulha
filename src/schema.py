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

RefusalReason = Literal["LGPD_PROTECTION", "OUT_OF_DOMAIN", "CREDENTIAL_PROTECTION"]

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
        max_length=CITATION_MAX_LENGTH,
        description="Trecho exato do texto ou dado utilizado para fundamentar a afirmação (até "
        f"{CITATION_MAX_LENGTH} caracteres).",
    )
    doc_type: Optional[DocType] = Field(
        default=None,
        description="Tipo de documento de origem do chunk.",
    )

    @field_validator("quotation")
    @classmethod
    def quotation_nao_pode_ser_so_espacos(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("quotation não pode ser vazia ou conter apenas espaços.")
        return stripped


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
