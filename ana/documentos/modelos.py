"""Modelos de dados para geração de documentos jurídicos."""

from enum import Enum

from pydantic import BaseModel, Field


class TipoPeca(str, Enum):
    """Tipos de peças processuais suportados."""

    PETICAO_INICIAL = "peticao_inicial"
    CONTESTACAO = "contestacao"
    RECURSO = "recurso"
    EXPORTAR_TRANSCRICAO = "exportar_transcricao"


_LABELS: dict[str, str] = {
    TipoPeca.PETICAO_INICIAL: "Petição Inicial",
    TipoPeca.CONTESTACAO: "Contestação",
    TipoPeca.RECURSO: "Recurso",
    TipoPeca.EXPORTAR_TRANSCRICAO: "Transcrição de Audiência",
}


def label_peca(tipo: TipoPeca) -> str:
    return _LABELS.get(tipo, tipo.value.replace("_", " ").title())


class RequisicaoGerarDocumento(BaseModel):
    """Requisição para geração de peça processual."""

    sessao_id: str = Field(..., description="ID da sessão do processo")
    tipo_peca: TipoPeca = Field(
        TipoPeca.PETICAO_INICIAL, description="Tipo de peça a gerar"
    )
    instrucoes: str = Field(
        "", description="Instruções adicionais para o modelo (ex: tese jurídica específica)"
    )
