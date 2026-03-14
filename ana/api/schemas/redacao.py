"""Contratos de input/output para o router /redacao."""

from pydantic import BaseModel, Field


class RequisicaoReformular(BaseModel):
    """Requisição para reformulação de texto jurídico.

    Attributes:
        texto: Texto a ser reformulado.
        instrucoes: Instruções adicionais opcionais para guiar a reescrita.
    """

    texto: str = Field(description="Texto a ser reformulado em linguagem jurídica formal")
    instrucoes: str = Field(
        default="",
        description="Instruções adicionais para guiar a reescrita (opcional)",
    )


class RespostaReformular(BaseModel):
    """Resposta com o texto reformulado.

    Attributes:
        texto_reformulado: Texto reescrito em linguagem jurídica formal.
        modelo_usado: Nome do modelo LLM utilizado.
    """

    texto_reformulado: str
    modelo_usado: str
