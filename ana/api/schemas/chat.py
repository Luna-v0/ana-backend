"""Schemas Pydantic para o endpoint de chat agêntico.

Define os contratos de entrada e saída do ``POST /chat``,
incluindo suporte a modo streaming (SSE) e não-streaming (JSON).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RequisicaoChat(BaseModel):
    """Requisição de mensagem ao agente jurídico.

    Attributes:
        mensagem: Texto da pergunta ou instrução do usuário.
        sessao_id: ID da sessão de processo para contexto e checkpointing.
            Se vazio, a conversa não terá contexto de processo.
        transcricao_anexada: Texto de transcrição de audiência anexado ao chat.
            Quando presente, é incluído como contexto adicional em todos os
            nós de análise do grafo.
        stream: Se True, a resposta é entregue via SSE (text/event-stream).
            Se False, bloqueia até gerar a resposta completa (JSON).
    """

    mensagem: str = Field(
        description="Mensagem ou pergunta do usuário ao assistente jurídico.",
        min_length=1,
        max_length=4096,
    )
    sessao_id: str = Field(
        default="",
        description="ID da sessão de processo para contexto RAG e memória multi-turno.",
    )
    transcricao_anexada: Optional[str] = Field(
        default=None,
        description=(
            "Texto de transcrição de audiência anexado pelo usuário. "
            "Usado como contexto adicional pelos nós de pesquisa e análise."
        ),
        max_length=50000,
    )
    documento_processo: Optional[str] = Field(
        default=None,
        description=(
            "Texto extraído de documento de processo jurídico (PDF ou texto). "
            "Usado como query e contexto para descoberta de leis relacionadas."
        ),
        max_length=100000,
    )
    stream: bool = Field(
        default=True,
        description="Se True, retorna resposta via SSE (streaming de tokens).",
    )


class RespostaChat(BaseModel):
    """Resposta completa do agente jurídico (modo não-streaming).

    Attributes:
        resposta: Texto final gerado pelo LLM.
        intencao: Intenção classificada pelo orquestrador.
        sessao_id: ID da sessão usada.
        descricao: Título curto gerado para o chat (máximo 8 palavras).
        contexto_rag: Chunks recuperados (legislação ou processo).
        validacao: Resultado da validação anti-alucinação das leis citadas.
        erro: Mensagem de erro se algum nó falhou (não fatal).
    """

    resposta: str = Field(description="Resposta gerada pelo assistente jurídico.")
    intencao: str = Field(description="Intenção classificada pelo orquestrador.")
    sessao_id: str = Field(description="ID da sessão usada na conversa.")
    descricao: Optional[str] = Field(
        default=None,
        description="Título curto do chat gerado pelo modelo (máximo 8 palavras).",
    )
    contexto_rag: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chunks de legislação ou processo usados como contexto.",
    )
    validacao: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Resultado da validação anti-alucinação das leis citadas.",
    )
    erro: Optional[str] = Field(
        default=None,
        description="Mensagem de erro não-fatal durante o processamento.",
    )


class EventoSSE(BaseModel):
    """Evento individual de Server-Sent Event no streaming do chat.

    Attributes:
        tipo: Tipo do evento — ``"token"`` para fragmento de texto,
            ``"metadados"`` para informações de contexto,
            ``"fim"`` para indicar encerramento do stream,
            ``"erro"`` para erros durante streaming.
        token: Fragmento de texto do LLM (apenas para tipo ``"token"``).
        dados: Payload adicional (para tipos ``"metadados"`` e ``"fim"``).
    """

    tipo: str = Field(description="Tipo do evento SSE.")
    token: Optional[str] = Field(default=None, description="Fragmento de texto.")
    dados: Optional[dict[str, Any]] = Field(default=None, description="Payload do evento.")
