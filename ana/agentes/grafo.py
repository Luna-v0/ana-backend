"""Grafo LangGraph do sistema ANA — orquestrador de agentes jurídicos.

Implementa o ``StateGraph(EstadoJuridico)`` com nós especializados para
cada tipo de tarefa jurídica. O roteamento é feito pela intenção
classificada pelo nó ``classificar``.

Fluxo completo:
    START → classificar → {roteador}
        "pesquisa_legal"   → pesquisar_legislacao → validar_leis → reformular → gerar_resposta
        "analise_processo" → analisar_processo ──────────────────→ reformular → gerar_resposta
        "verificar_prazo"  → analisar_processo ──────────────────→ reformular → gerar_resposta
        outros             ──────────────────────────────────────→ reformular → gerar_resposta

O checkpointer ``AsyncSqliteSaver`` persiste o estado por ``thread_id``
(= ``sessao_id``), habilitando conversas multi-turno com memória.

Uso:
    async with obter_grafo() as app:
        resultado = await app.ainvoke(estado, config={"configurable": {"thread_id": sessao_id}})
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from loguru import logger

from ana.agentes.estado import EstadoJuridico, INTENCOES_VALIDAS
from ana.agentes.nos.classificar import no_classificar
from ana.agentes.nos.pesquisar import no_pesquisar_legislacao
from ana.agentes.nos.analisar import no_analisar_processo
from ana.agentes.nos.validar import no_validar_leis
from ana.agentes.nos.reformular import no_reformular
from ana.agentes.nos.resposta import no_gerar_resposta


# =============================================================================
# Roteador de intenção
# =============================================================================

def _roteador(estado: EstadoJuridico) -> str:
    """Decide o próximo nó com base na intenção classificada.

    Args:
        estado: Estado atual com ``intencao`` preenchida pelo nó classificar.

    Returns:
        Nome do nó destino.
    """
    intencao = estado.get("intencao", "desconhecida")

    if intencao == "pesquisa_legal":
        return "pesquisar_legislacao"

    if intencao in ("analise_processo", "verificar_prazo"):
        return "analisar_processo"

    # Todas as outras intenções (desconhecida, gerar_documento, etc.)
    # vão direto para reformular → gerar_resposta
    return "reformular"


# =============================================================================
# Construção do grafo
# =============================================================================

def _construir_grafo():
    """Constrói e retorna o StateGraph sem checkpointer (para compilação posterior).

    Returns:
        Instância não-compilada de ``StateGraph[EstadoJuridico]``.
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(EstadoJuridico)

    g.add_node("classificar",          no_classificar)
    g.add_node("pesquisar_legislacao", no_pesquisar_legislacao)
    g.add_node("analisar_processo",    no_analisar_processo)
    g.add_node("validar_leis",         no_validar_leis)
    g.add_node("reformular",           no_reformular)
    g.add_node("gerar_resposta",       no_gerar_resposta)

    g.set_entry_point("classificar")

    g.add_conditional_edges(
        "classificar",
        _roteador,
        {
            "pesquisar_legislacao": "pesquisar_legislacao",
            "analisar_processo":    "analisar_processo",
            "reformular":           "reformular",
        },
    )

    # pesquisa_legal: pesquisar → validar → reformular
    g.add_edge("pesquisar_legislacao", "validar_leis")
    g.add_edge("validar_leis",         "reformular")

    # analise_processo / verificar_prazo: analisar → reformular
    g.add_edge("analisar_processo", "reformular")

    # Todos os ramos confluem em reformular → gerar_resposta → END
    g.add_edge("reformular",     "gerar_resposta")
    g.add_edge("gerar_resposta", END)

    return g


# =============================================================================
# Gerenciamento do checkpointer
# =============================================================================

def _caminho_checkpoints() -> str:
    """Resolve o caminho do banco SQLite de checkpoints.

    Returns:
        Path absoluto do arquivo ``.db`` conforme ambiente (Docker ou local).
    """
    if os.path.exists("/.dockerenv"):
        caminho = Path("/app/data/checkpoints.db")
    else:
        caminho = Path.home() / ".local" / "share" / "ana" / "checkpoints.db"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return str(caminho)


@asynccontextmanager
async def obter_grafo() -> AsyncGenerator[Any, None]:
    """Context manager assíncrono que fornece o grafo compilado com checkpointer.

    Garante que o ``AsyncSqliteSaver`` seja inicializado e encerrado
    corretamente. Deve ser usado como dependência FastAPI ou como
    context manager direto.

    Yields:
        Grafo compilado (``CompiledStateGraph``) pronto para ``ainvoke``
        ou ``astream``.

    Example:
        async with obter_grafo() as app:
            resultado = await app.ainvoke(estado, config=config)
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    caminho = _caminho_checkpoints()
    logger.debug(f"Checkpointer SQLite: {caminho}")

    async with AsyncSqliteSaver.from_conn_string(caminho) as checkpointer:
        grafo = _construir_grafo()
        app = grafo.compile(checkpointer=checkpointer)
        yield app


def config_sessao(sessao_id: str) -> dict:
    """Gera o dict de configuração LangGraph para uma sessão.

    O ``thread_id`` determina o histórico de checkpoints multi-turno.
    Usar o mesmo ``thread_id`` em chamadas subsequentes preserva contexto.

    Args:
        sessao_id: ID da sessão de processo (UUID).

    Returns:
        Dict de configuração compatível com ``app.ainvoke(estado, config=...)``.
    """
    return {"configurable": {"thread_id": sessao_id}}
