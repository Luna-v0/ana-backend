"""Funções de busca para documentos de sessões de processos.

Três modos de busca:
- intra: Apenas documentos desta sessão (tabela 'processos', filtro sessao_id)
- global: Legislação + documentos desta sessão (legislacao ∪ processos)
- cross: Documentos de todas as sessões exceto a atual
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def buscar_intra_sessao(
    query: str,
    sessao_id: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Busca apenas nos documentos da sessão indicada.

    Usa a tabela 'processos' com filtro sessao_id via FiltrosBusca.

    Args:
        query: Texto da query.
        sessao_id: ID da sessão a buscar.
        top_k: Número máximo de resultados.

    Returns:
        Lista de dicionários com 'id', 'score', 'payload'.
    """
    from ana.rag.modelos import FiltrosBusca
    from ana.rag.embeddings import GeradorEmbeddings
    from ana.storage.pgvector_store import IndexadorPgVector

    gerador = GeradorEmbeddings()
    vetor = gerador.gerar_query(query)

    filtros = FiltrosBusca(sessao_id=sessao_id, vigencia=None)
    indexador = IndexadorPgVector()

    resultados = indexador.busca_semantica(
        vetor_query=vetor,
        filtros=filtros,
        nome_colecao="processos",
        limite=top_k,
    )
    logger.debug(f"buscar_intra_sessao: {len(resultados)} resultados (sessao={sessao_id})")
    return resultados


def buscar_global(
    query: str,
    sessao_id: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Busca na legislação global + documentos da sessão.

    Combina resultados das tabelas 'legislacao_brasileira' e 'processos'
    (filtrado por sessao_id), fundidos por score decrescente.

    Args:
        query: Texto da query.
        sessao_id: ID da sessão para incluir documentos do processo.
        top_k: Número máximo de resultados combinados.

    Returns:
        Lista fundida ordenada por score decrescente.
    """
    from ana.config import obter_configuracao
    from ana.rag.modelos import FiltrosBusca
    from ana.rag.embeddings import GeradorEmbeddings
    from ana.storage.pgvector_store import IndexadorPgVector

    config = obter_configuracao()
    gerador = GeradorEmbeddings()
    vetor = gerador.gerar_query(query)
    indexador = IndexadorPgVector()

    # Busca em legislação
    res_leg = indexador.busca_semantica(
        vetor_query=vetor,
        filtros=FiltrosBusca(),
        nome_colecao=config.colecao_legislacao,
        limite=top_k,
    )

    # Busca em documentos da sessão
    res_proc = indexador.busca_semantica(
        vetor_query=vetor,
        filtros=FiltrosBusca(sessao_id=sessao_id, vigencia=None),
        nome_colecao="processos",
        limite=top_k,
    )

    # Funde e ordena por score
    combinados = res_leg + res_proc
    combinados.sort(key=lambda r: r["score"], reverse=True)
    return combinados[:top_k]


def buscar_cross_sessao(
    query: str,
    excluir_sessao_id: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Busca em documentos de todas as sessões exceto a indicada.

    Útil para encontrar precedentes em outros processos.

    Args:
        query: Texto da query.
        excluir_sessao_id: ID da sessão a excluir dos resultados.
        top_k: Número máximo de resultados.

    Returns:
        Lista de chunks de outros processos, ordenados por score.
    """
    from ana.rag.embeddings import GeradorEmbeddings
    from ana.storage.pgvector_store import IndexadorPgVector
    import numpy as np

    gerador = GeradorEmbeddings()
    vetor = gerador.gerar_query(query)
    vetor_np = np.array(vetor, dtype=np.float32)

    indexador = IndexadorPgVector()
    conn = indexador._get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id::text, payload, 1 - (vetor <=> %s) AS score
            FROM processos
            WHERE sessao_id IS DISTINCT FROM %s
            ORDER BY vetor <=> %s
            LIMIT %s
            """,
            (vetor_np, excluir_sessao_id, vetor_np, top_k),
        ).fetchall()
    finally:
        conn.close()

    resultados = [
        {"id": row[0], "payload": dict(row[1]), "score": float(row[2])}
        for row in rows
    ]
    logger.debug(
        f"buscar_cross_sessao: {len(resultados)} resultados "
        f"(excluindo sessao={excluir_sessao_id})"
    )
    return resultados
