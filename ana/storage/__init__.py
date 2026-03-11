"""Fábrica de backends de armazenamento do sistema ANA.

Seleciona automaticamente o backend de vector store e cache conforme
``storage_backend`` em ``config/modelos.yaml``:

- ``postgres`` (padrão) → `IndexadorPgVector` + `CachePostgres`
- ``qdrant``   (legado)  → `IndexadorQdrant` + `CacheScrapers` (SQLite)

Uso nos pipelines:
    >>> from ana.storage import obter_vector_store, obter_cache
    >>> store = obter_vector_store()   # IndexadorPgVector (padrão)
    >>> cache = obter_cache()          # CachePostgres (padrão)

A troca de backend é zero-código: basta alterar ``storage_backend`` no
arquivo ``config/modelos.yaml`` e reiniciar a aplicação.
"""

from __future__ import annotations

import os
from pathlib import Path

from ana.storage.protocols import CacheProtocol, VectorStoreProtocol

__all__ = [
    "obter_vector_store",
    "obter_cache",
    "VectorStoreProtocol",
    "CacheProtocol",
]


def obter_vector_store() -> VectorStoreProtocol:
    """Retorna o backend de vector store conforme ``storage_backend``.

    Returns:
        `IndexadorQdrant` se backend for ``'qdrant'`` (padrão),
        `IndexadorPgVector` se backend for ``'postgres'``.
    """
    if _backend() == "postgres":
        from ana.storage.pgvector_store import IndexadorPgVector

        return IndexadorPgVector()
    from ana.rag.indexador import IndexadorQdrant

    return IndexadorQdrant()


def obter_cache() -> CacheProtocol:
    """Retorna o backend de cache conforme ``storage_backend``.

    Returns:
        `CacheScrapers` (SQLite) se backend for ``'qdrant'`` (padrão),
        `CachePostgres` se backend for ``'postgres'``.
    """
    if _backend() == "postgres":
        from ana.storage.postgres_cache import CachePostgres

        return CachePostgres()
    try:
        from leis_br.cache import CacheScrapers
    except ImportError as e:
        raise RuntimeError(
            "leis-br não instalado. O cache SQLite requer o pacote leis-br.\n"
            "Use storage_backend: postgres ou instale leis-br."
        ) from e
    return CacheScrapers(_caminho_sqlite())


def _backend() -> str:
    """Lê o backend ativo de ``config/modelos.yaml``.

    Returns:
        ``'qdrant'`` ou ``'postgres'``.
    """
    from ana.config_modelos import obter_modelos

    return obter_modelos().storage_backend


def _caminho_sqlite() -> Path:
    """Resolve o caminho do banco SQLite conforme o ambiente.

    Returns:
        Path para ``scrapers.db`` em ``~/.local/share/ana`` (local)
        ou ``/app/data`` (Docker).
    """
    if os.path.exists("/.dockerenv"):
        return Path("/app/data/scrapers.db")
    return Path.home() / ".local" / "share" / "ana" / "scrapers.db"
