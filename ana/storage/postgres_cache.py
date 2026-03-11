"""Implementação PostgreSQL do CacheProtocol.

Funcionalidade idêntica a `CacheScrapers` (SQLite), substituindo
`sqlite3` por `psycopg` (psycopg3). Satisfaz `CacheProtocol` via
structural subtyping.

Schema:
    ``documentos_cache`` — tabela com URL como chave primária,
    hash SHA-256 do conteúdo, metadados e timestamp com timezone.

Dependências (grupo ``postgres``):
    - ``psycopg[binary]>=3.2.0``

Exemplo de uso:
    >>> from ana.storage.postgres_cache import CachePostgres
    >>> cache = CachePostgres()
    >>> cache.registrar("https://...", "abc123", "planalto", "Lei X")
    >>> cache.ja_coletado("https://...", "abc123")
    True
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from ana.config import obter_configuracao


class CachePostgres:
    """Controla quais URLs já foram coletadas usando PostgreSQL.

    Mesma interface de `CacheScrapers` — troca SQLite por psycopg3.

    Attributes:
        _dsn: DSN de conexão PostgreSQL.
    """

    def __init__(self) -> None:
        """Inicializa o cache, criando a tabela se não existir."""
        self._dsn = obter_configuracao().postgres_dsn
        self._inicializar()

    def _get_conn(self):
        """Abre conexão psycopg3 com autocommit.

        Returns:
            Conexão psycopg3 pronta para uso.
        """
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def _inicializar(self) -> None:
        """Cria a tabela de cache se não existir."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documentos_cache (
                    url         TEXT PRIMARY KEY,
                    hash        TEXT NOT NULL,
                    fonte       TEXT NOT NULL,
                    titulo      TEXT DEFAULT '',
                    vigencia    TEXT DEFAULT 'ativa',
                    data_coleta TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS documentos_cache_fonte_idx
                ON documentos_cache (fonte)
            """)
        finally:
            conn.close()

    def ja_coletado(self, url: str, hash_conteudo: str) -> bool:
        """Verifica se um documento já foi coletado com o mesmo hash.

        Args:
            url: URL canônica do documento.
            hash_conteudo: Hash SHA-256 truncado do conteúdo atual.

        Returns:
            True se o documento existe no cache com o mesmo hash.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT hash FROM documentos_cache WHERE url = %s", (url,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None and row[0] == hash_conteudo

    def registrar(
        self,
        url: str,
        hash_conteudo: str,
        fonte: str,
        titulo: str = "",
        vigencia: str = "ativa",
    ) -> None:
        """Registra ou atualiza um documento no cache.

        Args:
            url: URL canônica do documento.
            hash_conteudo: Hash SHA-256 truncado do conteúdo.
            fonte: Nome da fonte (ex: ``'planalto'``).
            titulo: Título do documento.
            vigencia: Status de vigência.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO documentos_cache
                    (url, hash, fonte, titulo, vigencia, data_coleta)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (url) DO UPDATE SET
                    hash        = EXCLUDED.hash,
                    fonte       = EXCLUDED.fonte,
                    titulo      = EXCLUDED.titulo,
                    vigencia    = EXCLUDED.vigencia,
                    data_coleta = NOW()
                """,
                (url, hash_conteudo, fonte, titulo, vigencia),
            )
        finally:
            conn.close()
        logger.debug(f"Cache PostgreSQL: registrado '{titulo}' de '{fonte}'")

    def ultima_coleta(self, fonte: str) -> datetime | None:
        """Retorna o timestamp da última coleta de uma fonte.

        Args:
            fonte: Nome da fonte.

        Returns:
            Datetime da última coleta ou None se nunca coletada.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(data_coleta) FROM documentos_cache WHERE fonte = %s",
                (fonte,),
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            # psycopg3 retorna datetime com timezone — normaliza para naive
            dt = row[0]
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        return None

    def total(self, fonte: str | None = None) -> int:
        """Conta documentos no cache, opcionalmente por fonte.

        Args:
            fonte: Filtrar por fonte. None = todos.

        Returns:
            Quantidade de documentos.
        """
        conn = self._get_conn()
        try:
            if fonte:
                row = conn.execute(
                    "SELECT COUNT(*) FROM documentos_cache WHERE fonte = %s",
                    (fonte,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM documentos_cache"
                ).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0

    def listar(self) -> list[dict]:
        """Lista todos os documentos no cache.

        Returns:
            Lista de dicionários com campos do cache.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT url, hash, fonte, titulo, vigencia, data_coleta
                FROM documentos_cache
                ORDER BY data_coleta DESC
                """
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "url": r[0],
                "hash": r[1],
                "fonte": r[2],
                "titulo": r[3],
                "vigencia": r[4],
                "data_coleta": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
