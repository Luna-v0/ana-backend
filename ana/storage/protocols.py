"""Protocolos de storage do sistema ANA.

Define os contratos de `VectorStoreProtocol` e `CacheProtocol` via
`typing.Protocol` (structural subtyping). Qualquer classe que implemente
os métodos listados satisfaz o protocolo sem herança explícita.

Backends disponíveis:
    - PostgreSQL + pgvector (padrão, ``storage_backend: postgres``)
    - Qdrant + SQLite (legado, ``storage_backend: qdrant``)

Exemplo de uso:
    >>> from ana.storage.protocols import VectorStoreProtocol
    >>> from ana.storage.pgvector_store import IndexadorPgVector
    >>> assert isinstance(IndexadorPgVector(), VectorStoreProtocol)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from ana.rag.modelos import ChunkJuridico, FiltrosBusca


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Contrato para backends de busca vetorial.

    Qualquer implementação que satisfaça este protocolo pode ser usada
    por `PipelineRetrieval` e `PipelineScrapers` sem alteração.

    Implementações concretas:
        - `ana.storage.pgvector_store.IndexadorPgVector` (backend padrão)
        - `ana.rag.indexador.IndexadorQdrant` (legado)
    """

    def criar_colecao(self, nome_colecao: str, recriar: bool = False) -> None:
        """Cria uma collection/tabela para armazenamento de chunks.

        Args:
            nome_colecao: Nome da collection a criar.
            recriar: Se True, apaga e recria caso já exista.
        """
        ...

    def criar_colecao_legislacao(self, recriar: bool = False) -> None:
        """Cria a collection global de legislação brasileira.

        Args:
            recriar: Se True, apaga e recria.
        """
        ...

    def criar_colecao_sessao(self, sessao_id: str, recriar: bool = False) -> str:
        """Cria collection isolada para uma sessão de processo.

        Args:
            sessao_id: ID único da sessão.
            recriar: Se True, apaga e recria.

        Returns:
            Nome da collection criada.
        """
        ...

    def indexar_chunks(
        self,
        chunks: list[ChunkJuridico],
        nome_colecao: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Indexa lista de chunks com embeddings no backend.

        Args:
            chunks: Chunks com `embedding` já preenchido.
            nome_colecao: Collection destino. Usa legislação global se None.
            batch_size: Tamanho do lote para upsert.

        Returns:
            Número de chunks indexados com sucesso.
        """
        ...

    def busca_semantica(
        self,
        vetor_query: list[float],
        filtros: FiltrosBusca | None = None,
        nome_colecao: str | None = None,
        limite: int = 20,
    ) -> list[dict[str, Any]]:
        """Busca vetorial semântica com filtros opcionais.

        Args:
            vetor_query: Vetor de embedding da query (ex: 1024 dims).
            filtros: Filtros de metadata para restringir a busca.
            nome_colecao: Collection a buscar. Usa legislação se None.
            limite: Número máximo de resultados.

        Returns:
            Lista de dicionários com ``id``, ``score`` e ``payload``.
        """
        ...

    def verificar_conexao(self) -> bool:
        """Verifica se o backend está acessível.

        Returns:
            True se a conexão foi bem-sucedida, False caso contrário.
        """
        ...

    def listar_colecoes(self) -> list[str]:
        """Lista todas as collections/tabelas existentes.

        Returns:
            Lista de nomes de collections.
        """
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Contrato para backends de cache de documentos coletados.

    Evita recoleta de documentos que não sofreram alterações, usando
    hash SHA-256 do conteúdo.

    Implementações concretas:
        - `ana.scrapers.cache.CacheScrapers` (SQLite, backend padrão)
        - `ana.storage.postgres_cache.CachePostgres` (PostgreSQL)
    """

    def ja_coletado(self, url: str, hash_conteudo: str) -> bool:
        """Verifica se um documento já foi coletado com o mesmo hash.

        Args:
            url: URL canônica do documento.
            hash_conteudo: Hash SHA-256 truncado do conteúdo atual.

        Returns:
            True se o documento existe no cache com o mesmo hash.
        """
        ...

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
        ...

    def ultima_coleta(self, fonte: str) -> datetime | None:
        """Retorna o timestamp da última coleta de uma fonte.

        Args:
            fonte: Nome da fonte.

        Returns:
            Datetime da última coleta ou None se nunca coletada.
        """
        ...

    def total(self, fonte: str | None = None) -> int:
        """Conta documentos no cache, opcionalmente por fonte.

        Args:
            fonte: Filtrar por fonte. None = todos.

        Returns:
            Quantidade de documentos.
        """
        ...

    def listar(self) -> list[dict]:
        """Lista todos os documentos no cache.

        Returns:
            Lista de dicionários com campos do cache.
        """
        ...
