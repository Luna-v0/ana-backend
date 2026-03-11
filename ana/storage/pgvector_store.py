"""Implementação pgvector do VectorStoreProtocol.

Usa PostgreSQL + pgvector (v0.7+) com índice HNSW e distância cosseno.
A qualidade matemática é idêntica ao Qdrant: ambos usam distância cosseno
com HNSW. Diferença prática: filtros via WHERE SQL (adequado para
<100k chunks).

Dependências (grupo ``postgres``):
    - ``psycopg[binary]>=3.2.0``
    - ``pgvector>=0.3.6``

Schema por collection (tabela):
    - ``id``          UUID PK gerado automaticamente
    - ``vetor``       vector(N) NOT NULL — embedding do chunk
    - ``texto``       TEXT — conteúdo textual
    - Colunas de metadata (vigencia, tipo, area, etc.) para filtragem SQL
    - ``payload``     JSONB — todos os campos, espelho do payload Qdrant

Índices:
    - HNSW em ``vetor`` (cosine ops)
    - B-tree em ``vigencia`` e ``tipo``

Exemplo de uso:
    >>> from ana.storage.pgvector_store import IndexadorPgVector
    >>> idx = IndexadorPgVector()
    >>> idx.criar_colecao_legislacao()
    >>> idx.indexar_chunks(chunks)
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from loguru import logger

from ana.config import obter_configuracao
from ana.config_modelos import obter_modelos
from ana.rag.modelos import ChunkJuridico, FiltrosBusca

# Regex para validar nomes de tabela (previne SQL injection)
_NOME_VALIDO = re.compile(r"^[a-z][a-z0-9_]*$")


def _validar_nome_tabela(nome: str) -> str:
    """Valida e retorna o nome de tabela para uso seguro em SQL.

    Args:
        nome: Nome da tabela a validar.

    Returns:
        Nome validado.

    Raises:
        ValueError: Se o nome contiver caracteres não permitidos.
    """
    if not _NOME_VALIDO.match(nome):
        raise ValueError(
            f"Nome de collection inválido: '{nome}'. "
            "Use apenas letras minúsculas, dígitos e underscores, "
            "iniciando com letra."
        )
    return nome


class IndexadorPgVector:
    """Gerencia indexação e busca semântica de chunks jurídicos no PostgreSQL.

    Implementa `VectorStoreProtocol` via structural subtyping — compatível
    com `IndexadorQdrant` sem herança.

    Responsável por:
    - Criação de tabelas com índice HNSW (pgvector cosine ops)
    - Indexação em lote de chunks com embeddings (UPSERT por UUID)
    - Busca vetorial com filtros SQL de metadata

    Attributes:
        colecao_legislacao: Nome da tabela global de legislação.
        prefixo_sessao: Prefixo para tabelas de sessão.
        dimensao_vetores: Dimensão dos vetores de embedding.
        _dsn: DSN de conexão PostgreSQL.
    """

    def __init__(self) -> None:
        """Inicializa o indexador com configuração do banco PostgreSQL."""
        config = obter_configuracao()
        config_modelos = obter_modelos().ativo.embeddings

        self._dsn = config.postgres_dsn
        self.colecao_legislacao = config.colecao_legislacao
        self.prefixo_sessao = config.prefixo_colecao_sessao
        self.dimensao_vetores = config_modelos.dimensao

    def _get_conn(self):
        """Abre conexão psycopg3 com pgvector registrado (autocommit).

        Returns:
            Conexão psycopg3 pronta para uso.
        """
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn, autocommit=True)
        register_vector(conn)
        return conn

    def _nome_colecao_sessao(self, sessao_id: str) -> str:
        """Retorna o nome de tabela para uma sessão (sanitizado).

        Args:
            sessao_id: ID único da sessão.

        Returns:
            Nome de tabela no formato ``'{prefixo}_{sessao_id_sanitizado}'``.
        """
        sanitizado = re.sub(r"[^a-z0-9]", "_", sessao_id.lower())
        return f"{self.prefixo_sessao}_{sanitizado}"

    def criar_colecao(self, nome_colecao: str, recriar: bool = False) -> None:
        """Cria tabela PostgreSQL para chunks jurídicos com índice HNSW.

        Schema criado:
        - Colunas tipadas para filtragem eficiente (vigencia, tipo, area, etc.)
        - Coluna ``payload`` JSONB espelhando o formato do Qdrant
        - Índice HNSW em ``vetor`` (cosine ops)
        - Índices B-tree em ``vigencia`` e ``tipo``

        Args:
            nome_colecao: Nome da tabela a criar.
            recriar: Se True, apaga e recria a tabela.
        """
        nome = _validar_nome_tabela(nome_colecao)
        conn = self._get_conn()
        try:
            if recriar:
                conn.execute(f"DROP TABLE IF EXISTS {nome}")
                logger.warning(f"Tabela pgvector apagada: {nome}")

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {nome} (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    vetor       vector({self.dimensao_vetores}) NOT NULL,
                    texto       TEXT,
                    fonte       TEXT,
                    tipo        TEXT,
                    area        TEXT,
                    vigencia    TEXT DEFAULT 'ativa',
                    orgao       TEXT,
                    artigo      TEXT,
                    titulo      TEXT,
                    capitulo    TEXT,
                    secao       TEXT,
                    url_origem  TEXT,
                    data_publicacao DATE,
                    sessao_id   TEXT,
                    payload     JSONB NOT NULL DEFAULT '{{}}'
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {nome}_vetor_hnsw
                ON {nome} USING hnsw (vetor vector_cosine_ops)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {nome}_vigencia_idx
                ON {nome} (vigencia)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {nome}_tipo_idx
                ON {nome} (tipo)
            """)
            logger.info(
                f"Tabela pgvector: {nome} "
                f"(dims={self.dimensao_vetores}, distância=COSINE/HNSW)"
            )
        finally:
            conn.close()

    def criar_colecao_legislacao(self, recriar: bool = False) -> None:
        """Cria a tabela global de legislação brasileira.

        Args:
            recriar: Se True, apaga e recria a tabela.
        """
        self.criar_colecao(self.colecao_legislacao, recriar=recriar)

    def criar_colecao_processos(self, recriar: bool = False) -> None:
        """Cria a tabela de documentos de sessões de processos.

        Os documentos enviados por usuários (PDFs, DOCX, TXT) são indexados
        aqui com ``sessao_id`` preenchido para filtragem por processo.

        Args:
            recriar: Se True, apaga e recria a tabela.
        """
        self.criar_colecao("processos", recriar=recriar)

    def criar_colecao_sessao(self, sessao_id: str, recriar: bool = False) -> str:
        """Cria tabela isolada para uma sessão de processo.

        Args:
            sessao_id: ID único da sessão.
            recriar: Se True, apaga e recria.

        Returns:
            Nome da tabela criada.
        """
        nome = self._nome_colecao_sessao(sessao_id)
        self.criar_colecao(nome, recriar=recriar)
        return nome

    def _chunk_para_payload(self, chunk: ChunkJuridico) -> dict[str, Any]:
        """Converte metadata do chunk para dicionário payload.

        Produz o mesmo formato do payload Qdrant para compatibilidade
        com o pipeline pós-retrieval.

        Args:
            chunk: Chunk jurídico com metadata completa.

        Returns:
            Dicionário plano com todos os campos serializáveis.
        """
        meta = chunk.metadata
        payload: dict[str, Any] = {
            "texto": chunk.texto,
            "fonte": meta.fonte,
            "tipo": meta.tipo.value,
            "vigencia": meta.vigencia.value,
        }
        if meta.area is not None:
            payload["area"] = meta.area.value
        if meta.titulo is not None:
            payload["titulo"] = meta.titulo
        if meta.capitulo is not None:
            payload["capitulo"] = meta.capitulo
        if meta.secao is not None:
            payload["secao"] = meta.secao
        if meta.artigo is not None:
            payload["artigo"] = meta.artigo
        if meta.orgao is not None:
            payload["orgao"] = meta.orgao
        if meta.url_origem is not None:
            payload["url_origem"] = meta.url_origem
        if meta.data_publicacao is not None:
            payload["data_publicacao"] = meta.data_publicacao.isoformat()
        if meta.sessao_id is not None:
            payload["sessao_id"] = meta.sessao_id
        return payload

    def indexar_chunks(
        self,
        chunks: list[ChunkJuridico],
        nome_colecao: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Indexa lista de chunks no PostgreSQL com UPSERT por UUID.

        Os chunks devem ter o campo `embedding` preenchido antes de chamar
        este método. Use `GeradorEmbeddings.gerar_batch()` para isso.

        Args:
            chunks: Lista de chunks com embeddings já gerados.
            nome_colecao: Nome da tabela. Usa legislação global se None.
            batch_size: Tamanho do lote para UPSERT.

        Returns:
            Número de chunks indexados com sucesso.

        Raises:
            ValueError: Se algum chunk não tiver embedding gerado.
        """
        import numpy as np
        from psycopg.types.json import Jsonb

        colecao = _validar_nome_tabela(nome_colecao or self.colecao_legislacao)

        sem_embedding = [i for i, c in enumerate(chunks) if c.embedding is None]
        if sem_embedding:
            raise ValueError(
                f"Chunks {sem_embedding[:5]} não têm embedding. "
                "Chame GeradorEmbeddings.gerar_batch() antes de indexar."
            )

        total_indexado = 0
        conn = self._get_conn()
        try:
            for i in range(0, len(chunks), batch_size):
                lote = chunks[i : i + batch_size]

                for chunk in lote:
                    meta = chunk.metadata
                    payload = self._chunk_para_payload(chunk)
                    chunk_id = chunk.id or str(uuid.uuid4())
                    vetor = np.array(chunk.embedding, dtype=np.float32)

                    conn.execute(
                        f"""
                        INSERT INTO {colecao}
                            (id, vetor, texto, fonte, tipo, area, vigencia,
                             orgao, artigo, titulo, capitulo, secao,
                             url_origem, data_publicacao, sessao_id, payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s,
                                %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            vetor           = EXCLUDED.vetor,
                            texto           = EXCLUDED.texto,
                            fonte           = EXCLUDED.fonte,
                            tipo            = EXCLUDED.tipo,
                            area            = EXCLUDED.area,
                            vigencia        = EXCLUDED.vigencia,
                            orgao           = EXCLUDED.orgao,
                            artigo          = EXCLUDED.artigo,
                            titulo          = EXCLUDED.titulo,
                            capitulo        = EXCLUDED.capitulo,
                            secao           = EXCLUDED.secao,
                            url_origem      = EXCLUDED.url_origem,
                            data_publicacao = EXCLUDED.data_publicacao,
                            sessao_id       = EXCLUDED.sessao_id,
                            payload         = EXCLUDED.payload
                        """,
                        (
                            chunk_id,
                            vetor,
                            chunk.texto,
                            meta.fonte,
                            meta.tipo.value,
                            meta.area.value if meta.area else None,
                            meta.vigencia.value,
                            meta.orgao,
                            meta.artigo,
                            meta.titulo,
                            meta.capitulo,
                            meta.secao,
                            meta.url_origem,
                            meta.data_publicacao,
                            meta.sessao_id,
                            Jsonb(payload),
                        ),
                    )

                total_indexado += len(lote)
                logger.debug(
                    f"pgvector: lote {i // batch_size + 1} indexado "
                    f"({len(lote)} chunks)"
                )
        finally:
            conn.close()

        logger.info(
            f"pgvector: {total_indexado} chunks indexados em '{colecao}'"
        )
        return total_indexado

    def busca_semantica(
        self,
        vetor_query: list[float],
        filtros: FiltrosBusca | None = None,
        nome_colecao: str | None = None,
        limite: int = 20,
    ) -> list[dict[str, Any]]:
        """Busca vetorial semântica no PostgreSQL com filtros opcionais.

        Usa operador ``<=>`` (distância cosseno) do pgvector com índice HNSW.
        Os filtros são aplicados via WHERE SQL — eficiente para as queries
        deste projeto (<100k chunks).

        Args:
            vetor_query: Vetor de embedding da query.
            filtros: Filtros de metadata para restringir a busca.
            nome_colecao: Tabela a buscar. Usa legislação se None.
            limite: Número máximo de resultados.

        Returns:
            Lista de dicionários com ``id``, ``score`` e ``payload``.
        """
        import numpy as np

        colecao = _validar_nome_tabela(nome_colecao or self.colecao_legislacao)
        vetor_np = np.array(vetor_query, dtype=np.float32)

        # Constrói condições WHERE parametrizadas
        condicoes: list[str] = []
        params: list[Any] = []

        if filtros:
            if filtros.vigencia is not None:
                condicoes.append("vigencia = %s")
                params.append(filtros.vigencia.value)
            if filtros.tipos:
                condicoes.append("tipo = ANY(%s)")
                params.append([t.value for t in filtros.tipos])
            if filtros.areas:
                condicoes.append("area = ANY(%s)")
                params.append([a.value for a in filtros.areas])
            if filtros.orgaos:
                condicoes.append("orgao = ANY(%s)")
                params.append(filtros.orgaos)
            if filtros.sessao_id:
                condicoes.append("sessao_id = %s")
                params.append(filtros.sessao_id)

        where = ("WHERE " + " AND ".join(condicoes)) if condicoes else ""

        sql = f"""
            SELECT id::text, payload, 1 - (vetor <=> %s) AS score
            FROM {colecao}
            {where}
            ORDER BY vetor <=> %s
            LIMIT %s
        """
        params_final = [vetor_np] + params + [vetor_np, limite]

        conn = self._get_conn()
        try:
            rows = conn.execute(sql, params_final).fetchall()
        finally:
            conn.close()

        return [
            {"id": row[0], "payload": dict(row[1]), "score": float(row[2])}
            for row in rows
        ]

    def verificar_conexao(self) -> bool:
        """Verifica se o PostgreSQL está acessível.

        Returns:
            True se conexão bem-sucedida, False caso contrário.
        """
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception as erro:
            logger.warning(f"PostgreSQL indisponível: {erro}")
            return False

    def listar_colecoes(self) -> list[str]:
        """Lista todas as tabelas no schema public.

        Returns:
            Lista de nomes de tabelas.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]
