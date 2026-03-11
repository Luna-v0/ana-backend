"""Módulo de indexação de chunks jurídicos no Qdrant.

Implementa a Etapa 4 do spec 02: indexação no Qdrant com metadata
rica para filtragem e busca híbrida.

Cada collection armazena chunks com:
- Vetores de embedding (1024 dims, distância cosseno)
- Payload com todos os campos de MetadataChunkJuridico
- IDs UUID únicos por chunk

Exemplo de uso:
    >>> from ana.rag.indexador import IndexadorQdrant
    >>> indexador = IndexadorQdrant()
    >>> indexador.criar_colecao_legislacao()
    >>> indexador.indexar_chunks(chunks)
"""

import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ana.config import obter_configuracao
from ana.config_modelos import obter_modelos
from ana.rag.modelos import ChunkJuridico, FiltrosBusca


class IndexadorQdrant:
    """Gerencia indexação e busca semântica de chunks jurídicos no Qdrant.

    Responsável por:
    - Criação e configuração de collections
    - Indexação em lote de chunks com embeddings
    - Busca vetorial com filtros de metadata

    Attributes:
        cliente: Instância do QdrantClient conectada ao Qdrant local.
        colecao_legislacao: Nome da collection global de legislação.
        dimensao_vetores: Dimensão dos vetores de embedding.
    """

    def __init__(self) -> None:
        """Inicializa o indexador com conexão ao Qdrant local."""
        config = obter_configuracao()
        config_modelos = obter_modelos().ativo.embeddings

        self.cliente = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            timeout=30,
        )
        self.colecao_legislacao = config.colecao_legislacao
        self.prefixo_sessao = config.prefixo_colecao_sessao
        self.dimensao_vetores = config_modelos.dimensao

    def _nome_colecao_sessao(self, sessao_id: str) -> str:
        """Retorna o nome da collection para uma sessão específica.

        Args:
            sessao_id: ID único da sessão de processo.

        Returns:
            Nome da collection no formato '{prefixo}_{sessao_id}'.
        """
        return f"{self.prefixo_sessao}_{sessao_id}"

    def criar_colecao(
        self,
        nome_colecao: str,
        recriar: bool = False,
    ) -> None:
        """Cria uma collection no Qdrant com configuração para chunks jurídicos.

        Configura:
        - Vetores: 1024 dims, distância cosseno (normalizado pelo e5-large)
        - Índices de payload para filtragem eficiente

        Args:
            nome_colecao: Nome da collection a criar.
            recriar: Se True, apaga e recria a collection se já existir.
        """
        colecoes_existentes = {
            c.name for c in self.cliente.get_collections().collections
        }

        if nome_colecao in colecoes_existentes:
            if recriar:
                logger.warning(f"Recriando collection: {nome_colecao}")
                self.cliente.delete_collection(nome_colecao)
            else:
                logger.info(f"Collection já existe: {nome_colecao}")
                return

        self.cliente.create_collection(
            collection_name=nome_colecao,
            vectors_config=VectorParams(
                size=self.dimensao_vetores,
                distance=Distance.COSINE,
            ),
        )

        # Cria índices de payload para filtragem eficiente na busca
        indices_payload = [
            ("tipo", PayloadSchemaType.KEYWORD),
            ("area", PayloadSchemaType.KEYWORD),
            ("vigencia", PayloadSchemaType.KEYWORD),
            ("orgao", PayloadSchemaType.KEYWORD),
            ("sessao_id", PayloadSchemaType.KEYWORD),
        ]
        for campo, tipo in indices_payload:
            self.cliente.create_payload_index(
                collection_name=nome_colecao,
                field_name=campo,
                field_schema=tipo,
            )

        logger.info(
            f"Collection criada: {nome_colecao} "
            f"(dims={self.dimensao_vetores}, distância=COSINE)"
        )

    def criar_colecao_legislacao(self, recriar: bool = False) -> None:
        """Cria a collection global de legislação brasileira.

        Args:
            recriar: Se True, apaga e recria a collection.
        """
        self.criar_colecao(self.colecao_legislacao, recriar=recriar)

    def criar_colecao_sessao(
        self,
        sessao_id: str,
        recriar: bool = False,
    ) -> str:
        """Cria collection isolada para uma sessão de processo.

        Cada processo tem sua própria collection para RAG isolado,
        conforme descrito no spec 05.

        Args:
            sessao_id: ID único da sessão.
            recriar: Se True, apaga e recria a collection.

        Returns:
            Nome da collection criada.
        """
        nome = self._nome_colecao_sessao(sessao_id)
        self.criar_colecao(nome, recriar=recriar)
        return nome

    def _chunk_para_payload(self, chunk: ChunkJuridico) -> dict[str, Any]:
        """Converte metadata do chunk para payload Qdrant.

        Args:
            chunk: Chunk jurídico com metadata completa.

        Returns:
            Dicionário plano com todos os campos serializáveis para o Qdrant.
        """
        meta = chunk.metadata
        payload: dict[str, Any] = {
            "texto": chunk.texto,
            "fonte": meta.fonte,
            "tipo": meta.tipo.value,
            "vigencia": meta.vigencia.value,
        }
        # Campos opcionais — só inclui se preenchidos
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
        """Indexa lista de chunks no Qdrant com seus embeddings.

        Os chunks devem ter o campo `embedding` preenchido antes de chamar
        este método. Use `GeradorEmbeddings.gerar_batch()` para isso.

        Args:
            chunks: Lista de chunks com embeddings já gerados.
            nome_colecao: Nome da collection. Usa legislação global se None.
            batch_size: Tamanho do batch para upsert (padrão: 100).

        Returns:
            Número de chunks indexados com sucesso.

        Raises:
            ValueError: Se algum chunk não tiver embedding gerado.
        """
        colecao = nome_colecao or self.colecao_legislacao

        # Valida embeddings
        sem_embedding = [i for i, c in enumerate(chunks) if c.embedding is None]
        if sem_embedding:
            raise ValueError(
                f"Chunks {sem_embedding[:5]} não têm embedding. "
                "Chame GeradorEmbeddings.gerar_batch() antes de indexar."
            )

        total_indexado = 0

        for i in range(0, len(chunks), batch_size):
            lote = chunks[i : i + batch_size]

            pontos = [
                PointStruct(
                    id=chunk.id or str(uuid.uuid4()),
                    vector=chunk.embedding,  # type: ignore[arg-type]
                    payload=self._chunk_para_payload(chunk),
                )
                for chunk in lote
            ]

            self.cliente.upsert(
                collection_name=colecao,
                points=pontos,
                wait=True,
            )
            total_indexado += len(lote)
            logger.debug(f"Indexado lote {i//batch_size + 1}: {len(lote)} chunks")

        logger.info(
            f"Indexação concluída: {total_indexado} chunks em '{colecao}'"
        )
        return total_indexado

    def busca_semantica(
        self,
        vetor_query: list[float],
        filtros: FiltrosBusca | None = None,
        nome_colecao: str | None = None,
        limite: int = 20,
    ) -> list[dict[str, Any]]:
        """Busca vetorial semântica no Qdrant com filtros opcionais.

        Args:
            vetor_query: Vetor de embedding da query (1024 dims).
            filtros: Filtros de metadata para restringir a busca.
            nome_colecao: Collection a buscar. Usa legislação se None.
            limite: Número máximo de resultados.

        Returns:
            Lista de dicionários com 'payload' e 'score' de cada resultado.
        """
        colecao = nome_colecao or self.colecao_legislacao
        filtro_qdrant = _construir_filtro_qdrant(filtros) if filtros else None

        resposta = self.cliente.query_points(
            collection_name=colecao,
            query=vetor_query,
            query_filter=filtro_qdrant,
            limit=limite,
            with_payload=True,
        )

        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in resposta.points
        ]

    def verificar_conexao(self) -> bool:
        """Verifica se o Qdrant está acessível.

        Returns:
            True se conexão bem-sucedida, False caso contrário.
        """
        try:
            self.cliente.get_collections()
            return True
        except Exception as erro:
            logger.warning(f"Qdrant indisponível: {erro}")
            return False

    def listar_colecoes(self) -> list[str]:
        """Lista todas as collections existentes no Qdrant.

        Returns:
            Lista de nomes de collections.
        """
        return [c.name for c in self.cliente.get_collections().collections]


def _construir_filtro_qdrant(filtros: FiltrosBusca) -> Filter:
    """Converte FiltrosBusca para filtro nativo do Qdrant.

    Args:
        filtros: Filtros definidos pela query do usuário ou agente.

    Returns:
        Objeto Filter do qdrant-client para uso na busca.
    """
    condicoes = []

    if filtros.tipos:
        condicoes.append(
            FieldCondition(
                key="tipo",
                match=MatchAny(any=[t.value for t in filtros.tipos]),
            )
        )

    if filtros.areas:
        condicoes.append(
            FieldCondition(
                key="area",
                match=MatchAny(any=[a.value for a in filtros.areas]),
            )
        )

    if filtros.vigencia is not None:
        condicoes.append(
            FieldCondition(
                key="vigencia",
                match=MatchValue(value=filtros.vigencia.value),
            )
        )

    if filtros.orgaos:
        condicoes.append(
            FieldCondition(
                key="orgao",
                match=MatchAny(any=filtros.orgaos),
            )
        )

    if filtros.sessao_id:
        condicoes.append(
            FieldCondition(
                key="sessao_id",
                match=MatchValue(value=filtros.sessao_id),
            )
        )

    return Filter(must=condicoes) if condicoes else Filter()
