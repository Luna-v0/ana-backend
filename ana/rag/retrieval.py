"""Pipeline de retrieval híbrido para o sistema RAG.

Implementa a seção 2.3 do spec 02: busca semântica + BM25 fundidos
com Reciprocal Rank Fusion (RRF), reranking com CrossEncoder e
Maximum Marginal Relevance (MMR) para diversidade.

Fluxo do retrieval:
    1. Busca semântica no PostgreSQL/pgvector (top-20)
    2. Busca BM25 sobre corpus em memória (top-20)
    3. Fusão com RRF (k=60)
    4. Reranking com CrossEncoder BAAI/bge-reranker-base
    5. MMR para diversidade nos resultados finais
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any


def _sigmoid(x: float) -> float:
    """Converte logit CrossEncoder para probabilidade [0, 1]."""
    return 1.0 / (1.0 + math.exp(-max(-88.0, min(88.0, x))))

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from ana.config_modelos import obter_modelos
from ana.rag.modelos import ChunkJuridico, FiltrosBusca, ResultadoBusca


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Funde múltiplos rankings via Reciprocal Rank Fusion.

    Conforme Cormack, Clarke & Buettcher (2009).
    Fórmula: RRF(d) = Σ 1/(k + rank(d)) para cada lista de ranking.

    Args:
        rankings: Lista de rankings (cada um é lista de IDs por relevância).
        k: Constante de suavização (60 é o valor canônico da literatura).

    Returns:
        Lista de (id, score_rrf) ordenada por score decrescente.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for posicao, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + posicao)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# Maximum Marginal Relevance
# =============================================================================

def maximum_marginal_relevance(
    vetor_query: list[float],
    candidatos: list[tuple[str, list[float]]],
    lambda_mmr: float = 0.5,
    top_k: int = 10,
) -> list[str]:
    """Seleciona top-k resultados com diversidade via MMR.

    MMR balanceia relevância com diversidade — evita retornar 10 artigos
    da mesma lei quando o usuário precisa de perspectivas múltiplas.

    Fórmula: MMR = argmax [λ·sim(d, q) - (1-λ)·max sim(d, dj)]
    onde dj são os documentos já selecionados.

    Args:
        vetor_query: Embedding da query do usuário.
        candidatos: Lista de (id, embedding) dos candidatos pós-reranking.
        lambda_mmr: Peso entre relevância (1.0) e diversidade (0.0).
        top_k: Número de resultados finais.

    Returns:
        Lista de IDs selecionados pelo critério MMR.
    """
    if not candidatos:
        return []

    if len(candidatos) <= top_k:
        return [cid for cid, _ in candidatos]

    q = np.array(vetor_query)
    embeddings = {cid: np.array(emb) for cid, emb in candidatos}
    ids_disponiveis = set(embeddings.keys())
    selecionados: list[str] = []

    # Normaliza query
    norma_q = np.linalg.norm(q)
    if norma_q > 0:
        q = q / norma_q

    while len(selecionados) < top_k and ids_disponiveis:
        melhor_id = None
        melhor_score = -math.inf

        for cid in ids_disponiveis:
            d = embeddings[cid]
            norma_d = np.linalg.norm(d)
            if norma_d > 0:
                d = d / norma_d

            # Relevância: similaridade com a query
            sim_query = float(np.dot(q, d))

            # Diversidade: máxima similaridade com já selecionados
            if selecionados:
                sims_sel = [
                    float(np.dot(d, embeddings[s] / (np.linalg.norm(embeddings[s]) or 1)))
                    for s in selecionados
                ]
                max_sim_sel = max(sims_sel)
            else:
                max_sim_sel = 0.0

            score_mmr = lambda_mmr * sim_query - (1 - lambda_mmr) * max_sim_sel

            if score_mmr > melhor_score:
                melhor_score = score_mmr
                melhor_id = cid

        if melhor_id:
            selecionados.append(melhor_id)
            ids_disponiveis.remove(melhor_id)

    return selecionados


# =============================================================================
# Índice BM25 em memória
# =============================================================================

class IndiceBM25:
    """Índice BM25 em memória sobre o corpus de chunks jurídicos.

    Construído na inicialização do sistema ou quando novos chunks são
    adicionados. Mantém mapeamento ID → texto para busca.

    Attributes:
        _indice: Instância do BM25Okapi.
        _ids: Lista de IDs na mesma ordem do corpus.
        _textos: Lista de textos do corpus.
    """

    def __init__(self) -> None:
        """Inicializa o índice BM25 vazio."""
        self._indice: BM25Okapi | None = None
        self._ids: list[str] = []
        self._textos: list[str] = []

    def construir(self, chunks: list[tuple[str, str]]) -> None:
        """Constrói o índice BM25 a partir de lista de (id, texto).

        Args:
            chunks: Lista de tuplas (id, texto) para indexação.
        """
        if not chunks:
            logger.warning("Tentativa de construir BM25 com lista vazia")
            return

        self._ids = [cid for cid, _ in chunks]
        self._textos = [texto for _, texto in chunks]
        corpus_tokenizado = [texto.lower().split() for texto in self._textos]
        self._indice = BM25Okapi(corpus_tokenizado)
        logger.info(f"Índice BM25 construído: {len(self._ids)} documentos")

    def buscar(self, query: str, top_n: int = 20) -> list[tuple[str, float]]:
        """Busca BM25 sobre o corpus.

        Args:
            query: Texto da query do usuário.
            top_n: Número máximo de resultados.

        Returns:
            Lista de (id, score_bm25) ordenada por score decrescente.
            Retorna lista vazia se o índice não foi construído.
        """
        if self._indice is None:
            logger.warning("Índice BM25 não construído. Chame construir() primeiro.")
            return []

        tokens_query = query.lower().split()
        scores = self._indice.get_scores(tokens_query)

        # Pares (id, score) ordenados decrescente, filtrando score > 0
        resultados = [
            (self._ids[i], float(scores[i]))
            for i in range(len(self._ids))
            if scores[i] > 0
        ]
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_n]

    @property
    def tamanho(self) -> int:
        """Retorna número de documentos no índice."""
        return len(self._ids)


# =============================================================================
# Reranker
# =============================================================================

class Reranker:
    """Reranker com CrossEncoder para reordenação por relevância.

    O CrossEncoder avalia pares (query, documento) e retorna score de
    relevância mais preciso que o embedding bi-encoder.

    Usa BAAI/bge-reranker-base por ser leve (~0.5GB VRAM) e eficaz.

    Attributes:
        modelo_nome: Nome do modelo CrossEncoder no HuggingFace Hub.
        dispositivo: Dispositivo de execução.
    """

    def __init__(
        self,
        modelo_nome: str | None = None,
        dispositivo: str | None = None,
    ) -> None:
        """Inicializa o reranker.

        Args:
            modelo_nome: Nome do modelo. Usa config/modelos.yaml se None.
            dispositivo: Dispositivo. Usa config se None.
        """
        config_reranker = obter_modelos().ativo.reranker
        self.modelo_nome = modelo_nome or config_reranker.modelo
        self.dispositivo = dispositivo or config_reranker.dispositivo
        self._modelo = None

    def _carregar_modelo(self):
        """Carrega o CrossEncoder (lazy loading)."""
        if self._modelo is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Carregando reranker: {self.modelo_nome}")
            self._modelo = CrossEncoder(
                self.modelo_nome,
                device=self.dispositivo,
            )
        return self._modelo

    def rerankar(
        self,
        query: str,
        candidatos: list[tuple[str, str]],
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """Reordena candidatos por relevância em relação à query.

        Args:
            query: Query do usuário.
            candidatos: Lista de (id, texto) para reranking.
            top_n: Retorna apenas os top-n. Retorna todos se None.

        Returns:
            Lista de (id, score_reranker) ordenada por score decrescente.
        """
        if not candidatos:
            return []

        modelo = self._carregar_modelo()
        pares = [(query, texto) for _, texto in candidatos]
        scores = modelo.predict(pares, show_progress_bar=False)

        resultados = [
            (candidatos[i][0], float(scores[i]))
            for i in range(len(candidatos))
        ]
        resultados.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            return resultados[:top_n]
        return resultados


# =============================================================================
# Pipeline de Retrieval Híbrido
# =============================================================================

class PipelineRetrieval:
    """Pipeline completo de retrieval híbrido para o sistema ANA.

    Combina busca semântica (Qdrant) + BM25 + RRF + Reranking + MMR
    conforme especificado na seção 2.3 do spec 02.

    Attributes:
        indexador: Backend de busca semântica (Qdrant ou pgvector).
        gerador_embeddings: Instância do GeradorEmbeddings.
        indice_bm25: Índice BM25 em memória.
        reranker: Instância do Reranker CrossEncoder.
    """

    def __init__(self) -> None:
        """Inicializa o pipeline de retrieval com seus componentes."""
        from ana.rag.embeddings import GeradorEmbeddings
        from ana.storage import obter_vector_store

        self.indexador = obter_vector_store()
        self.gerador_embeddings = GeradorEmbeddings()
        self.indice_bm25 = IndiceBM25()
        self.reranker = Reranker()

    def inicializar_bm25(self, nome_colecao: str | None = None) -> None:
        """Carrega textos do PostgreSQL e constrói o índice BM25 em memória.

        Deve ser chamado no startup da aplicação após o banco estar disponível.
        Executa uma query SELECT id, texto na tabela de legislação e alimenta
        o IndiceBM25 com os pares (id, texto) encontrados.

        Args:
            nome_colecao: Tabela a indexar. Usa legislação global se None.
        """
        from ana.config import obter_configuracao
        from ana.storage.pgvector_store import IndexadorPgVector

        if not isinstance(self.indexador, IndexadorPgVector):
            logger.info("BM25: backend não é pgvector — skipping inicialização automática")
            return

        config = obter_configuracao()
        colecao = nome_colecao or config.colecao_legislacao

        try:
            conn = self.indexador._get_conn()
            try:
                rows = conn.execute(
                    f"SELECT id::text, texto FROM {colecao} WHERE texto IS NOT NULL"
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"BM25: falha ao carregar textos de '{colecao}': {e}")
            return

        if not rows:
            logger.info(f"BM25: tabela '{colecao}' vazia — índice não construído")
            return

        self.indice_bm25.construir([(row[0], row[1]) for row in rows])

    def buscar(
        self,
        query: str,
        filtros: FiltrosBusca | None = None,
        nome_colecao: str | None = None,
        top_semantico: int = 20,
        top_bm25: int = 20,
        top_reranker: int = 15,
        top_final: int = 10,
        lambda_mmr: float = 0.5,
        usar_reranker: bool = True,
        usar_mmr: bool = True,
    ) -> list[dict[str, Any]]:
        """Executa o pipeline completo de busca híbrida.

        Etapas:
            1. Gera embedding da query
            2. Busca semântica no Qdrant (top_semantico)
            3. Busca BM25 (top_bm25)
            4. Fusão RRF dos dois rankings
            5. Reranking com CrossEncoder (se usar_reranker)
            6. MMR para diversidade (se usar_mmr)

        Args:
            query: Query do usuário em linguagem natural.
            filtros: Filtros de metadata (área, tipo, vigência, etc.).
            nome_colecao: Collection a buscar. Usa legislação se None.
            top_semantico: Candidatos da busca semântica.
            top_bm25: Candidatos da busca BM25.
            top_reranker: Candidatos após reranking.
            top_final: Resultados finais após MMR.
            lambda_mmr: Peso relevância/diversidade no MMR.
            usar_reranker: Ativa reranking com CrossEncoder.
            usar_mmr: Ativa diversidade com MMR.

        Returns:
            Lista de dicionários com 'id', 'score', 'payload' dos chunks,
            ordenados por relevância combinada.
        """
        logger.debug(f"Busca híbrida: '{query[:60]}...' " if len(query) > 60 else f"Busca híbrida: '{query}'")

        # 1. Embedding da query
        vetor_query = self.gerador_embeddings.gerar_query(query)

        # 2. Busca semântica
        resultados_sem = self.indexador.busca_semantica(
            vetor_query=vetor_query,
            filtros=filtros,
            nome_colecao=nome_colecao,
            limite=top_semantico,
        )
        ranking_sem = [r["id"] for r in resultados_sem]
        mapa_payloads = {r["id"]: r["payload"] for r in resultados_sem}

        # 3. Busca BM25
        resultados_bm25 = self.indice_bm25.buscar(query, top_n=top_bm25)
        ranking_bm25 = [r[0] for r in resultados_bm25]

        # Adiciona payloads BM25 ao mapa (se não presentes da busca semântica)
        for cid, _ in resultados_bm25:
            if cid not in mapa_payloads:
                mapa_payloads[cid] = {"id": cid}

        # 4. Fusão RRF
        fusao = reciprocal_rank_fusion([ranking_sem, ranking_bm25])
        ids_fundidos = [cid for cid, _ in fusao]
        scores_rrf = dict(fusao)

        if not ids_fundidos:
            logger.warning("Nenhum resultado encontrado na busca híbrida")
            return []

        # 5. Reranking com CrossEncoder
        if usar_reranker and ids_fundidos:
            candidatos_reranker = [
                (cid, mapa_payloads.get(cid, {}).get("texto", ""))
                for cid in ids_fundidos[:top_reranker]
                if mapa_payloads.get(cid, {}).get("texto")
            ]
            if candidatos_reranker:
                reranked = self.reranker.rerankar(
                    query=query,
                    candidatos=candidatos_reranker,
                    top_n=top_reranker,
                )
                ids_fundidos = [cid for cid, _ in reranked]
                # Normaliza logits do CrossEncoder para [0, 1] via sigmoid
                scores_rrf = {cid: _sigmoid(score) for cid, score in reranked}

        # 6. MMR para diversidade — com embeddings reais dos candidatos
        if usar_mmr and ids_fundidos:
            textos_mmr = [
                mapa_payloads.get(cid, {}).get("texto") or query
                for cid in ids_fundidos
            ]
            embeddings_mmr = self.gerador_embeddings.gerar_batch(textos_mmr)
            candidatos_mmr = list(zip(ids_fundidos, embeddings_mmr))
            ids_finais = maximum_marginal_relevance(
                vetor_query=vetor_query,
                candidatos=candidatos_mmr,
                lambda_mmr=lambda_mmr,
                top_k=top_final,
            )
        else:
            ids_finais = ids_fundidos[:top_final]

        # Monta resultado final
        resultado = [
            {
                "id": cid,
                "score": scores_rrf.get(cid, 0.0),
                "payload": mapa_payloads.get(cid, {}),
            }
            for cid in ids_finais
            if cid in mapa_payloads
        ]

        logger.info(
            f"Retrieval: {len(resultado)} resultados para '{query[:40]}...'"
            if len(query) > 40 else
            f"Retrieval: {len(resultado)} resultados para '{query}'"
        )
        return resultado


@lru_cache(maxsize=1)
def obter_pipeline_retrieval() -> PipelineRetrieval:
    """Retorna instância singleton do pipeline de retrieval (com cache).

    Returns:
        Instância única de PipelineRetrieval para o ciclo de vida da app.
    """
    return PipelineRetrieval()
