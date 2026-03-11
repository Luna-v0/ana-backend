"""Módulo de geração de embeddings semânticos para o sistema RAG.

Implementa a Etapa 3 do spec 02: geração de embeddings com o modelo
intfloat/multilingual-e5-large via sentence-transformers.

O modelo roda 100% local (via HuggingFace cache), sem envio de dados
para APIs externas — conforme princípio LGPD.

Exemplo de uso:
    >>> from ana.rag.embeddings import GeradorEmbeddings
    >>> gerador = GeradorEmbeddings()
    >>> vetores = gerador.gerar_batch(["texto jurídico 1", "texto 2"])
"""

from functools import lru_cache

from loguru import logger

from ana.config_modelos import obter_modelos


class GeradorEmbeddings:
    """Gera embeddings semânticos usando multilingual-e5-large.

    Singleton por sessão: o modelo é carregado uma vez e reutilizado
    para todas as operações de embedding da aplicação.

    O modelo intfloat/multilingual-e5-large foi selecionado conforme
    spec 02 por:
    - Melhor desempenho open-source para português
    - 100% Top-5 accuracy em benchmarks de RAG
    - 1024 dimensões — compatível com Qdrant

    Attributes:
        modelo_nome: Nome do modelo no HuggingFace Hub.
        dispositivo: 'cuda' ou 'cpu'.
        batch_size: Tamanho do batch para geração em lote.
        dimensao: Dimensão dos vetores gerados (1024).
    """

    def __init__(
        self,
        modelo_nome: str | None = None,
        dispositivo: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Inicializa o gerador de embeddings.

        Args:
            modelo_nome: Nome do modelo. Usa config/modelos.yaml se None.
            dispositivo: Dispositivo de execução. Usa config se None.
            batch_size: Batch size. Usa config se None.
        """
        config_modelos = obter_modelos().ativo.embeddings
        self.modelo_nome = modelo_nome or config_modelos.modelo
        self.dispositivo = dispositivo or config_modelos.dispositivo
        self.batch_size = batch_size or config_modelos.batch_size
        self.dimensao = config_modelos.dimensao
        self._modelo = None

    def _carregar_modelo(self):
        """Carrega o modelo de embeddings (lazy loading).

        O modelo é carregado apenas na primeira chamada de geração,
        economizando memória quando o módulo RAG não é usado.

        Returns:
            Instância do SentenceTransformer carregada.
        """
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Carregando modelo de embeddings: {self.modelo_nome} "
                f"(dispositivo={self.dispositivo})"
            )
            self._modelo = SentenceTransformer(
                self.modelo_nome,
                device=self.dispositivo,
            )
            logger.info(
                f"Modelo carregado: dimensão={self.dimensao}, "
                f"batch_size={self.batch_size}"
            )
        return self._modelo

    def gerar(self, texto: str) -> list[float]:
        """Gera embedding para um único texto.

        O modelo e5-large espera prefixo 'query: ' para queries e
        'passage: ' para documentos. Esta função trata o texto como
        passagem (documento) para indexação.

        Args:
            texto: Texto para gerar embedding.

        Returns:
            Lista de floats com 1024 dimensões.
        """
        vetores = self.gerar_batch([texto])
        return vetores[0]

    def gerar_batch(self, textos: list[str]) -> list[list[float]]:
        """Gera embeddings para múltiplos textos em batch.

        O modelo e5-large usa prefixo 'passage: ' para documentos
        e 'query: ' para queries. Adiciona o prefixo automaticamente
        para indexação de documentos.

        Args:
            textos: Lista de textos para gerar embeddings.

        Returns:
            Lista de vetores (lista de 1024 floats cada).
        """
        if not textos:
            return []

        modelo = self._carregar_modelo()

        # e5-large requer prefixo 'passage: ' para documentos
        textos_prefixados = [f"passage: {t}" for t in textos]

        logger.debug(f"Gerando embeddings para {len(textos)} textos...")
        vetores = modelo.encode(
            textos_prefixados,
            batch_size=self.batch_size,
            show_progress_bar=len(textos) > 100,
            normalize_embeddings=True,  # Normalização para distância cosseno
            convert_to_numpy=True,
        )

        return [v.tolist() for v in vetores]

    def gerar_query(self, query: str) -> list[float]:
        """Gera embedding para uma query de busca.

        Usa o prefixo 'query: ' conforme especificação do e5-large,
        diferente do prefixo 'passage: ' usado para documentos.

        Args:
            query: Texto da query do usuário.

        Returns:
            Lista de floats com 1024 dimensões.
        """
        modelo = self._carregar_modelo()
        vetor = modelo.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vetor.tolist()


@lru_cache(maxsize=1)
def obter_gerador_embeddings() -> GeradorEmbeddings:
    """Retorna instância singleton do gerador de embeddings (com cache).

    Returns:
        Instância única de GeradorEmbeddings para o ciclo de vida da app.
    """
    return GeradorEmbeddings()
