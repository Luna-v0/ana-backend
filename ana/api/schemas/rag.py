"""Contratos de input/output para o router /rag."""

from pydantic import BaseModel, Field

from ana.rag.modelos import AreaJuridica, FiltrosBusca, TipoDocumento, VigenciaStatus


# =============================================================================
# Ingestão
# =============================================================================

class RequiscaoIngestao(BaseModel):
    """Requisição para ingestão de texto de lei.

    Attributes:
        texto: Conteúdo textual do documento para chunking.
        fonte: Identificação da fonte (ex: 'Lei 13.709/2018 (LGPD)').
        tipo: Tipo do documento jurídico.
        area: Área do direito (opcional).
        vigencia: Status de vigência.
        orgao: Órgão emissor (opcional).
        url_origem: URL da fonte (opcional).
    """

    texto: str = Field(description="Texto completo do documento para ingestão")
    fonte: str = Field(description="Identificação da fonte")
    tipo: TipoDocumento = Field(
        default=TipoDocumento.LEI_FEDERAL,
        description="Tipo do documento jurídico",
    )
    area: AreaJuridica | None = Field(default=None, description="Área do direito")
    vigencia: VigenciaStatus = Field(
        default=VigenciaStatus.ATIVA,
        description="Status de vigência",
    )
    orgao: str | None = Field(default=None, description="Órgão emissor")
    url_origem: str | None = Field(default=None, description="URL da fonte")


class RespostaIngestao(BaseModel):
    """Resposta da ingestão de documento.

    Attributes:
        chunks_gerados: Número de chunks criados.
        chunks_indexados: Número de chunks indexados no PostgreSQL.
        fonte: Identificação da fonte processada.
    """

    chunks_gerados: int
    chunks_indexados: int
    fonte: str


# =============================================================================
# Busca
# =============================================================================

class RequiscaoBusca(BaseModel):
    """Requisição de busca híbrida no RAG.

    Attributes:
        query: Pergunta ou termo de busca jurídica.
        filtros: Filtros de metadata para restringir a busca.
        top_k: Número máximo de resultados.
        usar_reranker: Ativa reranking com CrossEncoder.
        usar_mmr: Ativa diversidade com MMR.
        validar: Ativa validação anti-alucinação das leis citadas na resposta.
    """

    query: str = Field(description="Pergunta ou termo de busca jurídica")
    filtros: FiltrosBusca = Field(
        default_factory=FiltrosBusca,
        description="Filtros de metadata",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Número máximo de resultados")
    usar_reranker: bool = Field(default=True, description="Ativa reranking com CrossEncoder")
    usar_mmr: bool = Field(default=True, description="Ativa diversidade com MMR")
    validar: bool = Field(
        default=False,
        description="Ativa validação anti-alucinação das leis citadas",
    )


class ResultadoChunk(BaseModel):
    """Resultado de busca: um chunk recuperado.

    Attributes:
        id: ID único do chunk no PostgreSQL.
        score: Score de relevância (RRF ou reranker).
        texto: Conteúdo textual do chunk.
        fonte: Identificação da fonte.
        artigo: Artigo específico (ex: 'Art. 5').
        area: Área jurídica.
        vigencia: Status de vigência.
        titulo: Título da hierarquia legal.
        capitulo: Capítulo da hierarquia legal.
        secao: Seção da hierarquia legal.
    """

    id: str
    score: float
    texto: str
    fonte: str
    artigo: str | None = None
    area: str | None = None
    vigencia: str | None = None
    titulo: str | None = None
    capitulo: str | None = None
    secao: str | None = None


class RespostaBusca(BaseModel):
    """Resposta da busca híbrida.

    Attributes:
        query: Query original do usuário.
        total: Total de chunks retornados.
        resultados: Lista de chunks relevantes.
        validacao: Resultado da validação anti-alucinação (apenas se validar=True).
    """

    query: str
    total: int
    resultados: list[ResultadoChunk]
    validacao: list[dict] | None = None


# =============================================================================
# Resumo (streaming — sem response_model)
# =============================================================================

class RequisicaoResumir(BaseModel):
    """Requisição para resumo gerado pelo SLM.

    Attributes:
        query: Pergunta original do usuário.
        chunks: Lista de trechos de lei recuperados para contexto.
    """

    query: str = Field(description="Pergunta original do usuário")
    chunks: list[ResultadoChunk] = Field(
        description="Artigos recuperados usados como contexto",
    )


# =============================================================================
# Status
# =============================================================================

class StatusRAG(BaseModel):
    """Status do pipeline RAG.

    Attributes:
        postgres_disponivel: True se PostgreSQL está acessível.
        colecoes: Lista de tabelas existentes no banco.
        indice_bm25_tamanho: Número de documentos no índice BM25.
        modelo_embeddings: Nome do modelo de embeddings ativo.
        perfil_modelos: Perfil de modelos ativo.
    """

    postgres_disponivel: bool
    colecoes: list[str]
    indice_bm25_tamanho: int
    modelo_embeddings: str
    perfil_modelos: str
