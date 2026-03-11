"""Router de endpoints RAG do sistema ANA.

Expõe endpoints para:
- Ingestão de documentos jurídicos
- Busca híbrida (semântica + BM25)
- Status do pipeline RAG
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ana.rag.modelos import (
    AreaJuridica,
    FiltrosBusca,
    TipoDocumento,
    VigenciaStatus,
)

router = APIRouter(prefix="/rag", tags=["RAG"])


# =============================================================================
# Modelos de request/response
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
    area: AreaJuridica | None = Field(
        default=None,
        description="Área do direito",
    )
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
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Número máximo de resultados",
    )
    usar_reranker: bool = Field(
        default=True,
        description="Ativa reranking com CrossEncoder",
    )
    usar_mmr: bool = Field(
        default=True,
        description="Ativa diversidade com MMR",
    )
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
        titulo: Título da hierarquia legal (ex: 'TÍTULO II Dos Direitos').
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


class RequisicaoResumir(BaseModel):
    """Requisição para resumo gerado pelo SLM.

    Attributes:
        query: Pergunta original do usuário.
        chunks: Lista de trechos de lei recuperados para contexto.
    """

    query: str = Field(description="Pergunta original do usuário")
    chunks: list[ResultadoChunk] = Field(
        description="Artigos recuperados usados como contexto"
    )


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


# =============================================================================
# Endpoints
# =============================================================================

@router.get(
    "/status",
    response_model=StatusRAG,
    summary="Status do pipeline RAG",
)
async def status_rag() -> StatusRAG:
    """Retorna o status atual do pipeline RAG.

    Returns:
        StatusRAG com informações do PostgreSQL, BM25 e modelos ativos.
    """
    from ana.storage.pgvector_store import IndexadorPgVector
    from ana.rag.retrieval import obter_pipeline_retrieval
    from ana.config_modelos import obter_modelos

    try:
        indexador = IndexadorPgVector()
        pg_ok = indexador.verificar_conexao()
        colecoes = indexador.listar_colecoes() if pg_ok else []
    except Exception:
        pg_ok = False
        colecoes = []

    pipeline = obter_pipeline_retrieval()
    config_modelos = obter_modelos()

    return StatusRAG(
        postgres_disponivel=pg_ok,
        colecoes=colecoes,
        indice_bm25_tamanho=pipeline.indice_bm25.tamanho,
        modelo_embeddings=config_modelos.ativo.embeddings.modelo,
        perfil_modelos=config_modelos.perfil_ativo,
    )


@router.post(
    "/ingerir",
    response_model=RespostaIngestao,
    summary="Ingerir documento jurídico",
    description=(
        "Processa texto de lei em chunks jurídicos e indexa no Qdrant. "
        "O chunking respeita a estrutura: Lei → Título → Capítulo → Artigo."
    ),
)
async def ingerir_documento(requisicao: RequiscaoIngestao) -> RespostaIngestao:
    """Ingere documento jurídico no pipeline RAG.

    Args:
        requisicao: Texto e metadata do documento a ingerir.

    Returns:
        RespostaIngestao com contagem de chunks gerados e indexados.

    Raises:
        HTTPException: Se Qdrant não estiver disponível ou ocorrer erro.
    """
    from ana.rag.ingestao import processar_documento
    from ana.rag.embeddings import GeradorEmbeddings
    from ana.storage import obter_vector_store

    try:
        # 1. Chunking jurídico
        chunks = processar_documento(
            texto=requisicao.texto,
            fonte=requisicao.fonte,
            tipo=requisicao.tipo,
            area=requisicao.area,
            vigencia=requisicao.vigencia,
            orgao=requisicao.orgao,
            url_origem=requisicao.url_origem,
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Nenhum artigo encontrado no texto de '{requisicao.fonte}'. "
                    "Verifique se o documento contém artigos no formato 'Art. X'."
                ),
            )

        # 2. Gerar embeddings
        gerador = GeradorEmbeddings()
        textos = [c.texto for c in chunks]
        embeddings = gerador.gerar_batch(textos)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 3. Indexar no PostgreSQL
        indexador = obter_vector_store()
        if not indexador.verificar_conexao():
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL não disponível. Verifique se o serviço está em execução.",
            )

        indexador.criar_colecao_legislacao()
        total_indexado = indexador.indexar_chunks(chunks)

        return RespostaIngestao(
            chunks_gerados=len(chunks),
            chunks_indexados=total_indexado,
            fonte=requisicao.fonte,
        )

    except HTTPException:
        raise
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na ingestão: {str(erro)}",
        ) from erro


@router.post(
    "/buscar",
    response_model=RespostaBusca,
    summary="Busca híbrida na legislação",
    description=(
        "Executa busca semântica + BM25 com RRF, reranking e MMR "
        "sobre a legislação brasileira indexada."
    ),
)
async def buscar(requisicao: RequiscaoBusca) -> RespostaBusca:
    """Executa busca híbrida no pipeline RAG.

    Args:
        requisicao: Query, filtros e parâmetros de busca.

    Returns:
        RespostaBusca com chunks mais relevantes ordenados por score.

    Raises:
        HTTPException: Se Qdrant não estiver disponível.
    """
    from ana.rag.retrieval import obter_pipeline_retrieval

    try:
        pipeline = obter_pipeline_retrieval()

        resultados = pipeline.buscar(
            query=requisicao.query,
            filtros=requisicao.filtros,
            top_final=requisicao.top_k,
            usar_reranker=requisicao.usar_reranker,
            usar_mmr=requisicao.usar_mmr,
        )

        chunks_resultado = [
            ResultadoChunk(
                id=r["id"],
                score=r["score"],
                texto=r["payload"].get("texto", ""),
                fonte=r["payload"].get("fonte", ""),
                artigo=r["payload"].get("artigo"),
                area=r["payload"].get("area"),
                vigencia=r["payload"].get("vigencia"),
                titulo=r["payload"].get("titulo"),
                capitulo=r["payload"].get("capitulo"),
                secao=r["payload"].get("secao"),
            )
            for r in resultados
        ]

        # Validação anti-alucinação (opcional)
        validacao = None
        if requisicao.validar and chunks_resultado:
            try:
                from ana.validacao import validar_resposta
                # Monta texto de resposta com os chunks mais relevantes
                texto_resposta = " ".join(
                    f"{c.fonte} {c.artigo or ''}: {c.texto}"
                    for c in chunks_resultado[:5]
                )
                validacao = validar_resposta(texto_resposta, usar_semantica=False)
            except Exception as e:
                validacao = [{"erro": str(e)}]

        return RespostaBusca(
            query=requisicao.query,
            total=len(chunks_resultado),
            resultados=chunks_resultado,
            validacao=validacao,
        )

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na busca: {str(erro)}",
        ) from erro


@router.post(
    "/resumir",
    summary="Resumo dos artigos via SLM (streaming)",
    description=(
        "Recebe a query e os artigos recuperados e retorna um resumo gerado "
        "pelo SLM configurado (pesquisador), via server-sent events."
    ),
    response_class=StreamingResponse,
)
async def resumir(requisicao: RequisicaoResumir) -> StreamingResponse:
    """Gera resumo jurídico dos artigos recuperados via SLM em streaming.

    O cliente deve consumir a resposta como text/plain chunked ou SSE.
    Cada chunk de texto é entregue conforme o modelo produz tokens.

    Args:
        requisicao: Query original e lista de chunks recuperados.

    Returns:
        StreamingResponse com texto plain gerado incrementalmente.
    """
    from ana.config import obter_configuracao
    from ana.config_modelos import obter_modelos
    from ana.providers.llm import OllamaLLMProvider

    config = obter_configuracao()
    modelos = obter_modelos()
    modelo = modelos.ativo.agentes.pesquisador

    # Monta contexto com os artigos mais relevantes (até 8 para não estourar o contexto)
    LIMITE_CONTEXTO = 8
    chunks = requisicao.chunks[:LIMITE_CONTEXTO]

    partes_contexto: list[str] = []
    for c in chunks:
        hier = " › ".join(h for h in [c.titulo, c.capitulo, c.secao] if h)
        cabecalho = f"[{c.fonte}{' — ' + c.artigo if c.artigo else ''}]"
        if hier:
            cabecalho += f" ({hier})"
        partes_contexto.append(f"{cabecalho}\n{c.texto}")

    contexto = "\n\n".join(partes_contexto)

    prompt = (
        "Você é um assistente jurídico brasileiro especializado em legislação. "
        "Com base apenas nos artigos de lei abaixo, responda de forma objetiva e precisa "
        "à pergunta do usuário. Cite as fontes (lei e artigo) ao longo da resposta. "
        "Se os artigos não forem suficientes para responder, diga isso claramente. "
        "Seja conciso: máximo 3 parágrafos.\n\n"
        f"PERGUNTA: {requisicao.query}\n\n"
        f"ARTIGOS RELEVANTES:\n{contexto}\n\n"
        "RESPOSTA:"
    )

    llm = OllamaLLMProvider(modelo=modelo, host=config.ollama_host)

    def gerar():
        try:
            for token in llm.invocar_stream(prompt, temperatura=0.2):
                yield token
        except Exception as e:
            yield f"\n\n[Erro ao gerar resumo: {e}]"

    return StreamingResponse(gerar(), media_type="text/plain; charset=utf-8")
