"""Router de endpoints RAG do sistema ANA.

Expõe endpoints para:
- Ingestão de documentos jurídicos
- Busca híbrida (semântica + BM25)
- Status do pipeline RAG
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ana.api.schemas.rag import (
    RequiscaoBusca,
    RequiscaoIngestao,
    RequisicaoResumir,
    RespostaBusca,
    RespostaIngestao,
    ResultadoChunk,
    StatusRAG,
)

router = APIRouter(prefix="/rag", tags=["RAG"])


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
        "Processa texto de lei em chunks jurídicos e indexa no PostgreSQL. "
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
        HTTPException 422: Se nenhum artigo for encontrado no texto.
        HTTPException 503: Se o PostgreSQL não estiver disponível.
        HTTPException 500: Se ocorrer erro durante a ingestão.
    """
    try:
        import asyncio
        from ana.rag.ingestao import processar_documento
        from ana.storage import obter_vector_store
        from ana.gpu import obter_gestor

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

        gestor = obter_gestor()
        async with gestor.usar("embeddings") as gerador:
            textos = [c.texto for c in chunks]
            embeddings = await asyncio.to_thread(gerador.gerar_batch, textos)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        indexador = obter_vector_store()
        if not await asyncio.to_thread(indexador.verificar_conexao):
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL não disponível. Verifique se o serviço está em execução.",
            )

        await asyncio.to_thread(indexador.criar_colecao_legislacao)
        total_indexado = await asyncio.to_thread(indexador.indexar_chunks, chunks)

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
        HTTPException 500: Se ocorrer erro durante a busca.
    """
    from ana.rag.retrieval import obter_pipeline_retrieval

    try:
        import asyncio
        from ana.gpu import obter_gestor

        pipeline = obter_pipeline_retrieval()
        gestor = obter_gestor()

        async with gestor.usar("embeddings"):
            async with gestor.usar("reranker"):
                resultados = await asyncio.to_thread(
                    pipeline.buscar,
                    requisicao.query,
                    requisicao.filtros,
                    None,  # nome_colecao
                    20,    # top_semantico
                    20,    # top_bm25
                    15,    # top_reranker
                    requisicao.top_k,
                    0.5,   # lambda_mmr
                    requisicao.usar_reranker,
                    requisicao.usar_mmr,
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

        validacao = None
        if requisicao.validar and chunks_resultado:
            try:
                from ana.validacao import validar_resposta
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
