"""Nó pesquisador de legislação — busca RAG híbrida na base jurídica.

Executa o pipeline RAG completo (embedding → pgvector → BM25 → RRF →
CrossEncoder → MMR) para recuperar os chunks mais relevantes da
legislação indexada.

O nó NÃO chama o LLM para síntese — apenas recupera o contexto e
monta o ``prompt_sintese`` para streaming posterior no endpoint ``/chat``.
Isso permite streaming de tokens com baixa latência para o usuário.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico
from ana.agentes.prompts import prompt_pesquisa_legal


def no_pesquisar_legislacao(estado: EstadoJuridico) -> dict:
    """Recupera chunks relevantes da legislação via pipeline RAG híbrido.

    Executa busca semântica + BM25 com Reciprocal Rank Fusion, reranking
    com CrossEncoder e diversificação via MMR, conforme Spec 02.

    Monta o ``prompt_sintese`` a partir dos chunks recuperados, pronto
    para uso no endpoint de streaming sem nova chamada ao grafo.

    Args:
        estado: Estado com ``mensagem_usuario`` para usar como query RAG.

    Returns:
        Dicionário com ``contexto_rag`` (lista de chunks) e
        ``prompt_sintese`` (string de prompt para o LLM pesquisador).
    """
    mensagem = estado.get("mensagem_usuario", "")
    transcricao = estado.get("transcricao_anexada")
    documento = estado.get("documento_processo")

    # Para documentos de processo: usa o texto do documento como query principal (lei discovery)
    # Para transcrições: enriquece a query da mensagem com termos da transcrição
    if documento:
        query = f"{mensagem}\n\n{documento[:800]}"
    elif transcricao:
        query = f"{mensagem}\n\nContexto: {transcricao[:500]}"
    else:
        query = mensagem

    try:
        from ana.rag.retrieval import obter_pipeline_retrieval

        pipeline = obter_pipeline_retrieval()
        resultados = pipeline.buscar(
            query=query,
            top_final=8,
            usar_reranker=True,
            usar_mmr=True,
        )

        logger.info(f"no_pesquisar_legislacao: {len(resultados)} chunks recuperados")

        # Normaliza para formato uniforme de contexto — inclui todos os campos de metadata
        # para que o frontend possa exibir os artigos completos (hierarquia, vigência, etc.)
        chunks = [
            {
                "texto":    r["payload"].get("texto", ""),
                "fonte":    r["payload"].get("fonte", ""),
                "artigo":   r["payload"].get("artigo", ""),
                "area":     r["payload"].get("area"),
                "vigencia": r["payload"].get("vigencia"),
                "titulo":   r["payload"].get("titulo"),
                "capitulo": r["payload"].get("capitulo"),
                "secao":    r["payload"].get("secao"),
                "score":    round(r.get("score", 0.0), 4),
            }
            for r in resultados
            if r["payload"].get("texto")
        ]

        # Quando há documento de processo, passa-o como contexto extra ao prompt
        contexto_extra = documento or transcricao
        return {
            "contexto_rag": chunks,
            "prompt_sintese": prompt_pesquisa_legal(mensagem, chunks, contexto_extra),
        }

    except Exception as e:
        logger.error(f"no_pesquisar_legislacao: erro no RAG: {e}")
        contexto_extra = documento or transcricao
        return {
            "contexto_rag": [],
            "prompt_sintese": prompt_pesquisa_legal(mensagem, [], contexto_extra),
            "erro": f"Falha na busca RAG: {e}",
        }
