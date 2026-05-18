"""Nó analista de processo — busca semântica nos documentos da sessão.

Recupera chunks dos documentos vinculados à sessão ativa (processo)
usando pgvector sobre a coleção isolada por ``sessao_id``.

Cobre dois casos de uso:
- ``analise_processo``: perguntas sobre fatos, partes, documentos do processo
- ``verificar_prazo``: busca de menções a datas e prazos nos documentos

Conforme Spec 06 (Sessões), cada sessão possui uma coleção de vetores
isolada — ``processos_{sessao_id}`` — garantindo que os documentos de
um processo não contaminem a busca de outro.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico
from ana.agentes.prompts import prompt_analise_processo


def no_analisar_processo(estado: EstadoJuridico) -> dict:
    """Busca chunks nos documentos da sessão (processo ativo) via pgvector.

    Usa embeddings para busca semântica nos documentos indexados na
    sessão identificada por ``sessao_id``. Se a sessão não tiver
    documentos indexados, retorna prompt orientativo ao usuário.

    Args:
        estado: Estado com ``mensagem_usuario`` e ``sessao_id``.

    Returns:
        Dicionário com ``contexto_rag`` (chunks do processo) e
        ``prompt_sintese`` (string de prompt para o modelo analista).
    """
    mensagem = estado.get("mensagem_usuario", "")
    sessao_id = estado.get("sessao_id", "")
    transcricao = estado.get("transcricao_anexada")

    numero_processo = _obter_numero_processo(sessao_id)

    try:
        from ana.rag.embeddings import obter_gerador_embeddings
        from ana.storage.pgvector_store import IndexadorPgVector

        gerador = obter_gerador_embeddings()
        vetor_query = gerador.gerar_query(mensagem)

        indexador = IndexadorPgVector()
        # Coleção isolada por sessão conforme Spec 06
        nome_colecao = f"processos_{sessao_id.replace('-', '_')}"

        resultados = indexador.busca_semantica(
            vetor_query=vetor_query,
            nome_colecao=nome_colecao,
            limite=6,
        )

        logger.info(
            f"no_analisar_processo: sessao={sessao_id} → {len(resultados)} chunks"
        )

        chunks = [
            {
                "texto": r["payload"].get("texto", ""),
                "fonte": r["payload"].get("nome", r["payload"].get("fonte", "Documento")),
                "score": round(r.get("score", 0.0), 4),
            }
            for r in resultados
            if r["payload"].get("texto")
        ]

        return {
            "contexto_rag": chunks,
            "prompt_sintese": prompt_analise_processo(
                mensagem, chunks, numero_processo, transcricao
            ),
        }

    except Exception as e:
        logger.warning(f"no_analisar_processo: erro na busca: {e}")
        return {
            "contexto_rag": [],
            "prompt_sintese": prompt_analise_processo(
                mensagem, [], numero_processo, transcricao
            ),
            "erro": f"Falha na análise do processo: {e}",
        }


def _obter_numero_processo(sessao_id: str) -> str:
    """Recupera o número do processo a partir do SQLite de sessões.

    Args:
        sessao_id: ID da sessão ativa.

    Returns:
        Número do processo ou string vazia se não encontrado.
    """
    if not sessao_id:
        return ""
    try:
        from ana.sessoes.repositorio import obter_sessao
        sessao = obter_sessao(sessao_id)
        return sessao.numero_processo if sessao else ""
    except Exception:
        return ""
