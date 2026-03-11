"""Pipeline de validação anti-alucinação de 3 camadas (Spec 07).

Camadas:
    1. Extração — regex identifica citações legais no texto
    2. Lookup — dicionário SQLite verifica existência e vigência
    3. Semântica — SLM compara afirmação com texto real (opcional)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ana.validacao.dicionario import DicionarioLeis
from ana.validacao.extracao import extrair_citacoes


def validar_resposta(
    resposta_llm: str,
    usar_semantica: bool = True,
    dicionario: DicionarioLeis | None = None,
) -> list[dict[str, Any]]:
    """Valida todas as citações legais encontradas em uma resposta do LLM.

    Executa o pipeline de 3 camadas para cada citação extraída:
    1. Regex → citações candidatas
    2. Dicionário → verifica existência e vigência
    3. SLM → validação semântica (se ``usar_semantica=True``)

    Args:
        resposta_llm: Texto de resposta gerado pelo LLM.
        usar_semantica: Ativa a camada de validação semântica via SLM.
            Desativado por padrão em produção para latência.
        dicionario: Instância do dicionário. Cria nova se None.

    Returns:
        Lista de resultados por citação, cada um com:
        - ``citacao``: Texto original da citação.
        - ``tipo``: Tipo da citação (lei, artigo, etc.).
        - ``lei``: Número/nome da lei.
        - ``artigo``: Número do artigo (se aplicável).
        - ``status``: Status de validação (EXISTE_E_VIGENTE, etc.).
        - ``detalhe``: Mensagem explicativa.
        - ``semantica``: Resultado da validação semântica (se ativada).
    """
    if not resposta_llm.strip():
        return []

    # Camada 1: Extração de citações
    citacoes = extrair_citacoes(resposta_llm)
    if not citacoes:
        logger.debug("Validação: nenhuma citação legal encontrada")
        return []

    logger.info(f"Validação: {len(citacoes)} citações extraídas")

    dic = dicionario or DicionarioLeis()
    resultados: list[dict[str, Any]] = []

    for cit in citacoes:
        lei_ref = cit.get("lei") or cit.get("texto_original")
        artigo_ref = cit.get("artigo")

        # Camada 2: Lookup no dicionário
        if lei_ref:
            lookup = dic.validar_existencia(lei_ref, artigo_ref)
        else:
            lookup = {
                "status": "LEI_NAO_ENCONTRADA",
                "lei": cit["texto_original"],
                "artigo": artigo_ref,
                "detalhe": "Referência não mapeável para número de lei.",
            }

        resultado: dict[str, Any] = {
            "citacao": cit["texto_original"],
            "tipo": cit["tipo"],
            "lei": lookup.get("lei"),
            "artigo": artigo_ref,
            "status": lookup["status"],
            "detalhe": lookup["detalhe"],
            "semantica": None,
        }

        # Camada 3: Validação semântica (apenas se lei existe e vigente)
        if usar_semantica and lookup["status"] == "EXISTE_E_VIGENTE":
            try:
                texto_real = _buscar_texto_artigo(
                    lei_ref or "", artigo_ref
                )
                if texto_real:
                    from ana.validacao.semantica import validar_semantica
                    resultado["semantica"] = validar_semantica(
                        texto_afirmado=cit["texto_original"],
                        texto_real=texto_real,
                    )
            except Exception as e:
                logger.debug(f"Validação semântica ignorada: {e}")

        resultados.append(resultado)

    return resultados


def _buscar_texto_artigo(lei: str, artigo: str | None) -> str | None:
    """Recupera o texto real de um artigo do PostgreSQL.

    Args:
        lei: Referência à lei.
        artigo: Número do artigo.

    Returns:
        Texto do artigo ou None se não encontrado.
    """
    if not artigo:
        return None

    try:
        from ana.config import obter_configuracao
        from ana.storage.pgvector_store import IndexadorPgVector

        config = obter_configuracao()
        indexador = IndexadorPgVector()
        conn = indexador._get_conn()
        try:
            row = conn.execute(
                f"""
                SELECT texto FROM {config.colecao_legislacao}
                WHERE fonte ILIKE %s AND artigo ILIKE %s
                LIMIT 1
                """,
                (f"%{lei}%", f"%{artigo}%"),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None
    except Exception:
        return None
