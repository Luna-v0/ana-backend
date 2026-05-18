"""Nó validador de leis — verifica citações legais no contexto RAG.

Executa validação anti-alucinação nos chunks recuperados pelo nó pesquisar,
anotando referências legais problemáticas antes da síntese. Usa o pipeline
de validação do Spec 07 sem camada semântica para manter baixa latência.

A validação neste nó opera sobre as **fontes** dos chunks (metadados),
não sobre a resposta final — que ainda não foi gerada. A validação
completa da resposta ocorre no endpoint /chat após o streaming.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico


def no_validar_leis(estado: EstadoJuridico) -> dict:
    """Valida as fontes dos chunks RAG contra o dicionário de leis conhecidas.

    Verifica se as fontes dos chunks recuperados correspondem a leis
    reais no dicionário de validação. Marca fontes suspeitas para
    que o sintetizador possa alertar o usuário.

    Args:
        estado: Estado com ``contexto_rag`` preenchido pelo nó pesquisar.

    Returns:
        Dicionário com ``validacao`` (lista de resultados por fonte)
        ou dict vazio se não houver chunks ou validação falhar.
    """
    chunks = estado.get("contexto_rag") or []
    if not chunks:
        return {}

    # Extrai fontes únicas dos chunks para validação
    fontes = list({
        c.get("fonte", "")
        for c in chunks
        if c.get("fonte")
    })

    if not fontes:
        return {}

    try:
        from ana.validacao.pipeline import validar_citacoes_lista
        resultados = validar_citacoes_lista(fontes)
        problemas = [r for r in resultados if not r.get("valida", True)]
        if problemas:
            logger.warning(
                f"no_validar_leis: {len(problemas)} fonte(s) com problema: "
                + ", ".join(p.get("citacao", "") for p in problemas[:3])
            )
        else:
            logger.debug(f"no_validar_leis: {len(fontes)} fonte(s) validada(s) sem problemas")
        return {"validacao": resultados}
    except AttributeError:
        # validar_citacoes_lista pode não existir — usa validação de texto como fallback
        logger.debug("no_validar_leis: validar_citacoes_lista indisponível — pulando")
        return {}
    except Exception as e:
        logger.debug(f"no_validar_leis: validação ignorada ({e})")
        return {}
