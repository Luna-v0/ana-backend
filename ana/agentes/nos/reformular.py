"""Nó reformulador jurídico — enriquece o prompt com instruções de redação formal.

Recebe o ``prompt_sintese`` montado pelos nós pesquisar/analisar e cria
um ``prompt_reformulacao`` enriquecido com instruções de português jurídico
brasileiro formal. O LLM pesquisador, ao receber este prompt aprimorado,
produz uma resposta em linguagem técnica precisa — dispensando uma segunda
chamada de reformulação.

Esta abordagem (prompt enrichment) é preferível a duas chamadas LLM em série
por razões de latência e custo — o modelo faz síntese + reformulação em
uma única passagem.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico
from ana.agentes.prompts import prompt_reformulacao


def no_reformular(estado: EstadoJuridico) -> dict:
    """Enriquece o prompt de síntese com instruções de redação jurídica formal.

    Se não houver ``prompt_sintese``, retorna o estado sem modificação
    (o nó ``gerar_resposta`` tratará o caso de intenção sem contexto).

    Args:
        estado: Estado atual com ``prompt_sintese`` (opcional) e
            ``mensagem_usuario``.

    Returns:
        Dicionário com ``prompt_reformulacao`` (prompt enriquecido) se
        ``prompt_sintese`` estiver presente; dict vazio caso contrário.
    """
    prompt_base = estado.get("prompt_sintese")

    if not prompt_base:
        logger.debug("no_reformular: sem prompt_sintese — sem modificação")
        return {}

    prompt_enriquecido = prompt_reformulacao(prompt_base)
    logger.debug(
        f"no_reformular: prompt enriquecido ({len(prompt_enriquecido)} chars)"
    )
    return {"prompt_reformulacao": prompt_enriquecido}
