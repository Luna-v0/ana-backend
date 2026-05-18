"""Nó classificador de intenção — Orquestrador do grafo LangGraph.

Usa um SLM rápido (``agentes.orquestrador`` do perfil ativo) para
classificar a mensagem do usuário em uma das intenções reconhecidas
pelo sistema ANA.

A classificação é feita via prompt de instrução única, sem histórico
de conversa, para minimizar latência. O modelo retorna apenas a
categoria, sem explicações.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico, INTENCOES_VALIDAS
from ana.agentes.prompts import prompt_classificacao


def no_classificar(estado: EstadoJuridico) -> dict:
    """Classifica a intenção do usuário usando o modelo orquestrador.

    Chama o LLM (SLM rápido) com um prompt de classificação de categoria
    única. Se a resposta não corresponder a uma intenção válida, usa
    ``"desconhecida"`` como fallback seguro.

    Args:
        estado: Estado atual do grafo com ``mensagem_usuario`` preenchida.

    Returns:
        Dicionário com ``intencao`` classificada (string da categoria).

    Example:
        >>> estado = {"mensagem_usuario": "Qual artigo da CLT fala de férias?"}
        >>> no_classificar(estado)
        {"intencao": "pesquisa_legal"}
    """
    mensagem = estado.get("mensagem_usuario", "").strip()
    if not mensagem:
        logger.warning("no_classificar: mensagem vazia — usando 'desconhecida'")
        return {"intencao": "desconhecida"}

    try:
        from ana.config import obter_configuracao
        from ana.config_modelos import obter_modelos
        from ana.providers.llm import OllamaLLMProvider

        config = obter_configuracao()
        modelos = obter_modelos()
        modelo = modelos.ativo.agentes.orquestrador

        llm = OllamaLLMProvider(modelo=modelo, host=config.ollama_host)
        transcricao = estado.get("transcricao_anexada")
        documento = estado.get("documento_processo")

        # Documento de processo → sempre pesquisa_legal (sem chamar LLM)
        if documento and not mensagem:
            logger.info("no_classificar: documento_processo sem mensagem → pesquisa_legal")
            return {"intencao": "pesquisa_legal"}

        prompt = prompt_classificacao(
            mensagem,
            transcricao_anexada=transcricao,
            documento_processo=documento,
        )
        resposta_bruta = llm.invocar(prompt, temperatura=0.0)

        # Normaliza resposta — extrai apenas a categoria
        intencao = resposta_bruta.strip().lower().split()[0] if resposta_bruta.strip() else ""
        intencao = intencao.rstrip(".,;:")

        if intencao not in INTENCOES_VALIDAS:
            logger.warning(
                f"no_classificar: intenção '{intencao}' não reconhecida "
                f"(resposta bruta: {resposta_bruta!r}) — usando 'desconhecida'"
            )
            intencao = "desconhecida"

        logger.info(f"no_classificar: '{mensagem[:50]}...' → {intencao}")
        return {"intencao": intencao}

    except Exception as e:
        logger.error(f"no_classificar: erro na chamada ao LLM: {e}")
        return {"intencao": "desconhecida", "erro": f"Falha na classificação: {e}"}
