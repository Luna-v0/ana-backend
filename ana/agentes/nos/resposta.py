"""Nó sintetizador de resposta — fallback para intenções sem rota específica.

Para intenções ``pesquisa_legal`` e ``analise_processo``, o prompt de
síntese é montado nos nós anteriores e o LLM é invocado diretamente
no endpoint ``/chat`` (com streaming). Este nó é ativado apenas para
intenções sem rota especializada (ex: ``desconhecida``, ``gerar_documento``).

Em modo não-streaming (invocação direta do grafo sem SSE), o nó
também executa a síntese final chamando o LLM pesquisador.
"""

from __future__ import annotations

from loguru import logger

from ana.agentes.estado import EstadoJuridico
from ana.agentes.prompts import prompt_resposta_generica


def no_gerar_resposta(estado: EstadoJuridico) -> dict:
    """Gera a resposta final para intenções sem rota RAG/analítica.

    Quando o estado já tem ``prompt_sintese`` (vindo de pesquisar ou analisar),
    invoca o LLM pesquisador de forma síncrona (modo não-streaming).

    Quando não há prompt de síntese (intenções desconhecidas), gera
    uma resposta orientativa usando prompt genérico.

    Args:
        estado: Estado atual com campos opcionais ``prompt_sintese``,
            ``contexto_rag`` e ``intencao``.

    Returns:
        Dicionário com ``resposta`` (texto final gerado) e opcionalmente
        ``validacao`` com resultado da verificação de citações legais.
    """
    prompt = estado.get("prompt_sintese")
    intencao = estado.get("intencao", "desconhecida")
    mensagem = estado.get("mensagem_usuario", "")

    # Sem contexto: resposta genérica (não chama LLM pesquisador)
    if not prompt:
        if intencao == "desconhecida":
            resposta = _resposta_orientativa(mensagem)
        elif intencao == "gerar_documento":
            resposta = (
                "Para gerar um documento jurídico, por favor forneça mais detalhes: "
                "tipo de peça (petição inicial, recurso, etc.), partes envolvidas e "
                "os fatos principais. Você também pode usar o painel de Redação."
            )
        elif intencao in ("transcrever_audio", "buscar_similar"):
            resposta = (
                f"A função '{intencao}' está disponível via endpoint específico. "
                "Use o painel correspondente na interface."
            )
        else:
            resposta = "Não consegui encontrar informações relevantes. Tente reformular sua pergunta."

        return {"resposta": resposta}

    # Tem prompt de síntese: invoca LLM (modo não-streaming / fallback)
    try:
        from ana.config import obter_configuracao
        from ana.config_modelos import obter_modelos
        from ana.providers.llm import OllamaLLMProvider

        config = obter_configuracao()
        modelo = obter_modelos().ativo.agentes.pesquisador
        llm = OllamaLLMProvider(modelo=modelo, host=config.ollama_host)

        resposta = llm.invocar(prompt, temperatura=0.2)
        logger.info(f"no_gerar_resposta: resposta gerada ({len(resposta)} chars)")

        # Validação de leis citadas (sem camada semântica para performance)
        validacao = _validar_citacoes(resposta)

        return {"resposta": resposta.strip(), "validacao": validacao}

    except Exception as e:
        logger.error(f"no_gerar_resposta: erro ao invocar LLM: {e}")
        return {
            "resposta": "Ocorreu um erro ao gerar a resposta. Tente novamente.",
            "erro": str(e),
        }


def _resposta_orientativa(mensagem: str) -> str:
    """Tenta gerar resposta orientativa via LLM ou retorna fallback fixo.

    Args:
        mensagem: Mensagem original do usuário.

    Returns:
        Texto de resposta genérica ou orientativa.
    """
    try:
        from ana.config import obter_configuracao
        from ana.config_modelos import obter_modelos
        from ana.providers.llm import OllamaLLMProvider

        config = obter_configuracao()
        modelo = obter_modelos().ativo.agentes.orquestrador
        llm = OllamaLLMProvider(modelo=modelo, host=config.ollama_host)
        return llm.invocar(prompt_resposta_generica(mensagem), temperatura=0.3).strip()
    except Exception:
        return (
            "Olá! Sou o assistente jurídico ANA. Posso ajudar com consultas sobre "
            "legislação brasileira, análise de documentos do processo e redação jurídica. "
            "Como posso ajudar?"
        )


def _validar_citacoes(texto: str) -> list[dict] | None:
    """Executa validação anti-alucinação nas citações legais do texto.

    Args:
        texto: Texto gerado pelo LLM para validação.

    Returns:
        Lista de resultados de validação ou None se falhar.
    """
    try:
        from ana.validacao.pipeline import validar_resposta
        return validar_resposta(texto, usar_semantica=False)
    except Exception as e:
        logger.debug(f"Validação de citações ignorada: {e}")
        return None
