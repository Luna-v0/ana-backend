"""Validação semântica de citações legais via SLM.

Compara o que foi afirmado pelo LLM com o texto real da lei para detectar
alucinações sutis (paráfrases incorretas, inversões de sentido, etc.).
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def validar_semantica(
    texto_afirmado: str,
    texto_real: str,
    modelo: str | None = None,
) -> dict[str, Any]:
    """Compara afirmação com o texto real da lei via SLM.

    Usa um modelo pequeno (qwen2.5:3b por padrão) para verificar se
    ``texto_afirmado`` é semanticamente consistente com ``texto_real``.

    Args:
        texto_afirmado: O que o LLM disse sobre a lei.
        texto_real: Texto original do artigo recuperado do PostgreSQL.
        modelo: Modelo Ollama a usar. Default: validador do config ativo.

    Returns:
        Dicionário com:
        - ``consistente``: bool
        - ``motivo``: Explicação da decisão.
        - ``confianca``: 'alta' | 'media' | 'baixa'
    """
    from ana.config import obter_configuracao
    from ana.config_modelos import obter_modelos
    from ana.providers.llm import OllamaLLMProvider

    config = obter_configuracao()
    modelo_cfg = obter_modelos().ativo.agentes

    modelo_nome = modelo or getattr(modelo_cfg, "validador", "qwen2.5:3b")

    prompt = (
        "Você é um verificador jurídico brasileiro. Compare a AFIRMAÇÃO com o TEXTO LEGAL.\n"
        "Responda APENAS com um JSON no formato:\n"
        '{"consistente": true|false, "motivo": "...", "confianca": "alta|media|baixa"}\n\n'
        f"AFIRMAÇÃO: {texto_afirmado[:800]}\n\n"
        f"TEXTO LEGAL: {texto_real[:800]}\n\n"
        "JSON:"
    )

    try:
        llm = OllamaLLMProvider(modelo=modelo_nome, host=config.ollama_host)
        resposta = llm.invocar(prompt, temperatura=0.0)
        import json
        # Extrai o JSON da resposta
        inicio = resposta.find("{")
        fim = resposta.rfind("}") + 1
        if inicio >= 0 and fim > inicio:
            dados = json.loads(resposta[inicio:fim])
            return {
                "consistente": bool(dados.get("consistente", False)),
                "motivo": str(dados.get("motivo", "")),
                "confianca": str(dados.get("confianca", "media")),
            }
    except Exception as e:
        logger.warning(f"Validação semântica falhou: {e}")

    return {
        "consistente": None,
        "motivo": "Validação semântica indisponível (SLM não acessível).",
        "confianca": "baixa",
    }
