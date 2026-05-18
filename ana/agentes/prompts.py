"""Templates de prompt para os agentes do sistema ANA.

Cada função retorna um prompt completo pronto para envio ao LLM,
seguindo as boas práticas de prompt engineering para modelos jurídicos.

Os prompts são escritos em português para maximizar precisão em
domínio jurídico brasileiro. Nenhum dado é enviado a APIs externas —
todos os LLMs rodam via Ollama local (conformidade LGPD).

Referência de modelos por agente em ``config/modelos.yaml``.
"""

from __future__ import annotations


# =============================================================================
# Classificação de Intenção (Orquestrador)
# =============================================================================

PROMPT_CLASSIFICACAO = """\
Você é um classificador de intenções jurídicas. Analise a mensagem do usuário \
e responda APENAS com uma das categorias abaixo, sem explicações adicionais.

CATEGORIAS:
- pesquisa_legal      → Pergunta sobre leis, artigos, normas, jurisprudência ou legislação
- analise_processo    → Pergunta sobre documentos, fatos ou dados do processo atual
- gerar_documento     → Solicitação para criar petição, recurso ou outro documento jurídico
- transcrever_audio   → Solicitação para transcrever ou analisar audiência gravada
- verificar_prazo     → Pergunta sobre prazos processuais
- buscar_similar      → Busca de processos semelhantes ou jurisprudência análoga
- desconhecida        → Não se enquadra nas categorias acima

{contexto_transcricao}MENSAGEM DO USUÁRIO:
{mensagem}

CATEGORIA (responda apenas a palavra da categoria):"""


def prompt_classificacao(
    mensagem: str,
    transcricao_anexada: str | None = None,
    documento_processo: str | None = None,
) -> str:
    """Monta o prompt de classificação de intenção para o orquestrador.

    Args:
        mensagem: Mensagem original do usuário.

    Returns:
        Prompt formatado para o modelo orquestrador.
    """
    if documento_processo:
        contexto = (
            "CONTEXTO: O usuário anexou um documento de processo jurídico. "
            "Se a mensagem pedir para descobrir leis aplicáveis ou relacionadas, classifique como 'pesquisa_legal'.\n\n"
        )
    elif transcricao_anexada:
        contexto = (
            "CONTEXTO: O usuário anexou uma transcrição de audiência ao chat. "
            "Se a mensagem se referir à transcrição ou ao seu conteúdo, classifique como 'analise_processo'.\n\n"
        )
    else:
        contexto = ""
    return PROMPT_CLASSIFICACAO.format(mensagem=mensagem.strip(), contexto_transcricao=contexto)


# =============================================================================
# Síntese de Legislação (Pesquisador)
# =============================================================================

def prompt_pesquisa_legal(
    mensagem: str,
    contexto_chunks: list[dict],
    transcricao_anexada: str | None = None,
) -> str:
    """Monta o prompt de síntese jurídica com base nos chunks recuperados.

    Formata os chunks como referências numeradas com fonte e artigo,
    instruindo o LLM a citar explicitamente cada fonte usada.

    Args:
        mensagem: Pergunta original do usuário.
        contexto_chunks: Lista de dicts com ``texto``, ``fonte``, ``artigo``.
        transcricao_anexada: Texto de transcrição de audiência (contexto adicional).

    Returns:
        Prompt formatado para o modelo pesquisador.
    """
    trans_bloco = ""
    if transcricao_anexada:
        trecho = transcricao_anexada[:2000]
        trans_bloco = (
            f"TRANSCRIÇÃO DE AUDIÊNCIA (contexto adicional):\n{trecho}\n\n"
        )

    if not contexto_chunks:
        return (
            "Você é um assistente jurídico brasileiro. "
            f"{trans_bloco}"
            f"O usuário perguntou: '{mensagem}'\n\n"
            "Não foram encontrados artigos relevantes no banco de dados. "
            "Informe isso claramente ao usuário e sugira reformular a pergunta."
        )

    partes: list[str] = []
    for i, chunk in enumerate(contexto_chunks, start=1):
        fonte = chunk.get("fonte", "Fonte desconhecida")
        artigo = chunk.get("artigo", "")
        texto = chunk.get("texto", "")
        cab = f"[{i}] {fonte}"
        if artigo:
            cab += f" — {artigo}"
        partes.append(f"{cab}\n{texto}")

    legislacao_fmt = "\n\n".join(partes)

    return (
        "Você é um assistente jurídico brasileiro especializado em legislação. "
        "Com base APENAS nos artigos abaixo, responda à pergunta do usuário de forma "
        "objetiva e precisa. Cite as fontes pelo número de referência [1], [2], etc. "
        "Se os artigos não forem suficientes, diga isso claramente. "
        "Seja conciso: máximo 3 parágrafos.\n\n"
        f"{trans_bloco}"
        f"PERGUNTA: {mensagem}\n\n"
        f"LEGISLAÇÃO RELEVANTE:\n{legislacao_fmt}\n\n"
        "RESPOSTA:"
    )


# =============================================================================
# Análise de Processo (Analista)
# =============================================================================

def prompt_analise_processo(
    mensagem: str,
    contexto_chunks: list[dict],
    numero_processo: str = "",
    transcricao_anexada: str | None = None,
) -> str:
    """Monta o prompt de análise documental do processo ativo.

    Args:
        mensagem: Pergunta sobre o processo.
        contexto_chunks: Chunks dos documentos do processo recuperados por pgvector.
        numero_processo: Número do processo (informativo no prompt).
        transcricao_anexada: Texto de transcrição de audiência (documento adicional).

    Returns:
        Prompt formatado para o modelo analista.
    """
    cabecalho = f"Processo {numero_processo}" if numero_processo else "Processo ativo"

    trans_bloco = ""
    if transcricao_anexada:
        trans_bloco = (
            f"[T] Transcrição de audiência (anexada pelo usuário)\n"
            f"{transcricao_anexada[:3000]}\n\n"
        )

    if not contexto_chunks and not transcricao_anexada:
        return (
            "Você é um assistente jurídico. "
            f"O usuário perguntou sobre o {cabecalho}: '{mensagem}'\n\n"
            "Não foram encontrados documentos relevantes neste processo. "
            "Informe isso ao usuário e sugira anexar documentos à sessão."
        )

    partes = [
        f"[{i}] {c.get('fonte', 'Documento')}\n{c.get('texto', '')}"
        for i, c in enumerate(contexto_chunks, start=1)
    ]
    docs_fmt = "\n\n".join(partes)

    return (
        f"Você é um assistente jurídico analisando o {cabecalho}. "
        "Com base nos trechos dos documentos abaixo, responda à pergunta do usuário "
        "de forma objetiva. Cite o documento de referência [1], [2], [T], etc. "
        "Se não houver informação suficiente, diga claramente.\n\n"
        f"PERGUNTA: {mensagem}\n\n"
        f"DOCUMENTOS DO PROCESSO:\n{trans_bloco}{docs_fmt}\n\n"
        "RESPOSTA:"
    )


# =============================================================================
# Resposta Genérica (fallback para intenções desconhecidas)
# =============================================================================

def prompt_resposta_generica(mensagem: str) -> str:
    """Monta prompt de resposta genérica para intenções não classificadas.

    Args:
        mensagem: Mensagem original do usuário.

    Returns:
        Prompt para resposta educada e orientadora.
    """
    return (
        "Você é um assistente jurídico brasileiro. "
        "O usuário enviou a seguinte mensagem:\n\n"
        f"'{mensagem}'\n\n"
        "Responda de forma educada e útil. Se for uma saudação, responda brevemente. "
        "Se for uma dúvida jurídica não bem formulada, peça para o usuário reformular "
        "com mais detalhes sobre a lei ou processo que deseja consultar. "
        "Seja breve (1-2 frases).\n\nRESPOSTA:"
    )


# =============================================================================
# Reformulação Jurídica (Redator)
# =============================================================================

def prompt_reformulacao(prompt_sintese: str) -> str:
    """Enriquece o prompt de síntese com instruções de reformulação jurídica.

    Transforma o prompt bruto de síntese em um prompt que instrui o modelo
    a gerar uma resposta em português jurídico brasileiro formal e técnico,
    mantendo precisão e citações.

    Args:
        prompt_sintese: Prompt bruto montado pelos nós de pesquisa/análise.

    Returns:
        Prompt enriquecido com instruções de reformulação jurídica.
    """
    instrucao = (
        "Você é um redator jurídico especializado em direito brasileiro. "
        "Sua tarefa é responder à questão abaixo em linguagem jurídica formal e técnica, "
        "usando terminologia processual adequada, construções em voz passiva quando pertinente, "
        "e fundamentação legal precisa. Cite os dispositivos legais no formato padrão "
        "(ex: 'nos termos do art. 5º, inciso X, da Constituição Federal'). "
        "Seja objetivo e fundamentado. Evite linguagem coloquial.\n\n"
    )
    return instrucao + prompt_sintese


# =============================================================================
# Geração de Descrição (Sintetizador)
# =============================================================================

def prompt_descricao(mensagem: str, resposta: str) -> str:
    """Gera prompt para criar uma descrição curta da conversa.

    Usado para gerar o título/descrição do chat exibido no histórico
    do frontend após a resposta ser gerada.

    Args:
        mensagem: Pergunta original do usuário.
        resposta: Resposta gerada pelo assistente.

    Returns:
        Prompt para geração de descrição de 1–2 frases.
    """
    return (
        "Crie um título curto (máximo 8 palavras) que descreva o tema desta consulta jurídica.\n\n"
        f"PERGUNTA: {mensagem[:200]}\n"
        f"RESPOSTA (trecho): {resposta[:300]}\n\n"
        "TÍTULO (sem aspas, sem ponto final):"
    )
