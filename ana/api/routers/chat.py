"""Router FastAPI para o endpoint de chat agêntico.

Implementa o ``POST /chat`` que orquestra o pipeline LangGraph:
1. Grafo executa: classificar → pesquisar/analisar → validar → reformular → gerar_resposta
2. Em modo streaming (padrão): SSE com tokens do LLM sintetizador (prompt reformulado)
3. Após streaming: gera ``descricao`` curta do chat (título) via LLM
4. Em modo não-streaming: resposta JSON completa

O ``sessao_id`` é o ``thread_id`` do checkpointer ``AsyncSqliteSaver``,
habilitando conversas multi-turno com memória persistente por processo.

Conformidade LGPD:
    Mensagens e contexto persistem apenas no SQLite local do backend.
    Nenhum dado de conversa é enviado a APIs externas.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from ana.api.schemas.chat import RequisicaoChat, RespostaChat

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/",
    response_model=None,
    summary="Chat com o agente jurídico ANA",
    description=(
        "Envia uma mensagem ao agente jurídico e recebe resposta contextualizada. "
        "O grafo classifica a intenção, busca contexto, reformula em linguagem jurídica "
        "e sintetiza a resposta. Com ``stream=true`` (padrão), entrega via SSE. "
        "O campo ``transcricao_anexada`` permite incluir transcrição de audiência como contexto."
    ),
)
async def chat(requisicao: RequisicaoChat) -> StreamingResponse | RespostaChat:
    """Processa mensagem via pipeline LangGraph com modo streaming ou JSON.

    Args:
        requisicao: Mensagem, sessao_id, transcricao_anexada e flag de streaming.

    Returns:
        ``StreamingResponse`` (SSE) se ``stream=True``,
        ou ``RespostaChat`` (JSON) se ``stream=False``.

    Raises:
        HTTPException 400: Mensagem vazia.
        HTTPException 500: Falha inesperada no grafo.
    """
    from ana.agentes.grafo import obter_grafo, config_sessao

    if not requisicao.mensagem.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")

    sessao_id = requisicao.sessao_id or "sessao-global"
    estado: dict = {
        "mensagem_usuario": requisicao.mensagem,
        "sessao_id": sessao_id,
    }
    if requisicao.transcricao_anexada:
        estado["transcricao_anexada"] = requisicao.transcricao_anexada
    if requisicao.documento_processo:
        estado["documento_processo"] = requisicao.documento_processo

    config = config_sessao(sessao_id)

    if requisicao.stream:
        return StreamingResponse(
            _stream_chat(estado, config),
            media_type="text/event-stream; charset=utf-8",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Modo bloqueante: executa grafo completo + LLM síncrono
    try:
        async with obter_grafo() as app:
            resultado = await app.ainvoke(estado, config=config)

        return RespostaChat(
            resposta=resultado.get("resposta", ""),
            intencao=resultado.get("intencao", "desconhecida"),
            sessao_id=sessao_id,
            descricao=resultado.get("descricao"),
            contexto_rag=resultado.get("contexto_rag") or [],
            validacao=resultado.get("validacao"),
            erro=resultado.get("erro"),
        )

    except Exception as e:
        logger.error(f"/chat (bloqueante): erro no grafo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {e}") from e


async def _stream_chat(
    estado: dict,
    config: dict,
) -> AsyncGenerator[str, None]:
    """Gerador SSE do chat agêntico.

    Pipeline de streaming:
    1. Executa o grafo LangGraph → intenção + contexto RAG + prompt_reformulacao
    2. Emite evento ``metadados`` (intenção, total de chunks)
    3. Chama LLM pesquisador com ``invocar_stream`` usando prompt reformulado → emite tokens
    4. Executa validação anti-alucinação na resposta completa
    5. Gera descrição curta (título) via LLM orquestrador (não-streaming)
    6. Emite evento ``fim`` com validação, fontes e descrição

    Args:
        estado: Estado inicial do grafo (mensagem + sessao_id + transcricao_anexada).
        config: Config LangGraph com thread_id.

    Yields:
        Strings no formato SSE: ``data: {...}\\n\\n``.
    """
    from ana.agentes.grafo import obter_grafo
    from ana.agentes.prompts import prompt_descricao
    from ana.config import obter_configuracao
    from ana.config_modelos import obter_modelos
    from ana.providers.llm import OllamaLLMProvider
    from ana.validacao.pipeline import validar_resposta

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        # ── Fase 1: grafo coleta intenção, contexto e prompt reformulado ──────
        async with obter_grafo() as app:
            resultado = await app.ainvoke(estado, config=config)

        intencao = resultado.get("intencao", "desconhecida")
        contexto_rag = resultado.get("contexto_rag") or []

        # Usa prompt_reformulacao se disponível, fallback para prompt_sintese
        prompt_final = resultado.get("prompt_reformulacao") or resultado.get("prompt_sintese")

        # Para pesquisa_legal: inclui os chunks completos no metadados para o frontend exibir
        # os artigos de lei originais antes de (e separado de) a resposta do LLM.
        chunks_para_frontend = contexto_rag if intencao == "pesquisa_legal" else []

        yield _sse({
            "tipo": "metadados",
            "dados": {
                "intencao": intencao,
                "total_chunks": len(contexto_rag),
                "sessao_id": estado.get("sessao_id", ""),
                "erro_parcial": resultado.get("erro"),
                "contexto_rag": chunks_para_frontend,
            },
        })

        # ── Fase 2: sem prompt de síntese → resposta direta do grafo ─────────
        if not prompt_final:
            resposta_direta = resultado.get("resposta", "")
            if resposta_direta:
                yield _sse({"tipo": "token", "token": resposta_direta})
            yield _sse({"tipo": "fim", "dados": {"intencao": intencao, "validacao": None, "descricao": None}})
            return

        # ── Fase 3: streaming de tokens do LLM pesquisador ───────────────────
        config_app = obter_configuracao()
        modelos = obter_modelos()
        modelo_pesquisador = modelos.ativo.agentes.pesquisador
        modelo_orquestrador = modelos.ativo.agentes.orquestrador
        llm = OllamaLLMProvider(modelo=modelo_pesquisador, host=config_app.ollama_host)

        tokens_coletados: list[str] = []

        # invocar_stream é síncrono — executa em thread para não bloquear o loop
        tokens = await asyncio.to_thread(
            lambda: list(llm.invocar_stream(prompt_final, temperatura=0.2))
        )
        for token in tokens:
            tokens_coletados.append(token)
            yield _sse({"tipo": "token", "token": token})

        resposta_completa = "".join(tokens_coletados)

        # ── Fase 4: validação anti-alucinação na resposta completa ────────────
        validacao = None
        try:
            validacao = await asyncio.to_thread(
                validar_resposta, resposta_completa, False
            )
        except Exception as e:
            logger.debug(f"Validação pós-streaming ignorada: {e}")

        # ── Fase 5: geração de descrição curta (título do chat) ───────────────
        descricao: str | None = None
        try:
            llm_orq = OllamaLLMProvider(modelo=modelo_orquestrador, host=config_app.ollama_host)
            prompt_desc = prompt_descricao(
                estado.get("mensagem_usuario", ""), resposta_completa
            )
            descricao = await asyncio.to_thread(
                lambda: llm_orq.invocar(prompt_desc, temperatura=0.1).strip().strip('"').strip("'")
            )
            logger.debug(f"Descrição gerada: {descricao!r}")
        except Exception as e:
            logger.debug(f"Geração de descrição ignorada: {e}")

        yield _sse({
            "tipo": "fim",
            "dados": {
                "intencao": intencao,
                "validacao": validacao,
                "descricao": descricao,
                "chunks_fontes": [
                    {"fonte": c.get("fonte", ""), "artigo": c.get("artigo", "")}
                    for c in contexto_rag[:5]
                ],
            },
        })

    except Exception as e:
        logger.error(f"_stream_chat: erro inesperado: {e}")
        yield _sse({"tipo": "erro", "dados": {"mensagem": str(e)}})
