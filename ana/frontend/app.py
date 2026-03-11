"""Frontend Gradio do sistema ANA — Attorney Normative Assistent.

Interface estilo chatbot escuro com barra lateral mínima:
- Barra lateral: navegação entre Chat e Transcrição
- Chat: busca híbrida com pills de vigência, seletor de área e +/- resultados
- Transcrição: upload de áudio com metadados opcionais

Uso:
    uv run ana-ui
    uv run python -m ana.frontend.app --backend http://localhost:8000 --porta 7860
"""

import argparse
import os

import gradio as gr

from ana.frontend.cliente import ClienteANA


def _url_backend_padrao() -> str:
    if url := os.environ.get("ANA_BACKEND_URL"):
        return url
    if os.path.exists("/.dockerenv"):
        return "http://backend:8000"
    return "http://localhost:8000"


_URL_BACKEND_PADRAO = _url_backend_padrao()

# =============================================================================
# Formatação
# =============================================================================

def _md_resultados_busca(resultados: list[dict]) -> str:
    if not resultados:
        return (
            "### Nenhum resultado encontrado\n\n"
            "_Verifique se há legislação indexada ou refine a busca._"
        )
    blocos = []
    for i, r in enumerate(resultados, 1):
        fonte = r.get("fonte", "Fonte desconhecida")
        artigo = r.get("artigo") or ""
        area = r.get("area") or ""
        vigencia = r.get("vigencia") or ""
        score = r.get("score", 0.0)
        texto = r.get("texto", "")
        titulo = f"### {i}. {fonte}"
        if artigo:
            titulo += f" — {artigo}"
        badges = []
        if area:
            badges.append(f"`{area}`")
        if vigencia:
            badges.append(f"`{vigencia}`")
        badges.append(f"score: `{score:.4f}`")
        blocos.append(
            f"{titulo}\n"
            f"{' &nbsp; '.join(badges)}\n\n"
            f"{texto}\n\n---"
        )
    return "\n".join(blocos)


# =============================================================================
# Funções de evento
# =============================================================================

_VIGENCIA_MAP = {
    "Somente vigentes": "ativa",
    "Somente revogadas": "revogada",
    "Todas": None,
}


def fn_buscar(
    url_backend: str,
    query: str,
    area: str,
    top_k: int,
    usar_reranker: bool,
    usar_mmr: bool,
    vigencia: str,
) -> tuple[str, str]:
    if not query.strip():
        return "⚠️ **Erro:** Digite uma consulta antes de buscar.", ""
    cliente = ClienteANA(url_backend)
    resp = cliente.buscar(
        query=query.strip(),
        area=area if area != "Todas" else None,
        top_k=int(top_k),
        usar_reranker=usar_reranker,
        usar_mmr=usar_mmr,
        vigencia=_VIGENCIA_MAP.get(vigencia, "ativa"),
    )
    if not resp.sucesso:
        return f"## ❌ Erro na Busca\n\n```\n{resp.erro}\n```", ""
    dados = resp.dados
    resultados = dados.get("resultados", [])
    return _md_resultados_busca(resultados), ""


def fn_chat(
    mensagem: str,
    historico: list,
    url_backend: str,
    area: str,
    top_k: int,
    vigencia: str,
) -> tuple[list, str]:
    """Executa busca e acrescenta ao histórico no formato messages (dicts)."""
    if not mensagem.strip():
        return historico, ""
    # Reranker e MMR sempre ativos
    resultado, _ = fn_buscar(url_backend, mensagem, area, top_k, True, True, vigencia)
    historico = historico + [
        {"role": "user", "content": mensagem},
        {"role": "assistant", "content": resultado},
    ]
    return historico, ""


def fn_top_k_dec(k: int) -> int:
    return max(1, int(k) - 1)


def fn_top_k_inc(k: int) -> int:
    return min(20, int(k) + 1)


def fn_transcrever(
    url_backend: str,
    arquivo_audio,
    numero_processo: str,
    data_audiencia: str,
    tipo_audiencia: str,
    vara: str,
    cidade_uf: str,
    min_speakers: int,
    max_speakers: int,
    identificar_por_llm: bool,
) -> tuple[str, str]:
    if arquivo_audio is None:
        return "⚠️ **Erro:** Selecione um arquivo de áudio antes de transcrever.", ""
    caminho = arquivo_audio if isinstance(arquivo_audio, str) else arquivo_audio.name
    cliente = ClienteANA(url_backend)
    resp = cliente.transcrever(
        caminho_audio=caminho,
        numero_processo=numero_processo.strip(),
        data_audiencia=data_audiencia.strip(),
        tipo_audiencia=tipo_audiencia.strip() or "Instrução e Julgamento",
        vara=vara.strip(),
        cidade_uf=cidade_uf.strip(),
        min_speakers=int(min_speakers),
        max_speakers=int(max_speakers),
        identificar_por_llm=identificar_por_llm,
    )
    if not resp.sucesso:
        if "503" in str(resp.erro) or "Módulo de transcrição" in resp.erro:
            return (
                "## ❌ Módulo de Transcrição Indisponível\n\n"
                f"```\n{resp.erro}\n```\n\n"
                "**Para habilitar:**\n```bash\nuv sync --group transcricao\n"
                "export HF_TOKEN=hf_seu_token_aqui\n```",
                "",
            )
        return f"## ❌ Erro na Transcrição\n\n```\n{resp.erro}\n```", ""
    dados = resp.dados
    md_status = (
        f"**Arquivo:** `{dados.get('arquivo_processado', '-')}`  \n"
        f"**Falantes:** `{dados.get('num_participantes', 0)}`  \n"
        f"**Duração:** `{int(dados.get('duracao_total', 0) // 60)}min "
        f"{int(dados.get('duracao_total', 0) % 60)}s`"
    )
    return dados.get("markdown", "_Transcrição vazia._"), md_status


# =============================================================================
# Layout
# =============================================================================

_AREAS = [
    "Todas", "civil", "penal", "trabalhista", "tributario",
    "consumidor", "dados", "administrativo",
    "constitucional", "processual_civil", "processual_penal",
]

_CSS = """
/* ===== Base escura ===== */
body, .gradio-container, .app, .main, .contain {
    background: #1c1c1c !important;
    color: #e0e0e0 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
footer, .footer { display: none !important; }

/* ===== Layout externo ===== */
#outer-row {
    height: 100vh !important;
    margin: 0 !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    overflow: hidden !important;
}

/* ===== Barra lateral ===== */
#sidebar {
    background: #161616 !important;
    border-right: 1px solid #262626 !important;
    min-width: 56px !important;
    max-width: 56px !important;
    padding: 14px 8px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    align-items: center !important;
    flex-shrink: 0 !important;
    height: 100vh !important;
}
#btn-nav-chat, #btn-nav-trans {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #484848 !important;
    width: 40px !important;
    min-width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    font-size: 18px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    box-shadow: none !important;
}
#btn-nav-chat:hover, #btn-nav-trans:hover {
    background: #242424 !important;
    color: #aaa !important;
}

/* ===== Colunas principais ===== */
#main-chat, #main-trans {
    background: #1c1c1c !important;
    padding: 0 !important;
    overflow-y: auto !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
}

/* ===== Cabeçalho de boas-vindas ===== */
#welcome {
    text-align: center;
    padding: 68px 20px 36px;
    flex-shrink: 0;
}
#welcome p, #welcome p strong {
    color: #e0e0e0 !important;
    font-size: 2.2em !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}

/* ===== Chatbot ===== */
#chatbot {
    background: transparent !important;
    border: none !important;
    flex: 1 !important;
}
#chatbot > div { background: transparent !important; }

/* ===== Área de input ===== */
#input-area {
    padding: 6px 18% 24px !important;
    background: #1c1c1c !important;
    flex-shrink: 0 !important;
}

/* ===== Pills de vigência ===== */
#vigencia-pills > div {
    background: transparent !important;
    border: none !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    justify-content: center !important;
    padding: 0 0 12px !important;
}
#vigencia-pills label {
    background: #222 !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 999px !important;
    padding: 5px 14px !important;
    color: #484848 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    margin: 0 !important;
    user-select: none !important;
}
#vigencia-pills label:has(input:checked) {
    border-color: #484848 !important;
    color: #c8c8c8 !important;
    background: #282828 !important;
}
#vigencia-pills input[type="radio"] { display: none !important; }

/* ===== Linha área + top-k ===== */
#controls-row {
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 8px !important;
    justify-content: center !important;
}
#dd-area {
    flex: 1 !important;
    max-width: 200px !important;
}
#dd-area label { display: none !important; }
#dd-area select, #dd-area input, #dd-area .wrap-inner, #dd-area ul {
    background: #222 !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 999px !important;
    color: #686868 !important;
    font-size: 13px !important;
    padding: 4px 12px !important;
}
#topk-group {
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    background: #222 !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 999px !important;
    padding: 3px 10px !important;
}
#btn-k-minus, #btn-k-plus {
    background: transparent !important;
    border: none !important;
    color: #585858 !important;
    font-size: 16px !important;
    width: 22px !important;
    min-width: 22px !important;
    height: 22px !important;
    padding: 0 !important;
    line-height: 1 !important;
    cursor: pointer !important;
    border-radius: 50% !important;
    box-shadow: none !important;
    transition: color 0.12s !important;
}
#btn-k-minus:hover, #btn-k-plus:hover { color: #bbb !important; }
#num-top-k {
    min-width: 28px !important;
    max-width: 28px !important;
}
#num-top-k label { display: none !important; }
#num-top-k input {
    background: transparent !important;
    border: none !important;
    color: #888 !important;
    font-size: 13px !important;
    text-align: center !important;
    padding: 0 !important;
    width: 28px !important;
    box-shadow: none !important;
}

/* ===== Caixa de mensagem ===== */
#msg-row { gap: 8px !important; align-items: flex-end !important; }
#txt-msg label { display: none !important; }
#txt-msg textarea {
    background: #222 !important;
    border: 1px solid #2c2c2c !important;
    border-radius: 14px !important;
    color: #e0e0e0 !important;
    font-size: 15px !important;
    resize: none !important;
    padding: 13px 16px !important;
    transition: border-color 0.15s !important;
}
#txt-msg textarea::placeholder {
    color: rgba(255, 255, 255, 0.18) !important;
}
#txt-msg textarea:focus {
    border-color: #404040 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ===== Botão enviar ===== */
#btn-enviar {
    background: #242424 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 10px !important;
    color: #888 !important;
    min-width: 72px !important;
    height: 46px !important;
    font-size: 15px !important;
    transition: background 0.15s, color 0.15s !important;
}
#btn-enviar:hover {
    background: #2c2c2c !important;
    color: #ddd !important;
}

/* ===== Botão limpar ===== */
#btn-limpar {
    background: transparent !important;
    border: none !important;
    color: #333 !important;
    font-size: 12px !important;
    margin-top: 6px !important;
    box-shadow: none !important;
    transition: color 0.15s !important;
}
#btn-limpar:hover { color: #777 !important; }

/* ===== Vista transcrição ===== */
#trans-header {
    text-align: center;
    padding: 48px 20px 24px;
}
#trans-header p, #trans-header p strong {
    color: #e0e0e0 !important;
    font-size: 1.8em !important;
    font-weight: 600 !important;
}
#trans-content {
    padding: 0 15% 32px !important;
}
"""


def criar_interface(url_backend: str = _URL_BACKEND_PADRAO) -> gr.Blocks:
    """Cria e retorna a interface Gradio estilo chatbot escuro."""
    with gr.Blocks() as interface:

        estado_backend = gr.State(url_backend)

        with gr.Row(elem_id="outer-row"):

            # ================================================================
            # Barra lateral
            # ================================================================
            with gr.Column(elem_id="sidebar", min_width=56, scale=0):
                btn_nav_chat = gr.Button("💬", elem_id="btn-nav-chat", size="sm")
                btn_nav_trans = gr.Button("🎙️", elem_id="btn-nav-trans", size="sm")

            # ================================================================
            # Vista: Chat
            # ================================================================
            with gr.Column(elem_id="main-chat", scale=1) as col_chat:

                gr.Markdown("**ANA**", elem_id="welcome")

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=460,
                    show_label=False,
                )

                with gr.Column(elem_id="input-area"):

                    # Vigência pills
                    rd_vigencia = gr.Radio(
                        choices=["Somente vigentes", "Todas", "Somente revogadas"],
                        value="Somente vigentes",
                        show_label=False,
                        elem_id="vigencia-pills",
                    )

                    # Área jurídica + top-k +/-
                    with gr.Row(elem_id="controls-row"):
                        dd_area = gr.Dropdown(
                            choices=_AREAS,
                            value="Todas",
                            show_label=False,
                            elem_id="dd-area",
                            scale=2,
                        )
                        with gr.Row(elem_id="topk-group"):
                            btn_k_minus = gr.Button("−", size="sm", elem_id="btn-k-minus", scale=0)
                            num_top_k = gr.Number(
                                value=5,
                                show_label=False,
                                minimum=1,
                                maximum=20,
                                precision=0,
                                elem_id="num-top-k",
                                scale=0,
                                min_width=28,
                            )
                            btn_k_plus = gr.Button("+", size="sm", elem_id="btn-k-plus", scale=0)

                    # Mensagem + enviar
                    with gr.Row(elem_id="msg-row"):
                        txt_msg = gr.Textbox(
                            placeholder="Qual lei você quer pesquisar?",
                            show_label=False,
                            elem_id="txt-msg",
                            scale=8,
                            lines=1,
                        )
                        btn_enviar = gr.Button("↩", elem_id="btn-enviar", scale=0)

                    btn_limpar = gr.Button("limpar conversa", elem_id="btn-limpar", size="sm")

            # ================================================================
            # Vista: Transcrição
            # ================================================================
            with gr.Column(elem_id="main-trans", scale=1, visible=False) as col_trans:

                gr.Markdown("**🎙️ Transcrição**", elem_id="trans-header")

                with gr.Column(elem_id="trans-content"):
                    audio_input = gr.File(
                        label="Arquivo de áudio ou vídeo (mp3, mp4, wav, ogg...)",
                        file_types=["audio", "video"],
                    )

                    with gr.Accordion("📋 Metadados da audiência (opcional)", open=False):
                        txt_processo = gr.Textbox(
                            label="Número do Processo (CNJ)",
                            placeholder="Ex: 1234567-89.2024.8.26.0100",
                        )
                        txt_data_audiencia = gr.Textbox(
                            label="Data da Audiência",
                            placeholder="Ex: 15/01/2025",
                        )
                        txt_tipo_audiencia = gr.Textbox(
                            label="Tipo da Audiência",
                            value="Instrução e Julgamento",
                        )
                        txt_vara = gr.Textbox(
                            label="Vara / Tribunal",
                            placeholder="Ex: 3ª Vara Cível — Foro Central",
                        )
                        txt_cidade_uf = gr.Textbox(
                            label="Cidade / UF",
                            placeholder="Ex: São Paulo/SP",
                        )
                        with gr.Row():
                            sl_min_speakers = gr.Slider(
                                label="Mín. falantes", minimum=1, maximum=4, value=2, step=1,
                            )
                            sl_max_speakers = gr.Slider(
                                label="Máx. falantes", minimum=2, maximum=10, value=6, step=1,
                            )
                        chk_identificar_llm = gr.Checkbox(
                            label="Identificar participantes com LLM",
                            value=True,
                        )

                    btn_transcrever = gr.Button(
                        "🎙️ Iniciar Transcrição", variant="primary", size="lg",
                    )
                    out_status_transcricao = gr.Markdown("")
                    out_transcricao = gr.Markdown(
                        "_Selecione um arquivo e clique em **Iniciar Transcrição**._"
                    )

        # ====================================================================
        # Eventos
        # ====================================================================

        # Navegação sidebar
        btn_nav_chat.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=[],
            outputs=[col_chat, col_trans],
        )
        btn_nav_trans.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[col_chat, col_trans],
        )

        # Top-k +/-
        btn_k_minus.click(fn=fn_top_k_dec, inputs=[num_top_k], outputs=[num_top_k])
        btn_k_plus.click(fn=fn_top_k_inc, inputs=[num_top_k], outputs=[num_top_k])

        # Chat
        _inputs_chat = [txt_msg, chatbot, estado_backend, dd_area, num_top_k, rd_vigencia]
        btn_enviar.click(fn=fn_chat, inputs=_inputs_chat, outputs=[chatbot, txt_msg])
        txt_msg.submit(fn=fn_chat, inputs=_inputs_chat, outputs=[chatbot, txt_msg])
        btn_limpar.click(fn=lambda: ([], ""), inputs=[], outputs=[chatbot, txt_msg])

        # Transcrição
        btn_transcrever.click(
            fn=fn_transcrever,
            inputs=[
                estado_backend, audio_input,
                txt_processo, txt_data_audiencia, txt_tipo_audiencia,
                txt_vara, txt_cidade_uf,
                sl_min_speakers, sl_max_speakers, chk_identificar_llm,
            ],
            outputs=[out_transcricao, out_status_transcricao],
        )

    return interface


# =============================================================================
# Inicialização
# =============================================================================

def iniciar_interface() -> None:
    """Ponto de entrada do script ana-ui."""
    parser = argparse.ArgumentParser(description="Frontend Gradio do sistema ANA")
    parser.add_argument("--backend", default=_URL_BACKEND_PADRAO)
    parser.add_argument("--porta", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--compartilhar", action="store_true")
    args = parser.parse_args()

    interface = criar_interface(url_backend=args.backend)
    interface.launch(
        server_name=args.host,
        server_port=args.porta,
        share=args.compartilhar,
        theme=gr.themes.Base(),
        css=_CSS,
    )


if __name__ == "__main__":
    iniciar_interface()
