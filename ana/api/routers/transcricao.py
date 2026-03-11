"""Router FastAPI para o módulo de transcrição de audiências.

Endpoints:
    GET  /transcricao/status  — Verifica disponibilidade do módulo e dependências
    POST /transcricao/transcrever — Transcreve um arquivo de áudio de audiência

Nota sobre performance:
    Uma audiência de 1h com Whisper large-v3 em GPU NVIDIA RTX 3060 leva ~6-12 min.
    O endpoint é síncrono — a conexão fica aberta durante todo o processamento.
    O timeout do cliente deve ser configurado de acordo (>= 15 minutos).

Nota (LGPD):
    Arquivos de áudio são processados em memória temporária e nunca persistidos
    pelo backend. O arquivo original permanece apenas na máquina do advogado.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from ana.config_modelos import obter_modelos

router = APIRouter(prefix="/transcricao", tags=["Transcrição"])


# ── Modelos de resposta ────────────────────────────────────────────────────────

class StatusTranscricao(BaseModel):
    """Status do módulo de transcrição.

    Attributes:
        disponivel: True se as dependências opcionais estão instaladas.
        whisperx_instalado: True se whisperx está importável.
        hf_token_configurado: True se HF_TOKEN está definido no ambiente.
        dispositivo: Dispositivo que será usado ('cuda' ou 'cpu').
        modelo_asr: Modelo Whisper configurado para uso.
        mensagem: Instrução de instalação se módulo indisponível.
    """

    disponivel: bool
    whisperx_instalado: bool
    hf_token_configurado: bool
    dispositivo: str
    modelo_asr: str
    mensagem: str = ""


class SegmentoResposta(BaseModel):
    """Segmento individual da transcrição para serialização JSON.

    Attributes:
        inicio: Timestamp de início em segundos.
        fim: Timestamp de fim em segundos.
        texto: Texto transcrito.
        speaker_id: Label do diarizador (ex: SPEAKER_00).
        timestamp_formatado: Início formatado como MM:SS ou HH:MM:SS.
        participante_nome: Nome do participante identificado (se disponível).
        participante_role: Role do participante (se identificado).
    """

    inicio: float
    fim: float
    texto: str
    speaker_id: str
    timestamp_formatado: str
    participante_nome: Optional[str] = None
    participante_role: Optional[str] = None


class RespostaTranscricao(BaseModel):
    """Resposta completa da transcrição de uma audiência.

    Attributes:
        segmentos: Lista de segmentos transcritos.
        mapeamento_speakers: Dicionário SPEAKER_XX → nome do participante.
        duracao_total: Duração total do áudio em segundos.
        num_participantes: Número de speakers únicos identificados.
        idioma: Idioma detectado.
        modelo_asr: Modelo de transcrição usado.
        modelo_diarizacao: Modelo de diarização usado.
        markdown: Transcrição completa formatada como markdown.
        arquivo_processado: Nome do arquivo de áudio processado.
    """

    segmentos: list[SegmentoResposta]
    mapeamento_speakers: dict[str, str]
    duracao_total: float
    num_participantes: int
    idioma: str
    modelo_asr: str
    modelo_diarizacao: str
    markdown: str
    arquivo_processado: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _verificar_whisperx_disponivel() -> bool:
    """Verifica se whisperx está instalado."""
    try:
        import whisperx  # noqa: F401
        return True
    except ImportError:
        return False


def _obter_dispositivo() -> str:
    """Retorna 'cuda' se GPU disponível, senão 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=StatusTranscricao,
    summary="Status do módulo de transcrição",
    description=(
        "Verifica se as dependências opcionais de transcrição estão instaladas "
        "(whisperx, pyannote) e se o token do HuggingFace está configurado. "
        "Também detecta disponibilidade de GPU (CUDA)."
    ),
)
async def status_transcricao() -> StatusTranscricao:
    """Retorna disponibilidade e configuração do módulo de transcrição.

    Returns:
        StatusTranscricao com detalhes sobre dependências e dispositivo disponível.
    """
    modelos = obter_modelos()
    whisperx_ok = _verificar_whisperx_disponivel()
    hf_token_ok = bool(os.environ.get("HF_TOKEN"))
    dispositivo = _obter_dispositivo()

    disponivel = whisperx_ok and hf_token_ok

    if not whisperx_ok:
        mensagem = (
            "whisperx não instalado. Execute: uv sync --group transcricao"
        )
    elif not hf_token_ok:
        mensagem = (
            "HF_TOKEN não configurado. Aceite os termos em "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "e configure: export HF_TOKEN=hf_seu_token"
        )
    else:
        mensagem = "Módulo de transcrição disponível"

    return StatusTranscricao(
        disponivel=disponivel,
        whisperx_instalado=whisperx_ok,
        hf_token_configurado=hf_token_ok,
        dispositivo=dispositivo,
        modelo_asr="large-v3",
        mensagem=mensagem,
    )


@router.post(
    "/transcrever",
    response_model=RespostaTranscricao,
    summary="Transcrever audiência judicial",
    description=(
        "Recebe um arquivo de áudio (mp3, wav, m4a, ogg, flac) e executa o pipeline "
        "completo: ASR (Whisper large-v3) + alinhamento word-level + diarização "
        "(pyannote) + identificação de participantes (metadata + LLM). "
        "AVISO: operação longa — uma audiência de 1h leva ~6-12 min em GPU."
    ),
)
async def transcrever_audiencia(
    audio: Annotated[UploadFile, File(description="Arquivo de áudio da audiência (mp3, wav, m4a, ogg, flac)")],
    numero_processo: Annotated[str, Form()] = "",
    data_audiencia: Annotated[str, Form()] = "",
    tipo_audiencia: Annotated[str, Form()] = "Instrução e Julgamento",
    vara: Annotated[str, Form()] = "",
    cidade_uf: Annotated[str, Form()] = "",
    min_speakers: Annotated[int, Form()] = 2,
    max_speakers: Annotated[int, Form()] = 6,
    identificar_por_llm: Annotated[bool, Form()] = True,
    juiz: Annotated[str, Form()] = "",
    advogado_autor: Annotated[str, Form()] = "",
    advogado_reu: Annotated[str, Form()] = "",
    testemunha_1: Annotated[str, Form()] = "",
    testemunha_2: Annotated[str, Form()] = "",
    perito: Annotated[str, Form()] = "",
) -> RespostaTranscricao:
    """Transcreve arquivo de áudio de audiência com diarização de falantes.

    Pipeline:
    1. Salva o arquivo de áudio em diretório temporário (nunca persistido)
    2. Executa ASR com Whisper large-v3 + alinhamento word-level
    3. Executa diarização com pyannote (quem falou quando)
    4. Identifica participantes via metadata + LLM (Camadas 1 e 2)
    5. Formata transcrição como markdown estruturado

    Args:
        audio: Arquivo de áudio enviado via multipart/form-data.
        numero_processo: Número do processo no formato CNJ.
        data_audiencia: Data da audiência (dd/mm/aaaa).
        tipo_audiencia: Tipo da audiência judicial.
        vara: Vara e foro onde ocorreu a audiência.
        cidade_uf: Cidade e UF do foro.
        min_speakers: Número mínimo de falantes esperados.
        max_speakers: Número máximo de falantes esperados.
        identificar_por_llm: Se True, usa LLM para identificar speakers por contexto.

    Returns:
        RespostaTranscricao com segmentos, mapeamento de speakers e markdown.

    Raises:
        HTTPException 503: Se o módulo de transcrição não estiver disponível.
        HTTPException 422: Se o arquivo não for um formato de áudio suportado.
        HTTPException 500: Se ocorrer erro durante o processamento.
    """
    from ana.transcricao import (
        DiarizadorSpeaker,
        MetadataAudiencia,
        MotorASR,
        ResultadoTranscricao,
        formatar_transcript_markdown,
        identificar_participantes,
    )
    from ana.transcricao.modelos import ParticipanteAudiencia, RoleParticipante

    # Verifica disponibilidade
    if not _verificar_whisperx_disponivel():
        raise HTTPException(
            status_code=503,
            detail=(
                "Módulo de transcrição não disponível. "
                "Instale: uv sync --group transcricao"
            ),
        )

    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(
            status_code=503,
            detail=(
                "HF_TOKEN não configurado. Configure: export HF_TOKEN=hf_seu_token"
            ),
        )

    # Valida extensão do arquivo
    extensoes_suportadas = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}
    nome_arquivo = audio.filename or "audio.mp3"
    sufixo = Path(nome_arquivo).suffix.lower()
    if sufixo not in extensoes_suportadas:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Formato de áudio não suportado: '{sufixo}'. "
                f"Formatos aceitos: {', '.join(sorted(extensoes_suportadas))}"
            ),
        )

    config = obter_modelos()
    perfil = config.ativo
    dispositivo = _obter_dispositivo()

    # Salva arquivo em diretório temporário (limpo automaticamente)
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho_audio = Path(tmpdir) / nome_arquivo
        try:
            with caminho_audio.open("wb") as f:
                shutil.copyfileobj(audio.file, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao salvar áudio: {e}") from e

        logger.info(
            f"Processando audiência: {nome_arquivo} "
            f"({caminho_audio.stat().st_size / 1024 / 1024:.1f} MB)"
        )

        try:
            # Etapa 1: Transcrição ASR
            motor_asr = MotorASR(
                modelo_nome="large-v3",
                dispositivo=dispositivo,
                idioma="pt",
                batch_size=16 if dispositivo == "cuda" else 4,
                compute_type="float16" if dispositivo == "cuda" else "int8",
            )
            segmentos = motor_asr.transcrever(caminho_audio)
            motor_asr.liberar_modelo()

            # Etapa 2: Diarização
            diarizador = DiarizadorSpeaker(
                dispositivo=dispositivo,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            segmentos = diarizador.diarizar(caminho_audio, segmentos)
            diarizador.liberar_pipeline()

            # Etapa 3: Identificação de participantes
            _MAPA_CAMPOS_ROLES = [
                ("juiz",           juiz,           RoleParticipante.JUIZ),
                ("advogado_autor", advogado_autor, RoleParticipante.ADVOGADO_AUTOR),
                ("advogado_reu",   advogado_reu,   RoleParticipante.ADVOGADO_REU),
                ("testemunha_1",   testemunha_1,   RoleParticipante.TESTEMUNHA),
                ("testemunha_2",   testemunha_2,   RoleParticipante.TESTEMUNHA),
                ("perito",         perito,          RoleParticipante.PERITO),
            ]
            participantes_esperados = {
                chave: ParticipanteAudiencia(nome=nome.strip(), role=role)
                for chave, nome, role in _MAPA_CAMPOS_ROLES
                if nome.strip()
            }
            metadata = MetadataAudiencia(
                numero_processo=numero_processo,
                data=data_audiencia,
                tipo_audiencia=tipo_audiencia,
                vara=vara,
                cidade_uf=cidade_uf,
                participantes_esperados=participantes_esperados,
            )

            if identificar_por_llm:
                from ana.config import obter_configuracao
                from ana.providers import OllamaLLMProvider

                provedor_llm = OllamaLLMProvider(
                    modelo=perfil.agentes.pesquisador,
                    host=obter_configuracao().ollama_host,
                )
                segmentos, mapeamento = identificar_participantes(
                    segmentos,
                    metadata,
                    provedor_llm=provedor_llm,
                )
            else:
                from ana.transcricao.modelos import aplicar_mapeamento
                mapeamento = {}
                segmentos = aplicar_mapeamento(segmentos, mapeamento)

            # Calcula duração total
            duracao_total = max((s.fim for s in segmentos), default=0.0)

            # Etapa 4: Formata resultado
            resultado = ResultadoTranscricao(
                segmentos=segmentos,
                metadata=metadata,
                mapeamento_speakers=mapeamento,
                duracao_total=duracao_total,
                modelo_asr="whisper-large-v3",
                modelo_diarizacao="pyannote/speaker-diarization-3.1",
                idioma="pt",
                arquivo_origem=nome_arquivo,
            )
            markdown = formatar_transcript_markdown(resultado)

        except Exception as e:
            logger.error(f"Erro durante transcrição: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro durante processamento: {type(e).__name__}: {e}",
            ) from e

    # Serializa para resposta
    segmentos_resposta = [
        SegmentoResposta(
            inicio=seg.inicio,
            fim=seg.fim,
            texto=seg.texto,
            speaker_id=seg.speaker_id,
            timestamp_formatado=seg.timestamp_formatado,
            participante_nome=seg.participante.nome if seg.participante else None,
            participante_role=seg.participante.role.value if seg.participante else None,
        )
        for seg in segmentos
    ]

    mapeamento_nomes = {
        speaker_id: p.label_formatado
        for speaker_id, p in mapeamento.items()
    }

    return RespostaTranscricao(
        segmentos=segmentos_resposta,
        mapeamento_speakers=mapeamento_nomes,
        duracao_total=duracao_total,
        num_participantes=resultado.num_participantes,
        idioma="pt",
        modelo_asr="whisper-large-v3",
        modelo_diarizacao="pyannote/speaker-diarization-3.1",
        markdown=markdown,
        arquivo_processado=nome_arquivo,
    )
