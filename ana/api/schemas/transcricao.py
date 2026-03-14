"""Contratos de input/output para o router /transcricao."""

from pydantic import BaseModel


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
    participante_nome: str | None = None
    participante_role: str | None = None


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
