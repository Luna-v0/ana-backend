"""Módulo de transcrição de audiências judiciais do sistema ANA.

Implementa o pipeline de transcrição com diarização de falantes:
1. MotorASR (WhisperX) — transcrição speech-to-text com alinhamento word-level
2. DiarizadorSpeaker (pyannote via WhisperX) — identificação de quem fala
3. Identificação de participantes (metadata + LLM por contexto)
4. Formatação markdown com timestamps e labels dos participantes

Dependências opcionais (instalar com: uv sync --group transcricao):
    - whisperx (inclui pyannote.audio)
    - torch (CUDA recomendado para performance aceitável)

Requer token do HuggingFace para pyannote:
    export HF_TOKEN=hf_seu_token_aqui
    Aceitar os termos em: https://huggingface.co/pyannote/speaker-diarization-3.1

Nota (LGPD):
    Arquivos de áudio de audiências contêm dados pessoais sensíveis (Art. 5, II LGPD).
    Todo processamento é 100% local. Nenhum áudio ou transcrição sai da máquina.
"""

from .diarizacao import DiarizadorSpeaker
from .formatacao import formatar_transcript_markdown
from .identificacao import identificar_participantes
from .modelos import (
    MetadataAudiencia,
    ParticipanteAudiencia,
    ResultadoTranscricao,
    RoleParticipante,
    SegmentoTranscricao,
    aplicar_mapeamento,
    extrair_mapeamento_json,
)
from .transcricao import MotorASR

__all__ = [
    "MotorASR",
    "DiarizadorSpeaker",
    "formatar_transcript_markdown",
    "identificar_participantes",
    "MetadataAudiencia",
    "ParticipanteAudiencia",
    "ResultadoTranscricao",
    "RoleParticipante",
    "SegmentoTranscricao",
    "aplicar_mapeamento",
    "extrair_mapeamento_json",
]
