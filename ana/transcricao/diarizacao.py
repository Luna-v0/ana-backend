"""Diarização de falantes via pyannote.audio.

Identifica "quem falou quando" no áudio da audiência:
- VAD (Voice Activity Detection) para filtrar silêncio
- Segmentação de trechos de fala
- Clustering de vozes similares
- Atribuição de speaker labels aos segmentos ASR

pyannote.audio é uma dependência opcional (grupo 'transcricao').
Requer token do HuggingFace para download do modelo.

Nota (LGPD):
    Vozes de partes e testemunhas são dados biométricos sensíveis (Art. 5, II LGPD).
    Nenhuma amostra de voz ou embedding de voz sai da máquina.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from .modelos import SegmentoTranscricao


def _verificar_pyannote() -> None:
    """Verifica se whisperx com suporte a diarização está instalado.

    Raises:
        ImportError: Se whisperx não estiver instalado.
        ValueError: Se HF_TOKEN não estiver configurado (necessário para pyannote).
    """
    try:
        import whisperx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "whisperx não está instalado. Para diarização, instale:\n"
            "  uv sync --group transcricao\n"
            "Também requer token do HuggingFace configurado em HF_TOKEN."
        ) from e

    if not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN não configurado. O pyannote.audio requer aceitar os termos "
            "em https://huggingface.co/pyannote/speaker-diarization-3.1 "
            "e configurar o token: export HF_TOKEN=seu_token_aqui"
        )


class DiarizadorSpeaker:
    """Diarizador de falantes usando pyannote via WhisperX.

    Identifica "quem falou quando" e atribui speaker labels
    (SPEAKER_00, SPEAKER_01, ...) aos segmentos do ASR.

    Attributes:
        dispositivo: Dispositivo de computação ('cuda' ou 'cpu').
        min_speakers: Número mínimo esperado de falantes.
        max_speakers: Número máximo esperado de falantes.
        hf_token: Token do HuggingFace para download do modelo pyannote.
    """

    def __init__(
        self,
        dispositivo: str = "cuda",
        min_speakers: int = 2,
        max_speakers: int = 6,
        hf_token: str | None = None,
    ) -> None:
        """Inicializa o diarizador sem carregar o modelo ainda.

        Args:
            dispositivo: 'cuda' para GPU (recomendado) ou 'cpu'.
            min_speakers: Número mínimo de falantes esperados na audiência.
            max_speakers: Número máximo de falantes (tipicamente 6 em audiências).
            hf_token: Token do HuggingFace. Se None, lê de HF_TOKEN do ambiente.
        """
        _verificar_pyannote()
        self.dispositivo = dispositivo
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._pipeline: Any = None

    @staticmethod
    def _patch_pyannote_plda() -> None:
        """Aplica monkey-patch no pyannote.audio para ignorar PLDA inacessível.

        O pyannote.audio >= 4.0 sempre tenta carregar um PLDA
        (pyannote/speaker-diarization-community-1) como efeito colateral do
        construtor SpeakerDiarization, mesmo quando o modelo
        speaker-diarization-3.1 usa AgglomerativeClustering — que NÃO usa PLDA.

        O patch faz get_plda retornar None silenciosamente quando o repositório
        é inacessível (GatedRepoError / HfHubHTTPError), evitando falha na
        inicialização sem afetar a qualidade da diarização.
        """
        try:
            import pyannote.audio.pipelines.speaker_diarization as _sd
            import pyannote.audio.pipelines.utils.getter as _getter
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        except ImportError:
            return

        _original = _getter.get_plda

        def _safe_get_plda(plda, token=None, cache_dir=None):
            if plda is None:
                return None
            try:
                return _original(plda, token=token, cache_dir=cache_dir)
            except (GatedRepoError, HfHubHTTPError):
                # PLDA só é necessário para VBxClustering.
                # speaker-diarization-3.1 usa AgglomerativeClustering → safe to skip.
                return None

        _getter.get_plda = _safe_get_plda
        _sd.get_plda = _safe_get_plda

    def _carregar_pipeline(self) -> None:
        """Carrega o pipeline de diarização do pyannote via WhisperX."""
        from whisperx.diarize import DiarizationPipeline

        if self._pipeline is None:
            logger.info(
                f"Carregando pipeline de diarização (pyannote) em {self.dispositivo}"
            )
            # Aplica patch antes de instanciar para contornar bug do pyannote 4.x
            # que tenta carregar PLDA desnecessário (inacessível sem acesso especial).
            self._patch_pyannote_plda()
            self._pipeline = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=self.hf_token,
                device=self.dispositivo,
            )
            logger.info("Pipeline de diarização carregado")

    def diarizar(
        self,
        caminho_audio: str | Path,
        segmentos_asr: list[SegmentoTranscricao],
    ) -> list[SegmentoTranscricao]:
        """Executa diarização e atribui speaker_id aos segmentos ASR.

        Pipeline:
        1. Roda pipeline pyannote no áudio para obter segmentos diarizados
        2. Usa whisperx.assign_word_speakers para alinhar com segmentos ASR

        Args:
            caminho_audio: Caminho para o mesmo arquivo de áudio transcrito.
            segmentos_asr: Segmentos da transcrição ASR (sem speaker definido).

        Returns:
            Segmentos com speaker_id (SPEAKER_00, SPEAKER_01, ...) atribuídos.

        Raises:
            FileNotFoundError: Se o arquivo de áudio não existir.
            RuntimeError: Se a diarização falhar.
        """
        import whisperx

        caminho = Path(caminho_audio)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho}")

        self._carregar_pipeline()

        logger.info(
            f"Diarizando: {caminho.name} "
            f"(min_speakers={self.min_speakers}, max_speakers={self.max_speakers})"
        )

        # Pré-carrega o áudio com whisperx (usa soundfile, não torchcodec).
        # DiarizationPipeline.__call__ aceita numpy array (float32 @ 16kHz) e
        # constrói internamente o dict {waveform, sample_rate} que passa ao pyannote.
        audio_numpy = whisperx.load_audio(str(caminho))  # numpy float32 @ 16kHz

        # Executa diarização para obter segmentos por locutor
        segmentos_diarizados = self._pipeline(
            audio_numpy,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # Converte segmentos ASR para o formato esperado pelo whisperx
        segmentos_dict = [
            {
                "start": seg.inicio,
                "end": seg.fim,
                "text": seg.texto,
                "words": [],  # já alinhado na etapa ASR
            }
            for seg in segmentos_asr
        ]
        resultado_com_speakers = {"segments": segmentos_dict}

        # Atribui speakers aos segmentos ASR
        resultado_atribuido = whisperx.assign_word_speakers(
            segmentos_diarizados,
            resultado_com_speakers,
        )

        # Atualiza os segmentos com o speaker_id
        speakers_encontrados: set[str] = set()
        for i, seg_dict in enumerate(resultado_atribuido["segments"]):
            speaker = seg_dict.get("speaker", f"SPEAKER_{i:02d}")
            if i < len(segmentos_asr):
                segmentos_asr[i].speaker_id = speaker
                speakers_encontrados.add(speaker)

        logger.info(
            f"Diarização concluída: {len(speakers_encontrados)} falante(s) identificado(s)"
        )
        return segmentos_asr

    def liberar_pipeline(self) -> None:
        """Libera o pipeline da VRAM quando não for mais necessário."""
        import gc
        import torch

        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline de diarização liberado da VRAM")
