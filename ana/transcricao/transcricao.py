"""Motor de transcrição ASR via WhisperX.

Encapsula o pipeline WhisperX com carregamento lazy dos modelos:
1. Transcrição Whisper large-v3 (ASR)
2. Alinhamento word-level (force alignment)

WhisperX é uma dependência opcional (grupo 'transcricao' do pyproject.toml).
Importar este módulo sem o grupo instalado lança ImportError com instrução clara.

Nota (LGPD):
    Todo processamento de áudio é 100% local. Nenhuma amostra de voz
    ou transcrição sai da máquina do advogado.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from .modelos import SegmentoTranscricao


def _verificar_whisperx() -> None:
    """Verifica se whisperx está instalado e levanta erro claro se não estiver.

    Raises:
        ImportError: Se whisperx não estiver instalado com instrução de como instalar.
    """
    try:
        import whisperx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "whisperx não está instalado. Instale o grupo de dependências de transcrição:\n"
            "  uv sync --group transcricao\n"
            "Também requer CUDA e PyTorch instalados."
        ) from e


class MotorASR:
    """Motor de transcrição ASR usando WhisperX com carregamento lazy.

    O modelo é carregado na memória apenas na primeira chamada a transcrever(),
    evitando alocar VRAM desnecessariamente quando o módulo é importado.

    Attributes:
        modelo_nome: Nome do modelo Whisper (ex: 'large-v3').
        dispositivo: Dispositivo de computação ('cuda' ou 'cpu').
        idioma: Código do idioma para transcrição (ex: 'pt').
        batch_size: Tamanho do lote para inferência (ajustar conforme VRAM disponível).
        compute_type: Tipo de computação ('float16' para GPU, 'int8' para CPU).
    """

    def __init__(
        self,
        modelo_nome: str = "large-v3",
        dispositivo: str = "cuda",
        idioma: str = "pt",
        batch_size: int = 16,
        compute_type: str = "float16",
    ) -> None:
        """Inicializa o motor ASR sem carregar o modelo ainda.

        Args:
            modelo_nome: Nome do modelo Whisper a usar.
            dispositivo: 'cuda' para GPU (recomendado) ou 'cpu'.
            idioma: Código ISO do idioma (padrão: 'pt' para português).
            batch_size: Lotes para processamento paralelo (reduzir se faltar VRAM).
            compute_type: Precisão numérica ('float16' para GPU, 'int8' para CPU).
        """
        _verificar_whisperx()
        self.modelo_nome = modelo_nome
        self.dispositivo = dispositivo
        self.idioma = idioma
        self.batch_size = batch_size
        self.compute_type = compute_type
        self._modelo: Any = None
        self._modelo_alinhamento: Any = None
        self._metadata_alinhamento: Any = None

    def _carregar_modelo(self) -> None:
        """Carrega o modelo Whisper e o modelo de alinhamento na VRAM."""
        import whisperx

        if self._modelo is None:
            logger.info(
                f"Carregando WhisperX {self.modelo_nome} em {self.dispositivo} "
                f"(compute_type={self.compute_type})"
            )
            self._modelo = whisperx.load_model(
                self.modelo_nome,
                self.dispositivo,
                language=self.idioma,
                compute_type=self.compute_type,
            )
            logger.info("Modelo ASR carregado")

        if self._modelo_alinhamento is None:
            logger.info("Carregando modelo de alinhamento word-level")
            self._modelo_alinhamento, self._metadata_alinhamento = (
                whisperx.load_align_model(
                    language_code=self.idioma,
                    device=self.dispositivo,
                )
            )
            logger.info("Modelo de alinhamento carregado")

    def transcrever(self, caminho_audio: str | Path) -> list[SegmentoTranscricao]:
        """Transcreve um arquivo de áudio e retorna segmentos com timestamps.

        Executa o pipeline completo:
        1. Transcrição ASR com Whisper large-v3
        2. Alinhamento word-level para timestamps precisos

        A diarização (quem falou) é feita separadamente em DiarizadorSpeaker.

        Args:
            caminho_audio: Caminho para o arquivo de áudio (mp3, wav, m4a, etc.).

        Returns:
            Lista de SegmentoTranscricao ordenados por tempo de início.
            Os campos speaker_id ainda são genéricos (SPEAKER_00) até a diarização.

        Raises:
            FileNotFoundError: Se o arquivo de áudio não existir.
            RuntimeError: Se ocorrer erro durante a transcrição.
        """
        import whisperx

        caminho = Path(caminho_audio)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {caminho}")

        self._carregar_modelo()

        logger.info(f"Transcrevendo: {caminho.name}")
        audio = whisperx.load_audio(str(caminho))
        resultado_asr = self._modelo.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.idioma,
        )

        # Alinha timestamps word-level para maior precisão de segmentação
        resultado_alinhado = whisperx.align(
            resultado_asr["segments"],
            self._modelo_alinhamento,
            self._metadata_alinhamento,
            audio,
            self.dispositivo,
            return_char_alignments=False,
        )

        segmentos = [
            SegmentoTranscricao(
                inicio=float(seg.get("start", 0.0)),
                fim=float(seg.get("end", 0.0)),
                texto=seg.get("text", "").strip(),
                speaker_id="SPEAKER_PENDENTE",  # atribuído pela diarização
            )
            for seg in resultado_alinhado["segments"]
            if seg.get("text", "").strip()
        ]

        logger.info(f"Transcrição concluída: {len(segmentos)} segmentos")
        return segmentos

    def liberar_modelo(self) -> None:
        """Libera o modelo da VRAM quando não for mais necessário."""
        import gc
        import torch

        self._modelo = None
        self._modelo_alinhamento = None
        self._metadata_alinhamento = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Modelo ASR liberado da VRAM")
