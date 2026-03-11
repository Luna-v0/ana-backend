"""Protocol para provedores de ASR (Automatic Speech Recognition)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ana.transcricao.modelos import SegmentoTranscricao


@runtime_checkable
class ASRProvider(Protocol):
    """Contrato para motores de transcrição ASR.

    Conforme: MotorASR em transcricao/transcricao.py
    """

    def transcrever(self, caminho_audio: str | Path) -> list[SegmentoTranscricao]: ...

    def liberar_modelo(self) -> None: ...
