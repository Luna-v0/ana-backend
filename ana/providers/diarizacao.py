"""Protocol para provedores de diarização de falantes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ana.transcricao.modelos import SegmentoTranscricao


@runtime_checkable
class DiarizacaoProvider(Protocol):
    """Contrato para diarizadores de falantes.

    Conforme: DiarizadorSpeaker em transcricao/diarizacao.py
    """

    def diarizar(
        self,
        caminho_audio: str | Path,
        segmentos_asr: list[SegmentoTranscricao],
    ) -> list[SegmentoTranscricao]: ...

    def liberar_pipeline(self) -> None: ...
