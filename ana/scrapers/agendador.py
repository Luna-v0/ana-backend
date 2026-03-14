"""Agendador de scrapers para o ANA.

Thin wrapper sobre :class:`leis_br.AgendadorScrapers` que injeta o
:class:`~ana.scrapers.pipeline.PipelineScrapers` com ingestão RAG.

Requer dependência opcional: uv sync --group scrapers
"""

from loguru import logger

try:
    from leis_br import AgendadorScrapers as _LeisBrAgendador
    _BASE_AGENDADOR: type = _LeisBrAgendador
except ImportError:
    _BASE_AGENDADOR = object

# Intervalo de coleta por fonte (em horas) — 7 dias para todas as fontes
INTERVALOS_HORAS: dict[str, int] = {
    "planalto": 168,
    "lexml": 168,
    "stf": 168,
    "stj": 168,
    "tst": 168,
}


class AgendadorScrapers(_BASE_AGENDADOR):  # type: ignore[misc]
    """AgendadorScrapers do ANA com ingestão RAG completa.

    Cria automaticamente o :class:`~ana.scrapers.pipeline.PipelineScrapers`
    (com chunking + embeddings + Qdrant) antes de iniciar o agendamento.
    """

    def iniciar(self) -> None:
        if _BASE_AGENDADOR is object:
            logger.warning("Agendador desativado: leis-br não instalado.")
            return

        if self._pipeline is None:
            from ana.scrapers.pipeline import PipelineScrapers
            self._pipeline = PipelineScrapers()

        super().iniciar()
