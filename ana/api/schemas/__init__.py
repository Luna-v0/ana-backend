"""Contratos de API do sistema ANA.

Cada módulo define os schemas Pydantic (request + response) de um router.
Os routers importam daqui e não definem modelos inline.

Módulos:
    chat        → /chat
    health      → /health
    rag         → /rag
    redacao     → /redacao
    scrapers    → /scrapers
    sessoes     → /sessoes
    transcricao → /transcricao
"""

from ana.api.schemas.chat import EventoSSE, RequisicaoChat, RespostaChat
from ana.api.schemas.health import InfoModelos, RespostaHealth, StatusServico
from ana.api.schemas.rag import (
    RequiscaoBusca,
    RequiscaoIngestao,
    RequisicaoResumir,
    RespostaBusca,
    RespostaIngestao,
    ResultadoChunk,
    StatusRAG,
)
from ana.api.schemas.redacao import RequisicaoReformular, RespostaReformular
from ana.api.schemas.scrapers import (
    InfoFonte,
    RequisicaoColeta,
    RespostaAtualizarTudo,
    RespostaColeta,
    RespostaFontes,
    StatusFonte,
    StatusScrapers,
)
from ana.api.schemas.sessoes import (
    RequisicaoAtualizarSessao,
    RequisicaoBuscaSessao,
    RequisicaoCriarSessao,
    RespostaBuscaSessao,
    RespostaDocumento,
    RespostaSessao,
    ResultadoBuscaSessao,
)
from ana.api.schemas.transcricao import (
    RespostaTranscricao,
    SegmentoResposta,
    StatusTranscricao,
)

__all__ = [
    # chat
    "RequisicaoChat",
    "RespostaChat",
    "EventoSSE",
    # health
    "StatusServico",
    "InfoModelos",
    "RespostaHealth",
    # rag
    "RequiscaoIngestao",
    "RespostaIngestao",
    "RequiscaoBusca",
    "ResultadoChunk",
    "RespostaBusca",
    "RequisicaoResumir",
    "StatusRAG",
    # redacao
    "RequisicaoReformular",
    "RespostaReformular",
    # scrapers
    "RequisicaoColeta",
    "RespostaColeta",
    "RespostaAtualizarTudo",
    "StatusFonte",
    "StatusScrapers",
    "InfoFonte",
    "RespostaFontes",
    # sessoes
    "RequisicaoCriarSessao",
    "RequisicaoAtualizarSessao",
    "RespostaSessao",
    "RespostaDocumento",
    "RequisicaoBuscaSessao",
    "ResultadoBuscaSessao",
    "RespostaBuscaSessao",
    # transcricao
    "StatusTranscricao",
    "SegmentoResposta",
    "RespostaTranscricao",
]
