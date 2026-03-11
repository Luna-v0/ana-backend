"""Módulo de scrapers do ANA — bridge sobre o pacote leis-br.

Coleta automaticamente legislação e jurisprudência de fontes públicas
brasileiras (Planalto, LexML, STF, STJ, TST) e alimenta o pipeline RAG.

Os scrapers, modelos e cache residem agora no pacote ``leis-br``.
Este módulo expõe a interface RAG-aware (com chunking + embeddings).

Fontes implementadas (via leis-br):
    planalto   → Principais leis federais compiladas
    lexml      → Legislação estruturada em XML (Senado Federal) — API descontinuada
    stf        → Súmulas ordinárias e vinculantes do STF — portal SPA (TODO)
    stj        → Súmulas do STJ — Cloudflare (TODO)
    tst        → Súmulas do TST — React SPA (TODO)

Uso rápido:
    >>> from ana.scrapers.pipeline import PipelineScrapers
    >>> pipeline = PipelineScrapers()
    >>> resultado = pipeline.coletar_fonte("planalto")
    >>> print(resultado.documentos_novos)
"""

# Modelos re-exportados de leis-br (quando disponível)
try:
    from leis_br.modelos import DocumentoColetado, ResultadoColeta  # noqa: F401
    __all__ = ["DocumentoColetado", "ResultadoColeta"]
except ImportError:
    __all__ = []
