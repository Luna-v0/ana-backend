"""Módulo de geração de documentos jurídicos do sistema ANA.

Gera peças processuais (petição inicial, contestação, recurso) como
documentos Word (.docx) usando python-docx, com conteúdo produzido
pelo LLM redator fundamentado em busca RAG de legislação e jurisprudência.

Uso::

    from ana.documentos.gerador import gerar_documento
    from ana.documentos.modelos import TipoPeca

    docx_bytes, nome = gerar_documento(
        sessao_id="sess_abc123",
        tipo_peca=TipoPeca.PETICAO_INICIAL,
    )
"""

from ana.documentos.gerador import gerar_documento
from ana.documentos.modelos import RequisicaoGerarDocumento, TipoPeca

__all__ = ["gerar_documento", "RequisicaoGerarDocumento", "TipoPeca"]
