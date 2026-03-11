"""Sistema de sessões de processos jurídicos (Spec 06).

Gerencia sessões por número de processo, com upload e indexação
de documentos (PDF, DOCX, TXT) em tabela vetorial isolada por sessao_id.

Uso:
    >>> from ana.sessoes.repositorio import criar_sessao, obter_sessao
    >>> from ana.sessoes.ingestao import ingerir_documento_sessao
    >>> from ana.sessoes.busca import buscar_intra_sessao, buscar_global
"""

from ana.sessoes.modelos import DocumentoSessao, Sessao
from ana.sessoes.repositorio import inicializar_banco

__all__ = [
    "Sessao",
    "DocumentoSessao",
    "inicializar_banco",
]
