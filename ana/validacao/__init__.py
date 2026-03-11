"""Módulo de validação anti-alucinação do sistema ANA (Spec 07).

Pipeline de 3 camadas para verificação de leis citadas em respostas do LLM:
1. Regex para extração de citações legais
2. Dicionário SQLite local para verificação de existência e vigência
3. SLM para validação semântica (opcional)

Uso rápido:
    >>> from ana.validacao import validar_resposta
    >>> resultados = validar_resposta("Conforme o Art. 7º da LGPD...")
    >>> resultados[0]["status"]
    'EXISTE_E_VIGENTE'
"""

from ana.validacao.pipeline import validar_resposta
from ana.validacao.extracao import extrair_citacoes
from ana.validacao.dicionario import DicionarioLeis

__all__ = [
    "validar_resposta",
    "extrair_citacoes",
    "DicionarioLeis",
]
