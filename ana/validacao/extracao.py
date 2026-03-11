"""Extração de citações legais via regex.

Implementa Spec 07 seção 6.3: padrões para leis, artigos, súmulas,
códigos e referências à Constituição Federal.
"""

from __future__ import annotations

import re
from typing import Any


# =============================================================================
# Padrões regex
# =============================================================================

_PATTERNS: list[tuple[str, str]] = [
    # Lei n. 12.345/2011, Lei 13.709/18, etc.
    (
        r"Lei\s+(?:n[ºo°.]?\s*)?(\d{1,6}[./]\d{2,4})",
        "lei",
    ),
    # Decreto n. 9.580/2018, Decreto-Lei 5.452/43, etc.
    (
        r"Decreto(?:-Lei)?\s+(?:n[ºo°.]?\s*)?(\d{1,6}[./]\d{2,4})",
        "decreto",
    ),
    # Art. 5º, Art. 7°, Artigo 12, etc.
    (
        r"[Aa]rt(?:igo)?\.?\s*(\d{1,4}[º°]?(?:\-[A-Z])?)",
        "artigo",
    ),
    # Súmula n. 123 do STF / STJ / TST
    (
        r"[Ss][úu]mula\s+(?:n[ºo°.]?\s*)?(\d{1,4})\s+(?:do\s+)?(STF|STJ|TST|TSE|STM)",
        "sumula",
    ),
    # Súmula Vinculante n. 11
    (
        r"[Ss][úu]mula\s+[Vv]inculante\s+(?:n[ºo°.]?\s*)?(\d{1,3})",
        "sumula_vinculante",
    ),
    # Códigos nominados
    (
        r"(C[óo]digo\s+(?:Civil|Penal|Tributário|Nacional|Comercial|"
        r"de\s+Defesa\s+do\s+Consumidor|de\s+Processo\s+Civil|"
        r"de\s+Processo\s+Penal|Eleitoral|Florestal|"
        r"de\s+Trânsito\s+Brasileiro))",
        "codigo",
    ),
    # Constituição Federal / CF/88
    (
        r"(?:CF|CF/88|Constitui[çc][ãa]o\s+Federal)(?:\s*(?:de\s+)?1988)?",
        "constituicao",
    ),
    # CLT
    (
        r"\bCLT\b",
        "clt",
    ),
    # ECA
    (
        r"\bECA\b",
        "eca",
    ),
    # LGPD
    (
        r"\bLGPD\b",
        "lgpd",
    ),
]

# Pré-compilados para performance
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), tipo)
    for pat, tipo in _PATTERNS
]


def extrair_citacoes(texto: str) -> list[dict[str, Any]]:
    """Extrai todas as citações legais de um texto.

    Args:
        texto: Texto a analisar (resposta de LLM, peça jurídica, etc.).

    Returns:
        Lista de dicionários com campos:
        - ``texto_original``: Trecho exato encontrado no texto.
        - ``posicao``: Índice de início no texto original.
        - ``tipo``: Tipo da citação (lei, artigo, sumula, etc.).
        - ``lei``: Número da lei (quando extraível).
        - ``artigo``: Número do artigo (quando presente).
    """
    citacoes: list[dict[str, Any]] = []
    visto: set[int] = set()  # Evita duplicatas por posição

    for padrao, tipo in _COMPILED:
        for m in padrao.finditer(texto):
            inicio = m.start()
            if inicio in visto:
                continue
            visto.add(inicio)

            citacao: dict[str, Any] = {
                "texto_original": m.group(0),
                "posicao": inicio,
                "tipo": tipo,
                "lei": None,
                "artigo": None,
            }

            # Extrai número da lei para os tipos que têm grupo capturado
            if tipo in ("lei", "decreto") and m.lastindex and m.lastindex >= 1:
                citacao["lei"] = m.group(1)
            elif tipo == "artigo" and m.lastindex and m.lastindex >= 1:
                citacao["artigo"] = m.group(1)
            elif tipo in ("sumula", "sumula_vinculante") and m.lastindex and m.lastindex >= 1:
                citacao["lei"] = m.group(1)
            elif tipo == "codigo" and m.lastindex and m.lastindex >= 1:
                citacao["lei"] = m.group(1)
            elif tipo == "constituicao":
                citacao["lei"] = "CF/1988"
            elif tipo == "clt":
                citacao["lei"] = "CLT (Decreto-Lei 5.452/1943)"
            elif tipo == "eca":
                citacao["lei"] = "ECA (Lei 8.069/1990)"
            elif tipo == "lgpd":
                citacao["lei"] = "LGPD (Lei 13.709/2018)"

            citacoes.append(citacao)

    # Ordena por posição no texto
    citacoes.sort(key=lambda c: c["posicao"])
    return citacoes
