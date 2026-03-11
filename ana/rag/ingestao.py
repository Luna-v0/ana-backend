"""Pipeline de ingestão de documentos jurídicos para o sistema RAG.

Implementa as Etapas 1 e 2 do spec 02:
- Etapa 1: Extração de texto de PDFs e páginas web
- Etapa 2: Chunking inteligente respeitando a hierarquia legal

Hierarquia de chunking:
    Lei → Título → Capítulo → Seção → Artigo (unidade base do chunk)

Nota (LGPD):
    Documentos de processos são tratados localmente. Nunca enviados
    para serviços externos.
"""

import re
from pathlib import Path
from typing import Optional

from loguru import logger

from ana.rag.modelos import (
    AreaJuridica,
    ChunkJuridico,
    MetadataChunkJuridico,
    TipoDocumento,
    VigenciaStatus,
)


# =============================================================================
# Padrões Regex para estrutura legal brasileira
# =============================================================================

# Artigo no INÍCIO DE LINHA com letra maiúscula (ignora referências internas)
_RE_ARTIGO = re.compile(
    r"^(Art\.\s*\d+[º°]?(?:\-[A-Z])?\.?)",
    re.MULTILINE,
)

# Título na hierarquia — suporta três formatos:
#   Planalto (duas linhas): "TÍTULO I\nDas Garantias..."
#   Vade Mecum (mesma linha): "TÍTULO II – Das Garantias Fundamentais"
#   Vade Mecum (linha quebrada sem hífen): "TÍTULO IX – DO\nHABEAS CORPUS"
#   A segunda linha é capturada apenas se não for um novo marcador estrutural.
_RE_TITULO = re.compile(
    r"^(TÍTULO\s+[IVXLCDM]+[^\n]*(?:\n(?!(?:TÍTULO|CAPÍTULO|Seção|Art\.)\s)[^\n]+)?)",
    re.MULTILINE,
)

# Capítulo na hierarquia
_RE_CAPITULO = re.compile(
    r"^(CAPÍTULO\s+[IVXLCDM]+[^\n]*(?:\n(?!(?:TÍTULO|CAPÍTULO|Seção|Art\.)\s)[^\n]+)?)",
    re.MULTILINE,
)

# Seção na hierarquia
_RE_SECAO = re.compile(
    r"^(Seção\s+[IVXLCDM]+[^\n]*(?:\n(?!(?:TÍTULO|CAPÍTULO|Seção|Art\.)\s)[^\n]+)?)",
    re.MULTILINE,
)

# Remove marcadores estruturais do corpo do chunk (ficam apenas na metadata)
_RE_ESTRUTURAL = re.compile(
    r"^(TÍTULO|CAPÍTULO|Seção)\s+[IVXLCDM]+.*$\n?(?:^.+$\n?)?",
    re.MULTILINE | re.IGNORECASE,
)

# Remove tags HTML que aparecem em PDFs gerados de HTML (ex: Vade Mecum)
_RE_HTML = re.compile(r"<[^>]+>")

# Quebra de linha com hífen (palavra parti-\nda → partida)
_RE_HIFEN_QUEBRA = re.compile(r"-\n")

# Quebra de linha comum → espaço
_RE_QUEBRA_LINHA = re.compile(r"\n+")

# Extrai número do artigo para uso como ID
_RE_NUM_ARTIGO = re.compile(r"\d+")

# Janela de lookback para extração de hierarquia.
# Evita contaminação cruzada entre leis num mesmo PDF (ex: Vade Mecum).
# 12000 chars ≈ ~10–20 artigos médios — suficiente para capturar cabeçalhos
# de capítulos longos sem carregar texto de leis inteiras anteriores.
_JANELA_HIERARQUIA = 12000


# =============================================================================
# Funções auxiliares
# =============================================================================

def _extrair_hierarquia(
    bloco: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extrai título, capítulo e seção mais recentes de um bloco de texto.

    Limita a busca aos últimos _JANELA_HIERARQUIA caracteres do bloco para
    evitar contaminação cruzada quando o texto contém múltiplas leis
    concatenadas (ex: Vade Mecum em PDF único).

    Args:
        bloco: Texto que precede o artigo atual.

    Returns:
        Tupla (titulo, capitulo, secao) mais recentes encontrados, ou None.
    """
    # Usa apenas a janela imediatamente antes do artigo
    janela = bloco[-_JANELA_HIERARQUIA:] if len(bloco) > _JANELA_HIERARQUIA else bloco

    titulo = None
    capitulo = None
    secao = None

    titulos = _RE_TITULO.findall(janela)
    if titulos:
        titulo = " ".join(titulos[-1].split())

    capitulos = _RE_CAPITULO.findall(janela)
    if capitulos:
        capitulo = " ".join(capitulos[-1].split())

    secoes = _RE_SECAO.findall(janela)
    if secoes:
        secao = " ".join(secoes[-1].split())

    return titulo, capitulo, secao


def _limpar_corpo_artigo(texto: str) -> str:
    """Remove marcadores estruturais e normaliza o texto do artigo.

    Marcadores de Título, Capítulo e Seção ficam na metadata do chunk,
    não no texto de busca para não enviesar BM25 e embeddings.

    Normalização adicional para PDFs gerados de HTML (ex: Vade Mecum):
      - Remove tags HTML (<p>, <br>, etc.) que ficam no texto do PDF
      - Concatena palavras hifenizadas quebradas por newline (parti-\nda → partida)
      - Transforma demais quebras de linha em espaço simples (texto corrido)

    Args:
        texto: Texto bruto do artigo (pode incluir marcadores estruturais
            de seções que aparecem antes do próximo artigo).

    Returns:
        Texto do artigo sem marcadores estruturais, em texto corrido, stripped.
    """
    texto = _RE_ESTRUTURAL.sub("", texto)
    texto = _RE_HTML.sub(" ", texto)
    texto = _RE_HIFEN_QUEBRA.sub("", texto)
    texto = _RE_QUEBRA_LINHA.sub(" ", texto)
    # Colapsa espaços múltiplos gerados pelas substituições acima
    texto = re.sub(r" {2,}", " ", texto)
    return texto.strip()


def _normalizar_artigo_id(marcador: str) -> str:
    """Normaliza o identificador do artigo.

    Args:
        marcador: Texto do marcador do artigo (ex: 'Art. 1º  ').

    Returns:
        String normalizada (ex: 'Art. 1º').
    """
    return re.sub(r"\s+", " ", marcador).strip().rstrip(".")


# =============================================================================
# Chunking jurídico
# =============================================================================

def chunkar_texto_juridico(
    texto: str,
    metadata_base: MetadataChunkJuridico,
) -> list[ChunkJuridico]:
    """Divide texto de lei em chunks por artigo com metadata jurídica.

    Cada artigo de lei torna-se um chunk independente. Artigos longos
    com múltiplos parágrafos e incisos ficam inteiros no chunk (geralmente
    cabem em 512 tokens). Marcadores estruturais (TÍTULO, CAPÍTULO, Seção)
    são extraídos para a metadata, não para o texto de busca.

    Args:
        texto: Texto completo da lei para chunking.
        metadata_base: Metadata base com informações da fonte (lei, tipo,
            área, vigência, etc.). Cada chunk herda estes valores e adiciona
            a hierarquia específica do artigo.

    Returns:
        Lista de ChunkJuridico, um por artigo principal encontrado.
    """
    chunks: list[ChunkJuridico] = []
    posicoes = [(m.start(), m.group(0)) for m in _RE_ARTIGO.finditer(texto)]

    if not posicoes:
        logger.warning(
            f"Nenhum artigo encontrado no texto para '{metadata_base.fonte}'. "
            "Verifique o formato do documento."
        )
        return chunks

    for i, (inicio, marcador) in enumerate(posicoes):
        fim = posicoes[i + 1][0] if i + 1 < len(posicoes) else len(texto)
        texto_bruto = texto[inicio:fim].strip()
        texto_limpo = _limpar_corpo_artigo(texto_bruto)

        bloco_anterior = texto[:inicio]
        titulo, capitulo, secao = _extrair_hierarquia(bloco_anterior)
        artigo_id = _normalizar_artigo_id(marcador)

        # Chunk herda metadata_base e adiciona hierarquia específica
        metadata_chunk = MetadataChunkJuridico(
            fonte=metadata_base.fonte,
            tipo=metadata_base.tipo,
            area=metadata_base.area,
            vigencia=metadata_base.vigencia,
            orgao=metadata_base.orgao,
            url_origem=metadata_base.url_origem,
            data_publicacao=metadata_base.data_publicacao,
            sessao_id=metadata_base.sessao_id,
            titulo=titulo,
            capitulo=capitulo,
            secao=secao,
            artigo=artigo_id,
        )

        chunks.append(ChunkJuridico(texto=texto_limpo, metadata=metadata_chunk))

    logger.info(
        f"Chunking concluído: {len(chunks)} artigos extraídos de '{metadata_base.fonte}'"
    )
    return chunks


# =============================================================================
# Extração de texto
# =============================================================================

def extrair_texto_pdf(caminho_pdf: Path) -> str:
    """Extrai texto de um arquivo PDF usando PyMuPDF.

    Mantém estrutura de parágrafos e respeita quebras de página.
    Para PDFs jurídicos com colunas ou tabelas, Docling é preferível.

    Args:
        caminho_pdf: Caminho absoluto para o arquivo PDF.

    Returns:
        Texto extraído como string.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ImportError: Se PyMuPDF (pymupdf) não estiver instalado.
    """
    if not caminho_pdf.exists():
        raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")

    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF não instalado. Execute: uv add pymupdf"
        ) from exc

    texto_paginas: list[str] = []
    with fitz.open(str(caminho_pdf)) as doc:
        for pagina in doc:
            texto_paginas.append(pagina.get_text())

    texto = "\n".join(texto_paginas)
    logger.info(f"PDF extraído: {caminho_pdf.name} ({len(texto)} chars)")
    return texto


def extrair_texto_docling(caminho: Path) -> str:
    """Extrai texto estruturado usando Docling (IBM).

    Preferível para PDFs complexos com colunas, tabelas e imagens.
    Mantém hierarquia de seções e artigos.

    Args:
        caminho: Caminho para PDF ou documento suportado pelo Docling.

    Returns:
        Texto extraído com estrutura preservada.

    Raises:
        ImportError: Se docling não estiver instalado.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Docling não instalado. Execute: uv add docling"
        ) from exc

    conversor = DocumentConverter()
    resultado = conversor.convert(str(caminho))
    texto = resultado.document.export_to_markdown()
    logger.info(f"Docling extraído: {caminho.name} ({len(texto)} chars)")
    return texto


def processar_documento(
    texto: str,
    fonte: str,
    tipo: TipoDocumento,
    area: Optional[AreaJuridica] = None,
    vigencia: VigenciaStatus = VigenciaStatus.ATIVA,
    orgao: Optional[str] = None,
    url_origem: Optional[str] = None,
    sessao_id: Optional[str] = None,
) -> list[ChunkJuridico]:
    """Processa texto de um documento jurídico completo em chunks.

    Função de conveniência que cria a metadata base e chama o chunker.

    Args:
        texto: Texto completo do documento.
        fonte: Identificação da fonte (ex: 'Lei 13.709/2018 (LGPD)').
        tipo: Tipo do documento jurídico.
        area: Área do direito (opcional).
        vigencia: Status de vigência (padrão: ATIVA).
        orgao: Órgão emissor (opcional).
        url_origem: URL da fonte original (opcional).
        sessao_id: ID de sessão de processo (apenas para docs de usuário).

    Returns:
        Lista de ChunkJuridico prontos para indexação.
    """
    metadata_base = MetadataChunkJuridico(
        fonte=fonte,
        tipo=tipo,
        area=area,
        vigencia=vigencia,
        orgao=orgao,
        url_origem=url_origem,
        sessao_id=sessao_id,
    )
    return chunkar_texto_juridico(texto, metadata_base)
