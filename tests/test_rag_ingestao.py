"""Testes do pipeline de ingestão RAG do sistema ANA.

Cobre chunking jurídico, extração de hierarquia e processamento
de documentos — sem dependência de GPU ou Qdrant.
"""

import pytest
from ana.rag.ingestao import (
    chunkar_texto_juridico,
    processar_documento,
    _extrair_hierarquia,
    _limpar_corpo_artigo,
)
from ana.rag.modelos import (
    AreaJuridica,
    ChunkJuridico,
    MetadataChunkJuridico,
    TipoDocumento,
    VigenciaStatus,
)

# Texto jurídico de exemplo para todos os testes
TEXTO_LGPD = """
TÍTULO I
DAS DISPOSIÇÕES GERAIS

CAPÍTULO I
DISPOSIÇÕES PRELIMINARES

Art. 1º Esta Lei dispõe sobre o tratamento de dados pessoais.
Parágrafo único. As normas gerais contidas nesta Lei são de interesse nacional.

Art. 2º A disciplina da proteção de dados pessoais tem como fundamentos:
I - o respeito à privacidade;
II - a autodeterminação informativa;
III - a liberdade de expressão.

TÍTULO II
DO TRATAMENTO DE DADOS PESSOAIS

CAPÍTULO I
DOS REQUISITOS

Art. 7º O tratamento de dados pessoais somente poderá ser realizado:
I - mediante o fornecimento de consentimento pelo titular;
II - para o cumprimento de obrigação legal.

Art. 8º O consentimento deverá ser fornecido por escrito ou por outro meio
que demonstre a manifestação de vontade do titular.
§ 1º Cabe ao controlador o ônus da prova.

CAPÍTULO II
DO TRATAMENTO DE DADOS SENSÍVEIS

Art. 11. O tratamento de dados pessoais sensíveis somente poderá ocorrer
com consentimento específico do titular.
""".strip()

METADATA_BASE = MetadataChunkJuridico(
    fonte="Lei 13.709/2018 (LGPD)",
    tipo=TipoDocumento.LEI_FEDERAL,
    area=AreaJuridica.DADOS,
    vigencia=VigenciaStatus.ATIVA,
    orgao="Congresso Nacional",
)


class TestChunkingJuridico:
    """Testes do chunker por artigo de lei."""

    def test_retorna_cinco_chunks(self):
        """Verifica que o texto da LGPD produz 5 chunks (artigos 1, 2, 7, 8, 11)."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        assert len(chunks) == 5

    def test_artigos_corretos(self):
        """Verifica os identificadores de artigo encontrados."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        artigos = [c.metadata.artigo for c in chunks]
        assert any("1º" in a for a in artigos)
        assert any("2º" in a for a in artigos)
        assert any("7º" in a for a in artigos)
        assert any("8º" in a for a in artigos)
        assert any("11" in a for a in artigos)

    def test_ignora_referencias_internas(self):
        """Verifica que referências como 'do art. 7º' não viram chunks."""
        texto_com_ref = TEXTO_LGPD + "\nArt. 12. Aplica-se o disposto no art. 7º desta Lei."
        chunks = chunkar_texto_juridico(texto_com_ref, METADATA_BASE)
        artigos = [c.metadata.artigo for c in chunks]
        # Deve ter apenas art. 12 adicional, não duplicar art. 7
        ocorrencias_7 = [a for a in artigos if "7" in a]
        assert len(ocorrencias_7) == 1, f"Art. 7 duplicado: {artigos}"

    def test_fronteiras_entre_artigos(self):
        """Verifica que Art. 11 não aparece no texto do Art. 8."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        art8 = next(c for c in chunks if "8º" in (c.metadata.artigo or ""))
        assert "Art. 11" not in art8.texto
        # O corpo de Art. 11 não deve aparecer no Art. 8
        assert "somente poderá ocorrer" not in art8.texto

    def test_incisos_no_mesmo_chunk(self):
        """Verifica que incisos do Art. 2 estão no mesmo chunk que o cabeçalho."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        art2 = next(c for c in chunks if "2º" in (c.metadata.artigo or ""))
        assert "autodeterminação" in art2.texto.lower()
        assert "liberdade de expressão" in art2.texto.lower()

    def test_hierarquia_titulo_extraida(self):
        """Verifica que o Título é extraído corretamente para Art. 7."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        art7 = next(c for c in chunks if "7º" in (c.metadata.artigo or ""))
        assert art7.metadata.titulo is not None
        assert "TÍTULO II" in art7.metadata.titulo

    def test_hierarquia_capitulo_extraido(self):
        """Verifica que o Capítulo é extraído corretamente para Art. 7."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        art7 = next(c for c in chunks if "7º" in (c.metadata.artigo or ""))
        assert art7.metadata.capitulo is not None
        assert "CAPÍTULO I" in art7.metadata.capitulo

    def test_art11_capitulo_ii(self):
        """Verifica que Art. 11 está associado ao Capítulo II."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        art11 = next(c for c in chunks if "11" in (c.metadata.artigo or ""))
        assert art11.metadata.capitulo is not None
        assert "CAPÍTULO II" in art11.metadata.capitulo

    def test_marcadores_estruturais_removidos_do_texto(self):
        """Verifica que TÍTULO/CAPÍTULO não aparecem no texto do chunk."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        for chunk in chunks:
            assert "TÍTULO" not in chunk.texto, (
                f"TÍTULO encontrado no texto do chunk {chunk.metadata.artigo}"
            )

    def test_metadata_herdada_da_base(self):
        """Verifica que cada chunk herda fonte, tipo, área e vigência da base."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        for chunk in chunks:
            assert chunk.metadata.fonte == "Lei 13.709/2018 (LGPD)"
            assert chunk.metadata.tipo == TipoDocumento.LEI_FEDERAL
            assert chunk.metadata.area == AreaJuridica.DADOS
            assert chunk.metadata.vigencia == VigenciaStatus.ATIVA

    def test_texto_vazio_retorna_lista_vazia(self):
        """Verifica que texto sem artigos retorna lista vazia."""
        chunks = chunkar_texto_juridico("Texto sem artigos aqui.", METADATA_BASE)
        assert chunks == []

    def test_chunks_sao_instancias_chunk_juridico(self):
        """Verifica que os chunks são instâncias corretas de ChunkJuridico."""
        chunks = chunkar_texto_juridico(TEXTO_LGPD, METADATA_BASE)
        for chunk in chunks:
            assert isinstance(chunk, ChunkJuridico)
            assert chunk.texto != ""
            assert chunk.embedding is None  # Ainda não gerado


class TestProcessarDocumento:
    """Testes da função de conveniência processar_documento."""

    def test_processa_com_parametros_completos(self):
        """Verifica processamento com todos os parâmetros fornecidos."""
        chunks = processar_documento(
            texto=TEXTO_LGPD,
            fonte="Lei 13.709/2018 (LGPD)",
            tipo=TipoDocumento.LEI_FEDERAL,
            area=AreaJuridica.DADOS,
            vigencia=VigenciaStatus.ATIVA,
            orgao="Congresso Nacional",
            url_origem="https://planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
        )
        assert len(chunks) == 5
        assert chunks[0].metadata.orgao == "Congresso Nacional"
        assert chunks[0].metadata.url_origem is not None

    def test_processa_sem_area(self):
        """Verifica processamento sem área jurídica definida."""
        chunks = processar_documento(
            texto=TEXTO_LGPD,
            fonte="Lei Teste",
            tipo=TipoDocumento.LEI_FEDERAL,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.area is None

    def test_sessao_id_propagada(self):
        """Verifica que sessao_id é propagada para todos os chunks."""
        chunks = processar_documento(
            texto=TEXTO_LGPD,
            fonte="Documento do Processo",
            tipo=TipoDocumento.DOCUMENTO_USUARIO,
            sessao_id="sessao-teste-001",
        )
        for chunk in chunks:
            assert chunk.metadata.sessao_id == "sessao-teste-001"


class TestExtrairHierarquia:
    """Testes da extração de hierarquia do texto anterior ao artigo."""

    def test_extrai_titulo(self):
        """Verifica extração de TÍTULO do bloco anterior."""
        bloco = "TÍTULO II\nDO TRATAMENTO DE DADOS PESSOAIS\n\n"
        titulo, _, _ = _extrair_hierarquia(bloco)
        assert titulo is not None
        assert "TÍTULO II" in titulo

    def test_extrai_capitulo(self):
        """Verifica extração de CAPÍTULO do bloco anterior."""
        bloco = "CAPÍTULO I\nDOS REQUISITOS\n\n"
        _, capitulo, _ = _extrair_hierarquia(bloco)
        assert capitulo is not None
        assert "CAPÍTULO I" in capitulo

    def test_retorna_none_sem_estrutura(self):
        """Verifica retorno None quando não há estrutura hierárquica."""
        titulo, capitulo, secao = _extrair_hierarquia("Texto simples sem hierarquia")
        assert titulo is None
        assert capitulo is None
        assert secao is None

    def test_extrai_ultimo_titulo(self):
        """Verifica que extrai o TÍTULO mais recente (último) do bloco."""
        bloco = "TÍTULO I\nDAS DISPOSIÇÕES\n\nTÍTULO II\nDO TRATAMENTO\n\n"
        titulo, _, _ = _extrair_hierarquia(bloco)
        assert titulo is not None
        assert "TÍTULO II" in titulo


class TestLimparCorpoArtigo:
    """Testes da limpeza de marcadores estruturais do corpo do artigo."""

    def test_remove_titulo(self):
        """Verifica que marcador TÍTULO é removido do corpo do chunk."""
        texto = "Art. 1º Texto aqui.\n\nTÍTULO II\nDO TRATAMENTO\n"
        limpo = _limpar_corpo_artigo(texto)
        assert "TÍTULO II" not in limpo
        assert "Art. 1º" in limpo

    def test_remove_capitulo(self):
        """Verifica que marcador CAPÍTULO é removido do corpo do chunk."""
        texto = "Art. 5º Texto.\n\nCAPÍTULO II\nDOS DADOS SENSÍVEIS\n"
        limpo = _limpar_corpo_artigo(texto)
        assert "CAPÍTULO II" not in limpo

    def test_preserva_conteudo_do_artigo(self):
        """Verifica que o conteúdo legal do artigo é preservado."""
        texto = "Art. 7º O tratamento somente poderá ser realizado:\nI - com consentimento."
        limpo = _limpar_corpo_artigo(texto)
        assert "consentimento" in limpo
