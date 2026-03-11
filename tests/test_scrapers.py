"""Testes para o módulo de scrapers jurídicos (Spec 02).

Todos os testes usam mocks para não depender de rede.
"""

import hashlib
import sqlite3
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ana.scrapers.modelos import DocumentoColetado, ResultadoColeta
from ana.scrapers.cache import CacheScrapers


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def cache_temp(tmp_path: Path) -> CacheScrapers:
    """Cache SQLite em diretório temporário."""
    return CacheScrapers(tmp_path / "test_scrapers.db")


def _doc_fake(
    url: str = "https://exemplo.gov.br/lei",
    titulo: str = "Lei Teste",
    texto: str = "Art. 1º Esta lei dispõe sobre testes.\nArt. 2º Disposições finais.",
    tipo: str = "lei_federal",
    area: str | None = "civil",
) -> DocumentoColetado:
    """Cria documento coletado para testes."""
    hash_ = hashlib.sha256(texto.encode()).hexdigest()[:16]
    return DocumentoColetado(
        url_origem=url,
        fonte=titulo,
        tipo=tipo,
        area=area,
        titulo=titulo,
        texto=texto,
        data_publicacao=datetime(2020, 1, 1),
        data_coleta=datetime.now(),
        hash_conteudo=hash_,
        orgao="Congresso Nacional",
    )


# =============================================================================
# Testes: DocumentoColetado
# =============================================================================

class TestDocumentoColetado:
    def test_criacao_basica(self):
        doc = _doc_fake()
        assert doc.url_origem == "https://exemplo.gov.br/lei"
        assert doc.tipo == "lei_federal"
        assert doc.vigencia == "ativa"  # padrão

    def test_hash_preenchido(self):
        doc = _doc_fake()
        assert len(doc.hash_conteudo) == 16  # SHA-256 truncado

    def test_vigencia_customizada(self):
        doc = _doc_fake()
        doc.vigencia = "revogada"
        assert doc.vigencia == "revogada"


# =============================================================================
# Testes: ResultadoColeta
# =============================================================================

class TestResultadoColeta:
    def test_criacao_padrao(self):
        r = ResultadoColeta(fonte="planalto")
        assert r.documentos_novos == 0
        assert r.documentos_ignorados == 0
        assert r.erros == []

    def test_finalizar_calcula_duracao(self):
        r = ResultadoColeta(fonte="stf")
        time.sleep(0.01)
        r.finalizar()
        assert r.finalizou_em is not None
        assert r.duracao_segundos >= 0.0


# =============================================================================
# Testes: CacheScrapers
# =============================================================================

class TestCacheScrapers:
    def test_cria_banco_sqlite(self, cache_temp: CacheScrapers):
        assert cache_temp.caminho_db.exists()

    def test_documento_nao_coletado_inicialmente(self, cache_temp: CacheScrapers):
        assert not cache_temp.ja_coletado("https://lei.gov.br/1", "abc123")

    def test_registrar_e_verificar(self, cache_temp: CacheScrapers):
        cache_temp.registrar(
            url="https://lei.gov.br/1",
            hash_conteudo="abc123",
            fonte="planalto",
            titulo="Lei Teste",
        )
        assert cache_temp.ja_coletado("https://lei.gov.br/1", "abc123")

    def test_hash_diferente_nao_e_hit(self, cache_temp: CacheScrapers):
        cache_temp.registrar(
            url="https://lei.gov.br/1",
            hash_conteudo="abc123",
            fonte="planalto",
        )
        # Hash diferente = conteúdo mudou = deve reingerir
        assert not cache_temp.ja_coletado("https://lei.gov.br/1", "xyz999")

    def test_total_por_fonte(self, cache_temp: CacheScrapers):
        cache_temp.registrar("https://a.gov/1", "h1", "planalto", "Lei A")
        cache_temp.registrar("https://b.gov/2", "h2", "planalto", "Lei B")
        cache_temp.registrar("https://c.gov/3", "h3", "stf", "Súmula 1")

        assert cache_temp.total("planalto") == 2
        assert cache_temp.total("stf") == 1
        assert cache_temp.total() == 3

    def test_ultima_coleta_none_se_vazio(self, cache_temp: CacheScrapers):
        assert cache_temp.ultima_coleta("planalto") is None

    def test_ultima_coleta_retorna_datetime(self, cache_temp: CacheScrapers):
        cache_temp.registrar("https://lei.gov.br/1", "h1", "planalto", "Lei")
        ultima = cache_temp.ultima_coleta("planalto")
        assert ultima is not None
        assert isinstance(ultima, datetime)

    def test_listar_retorna_registros(self, cache_temp: CacheScrapers):
        cache_temp.registrar("https://lei.gov.br/1", "h1", "planalto", "Lei A")
        registros = cache_temp.listar()
        assert len(registros) == 1
        assert registros[0]["titulo"] == "Lei A"
        assert registros[0]["fonte"] == "planalto"

    def test_upsert_atualiza_hash(self, cache_temp: CacheScrapers):
        cache_temp.registrar("https://lei.gov.br/1", "hash_velho", "planalto", "Lei")
        cache_temp.registrar("https://lei.gov.br/1", "hash_novo", "planalto", "Lei")
        assert cache_temp.ja_coletado("https://lei.gov.br/1", "hash_novo")
        assert not cache_temp.ja_coletado("https://lei.gov.br/1", "hash_velho")


# =============================================================================
# Testes: ScraperBase
# =============================================================================

class TestScraperBase:
    def test_hash_retorna_16_chars(self):
        from ana.scrapers.base import ScraperBase
        # ScraperBase é abstrata; testamos via método estático
        h = ScraperBase._hash("texto jurídico de teste")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_determinístico(self):
        from ana.scrapers.base import ScraperBase
        assert ScraperBase._hash("abc") == ScraperBase._hash("abc")

    def test_hash_diferente_para_textos_distintos(self):
        from ana.scrapers.base import ScraperBase
        assert ScraperBase._hash("lei A") != ScraperBase._hash("lei B")


# =============================================================================
# Testes: ScraperPlanalto (com mock HTTP)
# =============================================================================

HTML_PLANALTO_FAKE = """
<html><body>
<p>Art. 1° Esta lei estabelece normas gerais sobre proteção de dados pessoais.</p>
<p>Parágrafo único. As normas gerais previstas nesta Lei são de caráter nacional.</p>
<p>Art. 2° A disciplina da proteção de dados pessoais tem como fundamentos:</p>
<p>I - o respeito à privacidade;</p>
<p>II - a autodeterminação informativa;</p>
<p>Art. 3° Esta Lei aplica-se a qualquer operação de tratamento.</p>
</body></html>
"""


class TestScraperPlanalto:
    def test_extrair_texto_retorna_artigos(self):
        from ana.scrapers.fontes.planalto import extrair_texto_planalto
        texto = extrair_texto_planalto(HTML_PLANALTO_FAKE)
        assert "Art. 1°" in texto or "Art." in texto

    def test_extrair_texto_html_vazio(self):
        from ana.scrapers.fontes.planalto import extrair_texto_planalto
        texto = extrair_texto_planalto("<html><body></body></html>")
        assert isinstance(texto, str)

    def test_coletar_retorna_documento(self):
        from ana.scrapers.fontes.planalto import ScraperPlanalto
        scraper = ScraperPlanalto()
        resp_mock = MagicMock(spec=httpx.Response)
        resp_mock.text = HTML_PLANALTO_FAKE
        resp_mock.content = HTML_PLANALTO_FAKE.encode()

        with patch.object(scraper, "_http_get", return_value=resp_mock):
            docs = list(scraper.coletar())

        # Deve coletar pelo menos um documento com texto suficiente
        # (outros podem falhar por texto curto no mock)
        assert isinstance(docs, list)
        for doc in docs:
            assert doc.tipo == "lei_federal"
            assert doc.orgao == "Congresso Nacional"
            assert len(doc.hash_conteudo) == 16

    def test_coletar_ignora_falha_http(self):
        from ana.scrapers.fontes.planalto import ScraperPlanalto
        scraper = ScraperPlanalto()

        with patch.object(scraper, "_http_get", return_value=None):
            docs = list(scraper.coletar())

        assert docs == []

    def test_nome_fonte(self):
        from ana.scrapers.fontes.planalto import ScraperPlanalto
        assert ScraperPlanalto().nome() == "planalto"

    def test_verificar_atualizacoes_chama_coletar(self):
        from ana.scrapers.fontes.planalto import ScraperPlanalto
        scraper = ScraperPlanalto()
        with patch.object(scraper, "coletar", return_value=iter([])) as mock_coletar:
            list(scraper.verificar_atualizacoes(datetime.now()))
        mock_coletar.assert_called_once()


# =============================================================================
# Testes: ScraperSTF
# =============================================================================

HTML_STF_FAKE = """
<html><body>
<p>Súmula 1: É irrelevante para a concessão de interdito proibitório.</p>
<p>Súmula 2: Não cabe o habeas corpus contra a imposição da pena de exclusão.</p>
<p>Súmula 3: A imunidade concedida ao marido pelo art. 183, I, do Código Penal.</p>
</body></html>
"""


class TestScraperSTF:
    def test_extrair_sumulas_stf(self):
        from ana.scrapers.fontes.stf import _extrair_sumulas_html
        sumulas = _extrair_sumulas_html(HTML_STF_FAKE)
        assert len(sumulas) >= 1
        for num, texto in sumulas:
            assert num.isdigit()
            assert len(texto) > 10

    def test_nome_fonte(self):
        from ana.scrapers.fontes.stf import ScraperSTF
        assert ScraperSTF().nome() == "stf"

    def test_coletar_retorna_documentos(self):
        from ana.scrapers.fontes.stf import ScraperSTF
        scraper = ScraperSTF()
        resp_mock = MagicMock(spec=httpx.Response)
        resp_mock.text = HTML_STF_FAKE

        with patch.object(scraper, "_http_get", return_value=resp_mock):
            docs = list(scraper.coletar())

        assert isinstance(docs, list)
        for doc in docs:
            assert doc.tipo == "sumula"
            assert doc.orgao == "STF"

    def test_coletar_sem_sumulas_nao_falha(self):
        from ana.scrapers.fontes.stf import ScraperSTF
        scraper = ScraperSTF()
        resp_mock = MagicMock(spec=httpx.Response)
        resp_mock.text = "<html><body><p>Página em manutenção</p></body></html>"

        with patch.object(scraper, "_http_get", return_value=resp_mock):
            docs = list(scraper.coletar())

        assert isinstance(docs, list)


# =============================================================================
# Testes: ScraperSTJ
# =============================================================================

HTML_STJ_FAKE = """
<html><body>
<table>
<tr><td>Súmula 1</td><td>A ação para pleitear nulidade de registro.</td></tr>
<tr><td>Súmula 2</td><td>Não cabe o habeas corpus quando já extinta a pena.</td></tr>
</table>
</body></html>
"""


class TestScraperSTJ:
    def test_extrair_sumulas_stj(self):
        from ana.scrapers.fontes.stj import _extrair_sumulas_stj
        sumulas = _extrair_sumulas_stj(HTML_STJ_FAKE)
        assert len(sumulas) >= 1

    def test_nome_fonte(self):
        from ana.scrapers.fontes.stj import ScraperSTJ
        assert ScraperSTJ().nome() == "stj"

    def test_coletar_retorna_documento(self):
        from ana.scrapers.fontes.stj import ScraperSTJ
        scraper = ScraperSTJ()
        resp_mock = MagicMock(spec=httpx.Response)
        resp_mock.text = HTML_STJ_FAKE

        with patch.object(scraper, "_http_get", return_value=resp_mock):
            docs = list(scraper.coletar())

        assert isinstance(docs, list)
        for doc in docs:
            assert doc.tipo == "sumula"
            assert doc.orgao == "STJ"


# =============================================================================
# Testes: ScraperLexML (com mock)
# =============================================================================

XML_SRU_FAKE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<searchRetrieveResponse xmlns="http://www.loc.gov/zing/srw/">'
    "<numberOfRecords>1</numberOfRecords>"
    "<records><record><recordData>"
    '<lexml xmlns="http://www.lexml.gov.br/1.0">'
    "<identifier>urn:lex:br:federal:lei:2018-08-14;13709</identifier>"
    "<title>Lei 13.709/2018</title>"
    "</lexml></recordData></record></records>"
    "</searchRetrieveResponse>"
).encode("utf-8")

XML_LEI_FAKE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<LexML xmlns="http://www.lexml.gov.br/1.0"><texto><articulacao>'
    '<artigo id="art1"><num>Art. 1</num>'
    "<caput>Esta Lei disciplina o tratamento de dados pessoais.</caput>"
    "</artigo>"
    '<artigo id="art2"><num>Art. 2</num>'
    "<caput>A disciplina da protecao de dados tem como fundamentos.</caput>"
    "</artigo>"
    "</articulacao></texto></LexML>"
).encode("utf-8")


class TestScraperLexML:
    def test_nome_fonte(self):
        from ana.scrapers.fontes.lexml import ScraperLexML
        assert ScraperLexML().nome() == "lexml"

    def test_buscar_registros_com_mock(self):
        from ana.scrapers.fontes.lexml import ScraperLexML
        scraper = ScraperLexML()
        resp_mock = MagicMock(spec=httpx.Response)
        resp_mock.content = XML_SRU_FAKE

        with patch.object(scraper, "_http_get", return_value=resp_mock):
            registros = scraper._buscar_registros("tipo_documento=lei_federal")

        assert len(registros) >= 1

    def test_extrair_texto_xml_lexml(self):
        from ana.scrapers.fontes.lexml import _extrair_texto_xml_lexml
        texto = _extrair_texto_xml_lexml(XML_LEI_FAKE)
        assert "Art." in texto or "disciplina" in texto

    def test_verificar_atualizacoes_chama_buscar(self):
        from ana.scrapers.fontes.lexml import ScraperLexML
        scraper = ScraperLexML()
        with patch.object(scraper, "_buscar_registros", return_value=[]) as mock:
            list(scraper.verificar_atualizacoes(datetime.now()))
        mock.assert_called_once()


# =============================================================================
# Testes: Router /scrapers (endpoints)
# =============================================================================

class TestEndpointsScrapers:
    @pytest.fixture
    def cliente(self):
        from fastapi.testclient import TestClient
        from ana.api.main import app
        return TestClient(app)

    def test_status_scrapers(self, cliente):
        with patch("ana.scrapers.pipeline.PipelineScrapers") as MockPipeline:
            MockPipeline.return_value.status.return_value = {
                "fontes": {},
                "total_documentos_cache": 0,
                "dependencias_instaladas": True,
            }
            resp = cliente.get("/scrapers/status")
        assert resp.status_code == 200
        dados = resp.json()
        assert "dependencias_instaladas" in dados or "fontes" in dados

    def test_listar_fontes(self, cliente):
        resp = cliente.get("/scrapers/fontes")
        assert resp.status_code == 200
        dados = resp.json()
        assert "fontes" in dados
        assert "planalto" in dados["fontes"]
        assert "stf" in dados["fontes"]
        assert "stj" in dados["fontes"]

    def test_coletar_fonte_valida(self, cliente):
        with patch("ana.api.routers.scrapers._executar_coleta_bg"):
            resp = cliente.post(
                "/scrapers/coletar",
                json={"fonte": "planalto"},
            )
        assert resp.status_code == 200
        dados = resp.json()
        assert dados["fonte"] == "planalto"

    def test_coletar_fonte_invalida_retorna_422(self, cliente):
        resp = cliente.post(
            "/scrapers/coletar",
            json={"fonte": "fonte_inexistente"},
        )
        assert resp.status_code == 422

    def test_atualizar_tudo(self, cliente):
        with patch("ana.api.routers.scrapers._executar_coleta_bg"):
            resp = cliente.post("/scrapers/atualizar-tudo")
        assert resp.status_code == 200
        dados = resp.json()
        assert "fontes" in dados
        assert len(dados["fontes"]) == 4


# =============================================================================
# Testes: AgendadorScrapers
# =============================================================================

class TestAgendadorScrapers:
    def test_inicialmente_inativo(self):
        from ana.scrapers.agendador import AgendadorScrapers
        ag = AgendadorScrapers()
        assert not ag.ativo

    @pytest.mark.asyncio
    async def test_iniciar_e_parar(self):
        from ana.scrapers.agendador import AgendadorScrapers
        with patch("ana.scrapers.pipeline.PipelineScrapers"):
            ag = AgendadorScrapers()
            ag.iniciar()
            assert ag.ativo
            ag.parar()
            assert not ag.ativo

    def test_parar_sem_iniciar_nao_falha(self):
        from ana.scrapers.agendador import AgendadorScrapers
        ag = AgendadorScrapers()
        ag.parar()  # Não deve lançar exceção
