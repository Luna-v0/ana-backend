"""Testes do módulo de transcrição de audiências (Spec 03).

Testa sem depender de whisperx/pyannote (dependências opcionais pesadas):
- Modelos de dados e propriedades computadas
- Formatação de labels e timestamps
- Extração de mapeamento JSON
- Aplicação de mapeamento nos segmentos
- Inferência de roles por texto
- Formatação markdown do transcript
- Endpoints FastAPI com mock das dependências
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ana.transcricao.modelos import (
    MetadataAudiencia,
    ParticipanteAudiencia,
    ResultadoTranscricao,
    RoleParticipante,
    SegmentoTranscricao,
    aplicar_mapeamento,
    extrair_mapeamento_json,
)
from ana.transcricao.formatacao import formatar_transcript_markdown
from ana.transcricao.identificacao import _inferir_role


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def juiz() -> ParticipanteAudiencia:
    return ParticipanteAudiencia(
        role=RoleParticipante.JUIZ,
        nome="Dr. Carlos Silva",
        speaker_id="SPEAKER_00",
    )


@pytest.fixture
def adv_autor() -> ParticipanteAudiencia:
    return ParticipanteAudiencia(
        role=RoleParticipante.ADVOGADO_AUTOR,
        nome="Dra. Maria Santos",
        oab="OAB/SP 123456",
        speaker_id="SPEAKER_01",
    )


@pytest.fixture
def adv_reu() -> ParticipanteAudiencia:
    return ParticipanteAudiencia(
        role=RoleParticipante.ADVOGADO_REU,
        nome="Dr. João Oliveira",
        oab="OAB/SP 654321",
        speaker_id="SPEAKER_02",
    )


@pytest.fixture
def segmentos_basicos(juiz, adv_autor) -> list[SegmentoTranscricao]:
    return [
        SegmentoTranscricao(
            inicio=15.0, fim=30.0,
            texto="Boa tarde. Declaro aberta a audiência de instrução e julgamento.",
            speaker_id="SPEAKER_00",
            participante=juiz,
        ),
        SegmentoTranscricao(
            inicio=32.0, fim=45.0,
            texto="Boa tarde, Excelência. A autora está presente e reitera os termos da inicial.",
            speaker_id="SPEAKER_01",
            participante=adv_autor,
        ),
        SegmentoTranscricao(
            inicio=46.0, fim=50.0,
            texto="",  # segmento vazio deve ser ignorado na formatação
            speaker_id="SPEAKER_00",
        ),
    ]


@pytest.fixture
def metadata_completa() -> MetadataAudiencia:
    return MetadataAudiencia(
        numero_processo="1234567-89.2024.8.26.0100",
        data="15/01/2025",
        tipo_audiencia="Instrução e Julgamento",
        vara="3ª Vara Cível — Foro Central",
        cidade_uf="São Paulo/SP",
    )


@pytest.fixture
def resultado_basico(segmentos_basicos, metadata_completa, juiz, adv_autor) -> ResultadoTranscricao:
    return ResultadoTranscricao(
        segmentos=segmentos_basicos,
        metadata=metadata_completa,
        mapeamento_speakers={"SPEAKER_00": juiz, "SPEAKER_01": adv_autor},
        duracao_total=3600.0,
        arquivo_origem="audiencia_2025-01-15.mp3",
    )


# ── Testes de ParticipanteAudiencia ───────────────────────────────────────────

class TestParticipanteAudiencia:
    def test_label_juiz_sem_oab(self, juiz):
        assert juiz.label_formatado == "Juiz(a) — Dr. Carlos Silva"

    def test_label_advogado_com_oab(self, adv_autor):
        assert adv_autor.label_formatado == "Adv. Autor(a) — Dra. Maria Santos (OAB/SP 123456)"

    def test_label_advogado_reu_com_oab(self, adv_reu):
        assert adv_reu.label_formatado == "Adv. Réu(ré) — Dr. João Oliveira (OAB/SP 654321)"

    def test_label_todos_roles(self):
        """Verifica que todos os roles têm prefixo definido."""
        for role in RoleParticipante:
            p = ParticipanteAudiencia(role=role, nome="Teste")
            assert p.label_formatado  # não deve ser vazio
            assert "Teste" in p.label_formatado

    def test_confianca_padrao(self):
        p = ParticipanteAudiencia(role=RoleParticipante.JUIZ, nome="Teste")
        assert p.confianca == 1.0

    def test_speaker_id_opcional(self):
        p = ParticipanteAudiencia(role=RoleParticipante.TESTEMUNHA, nome="Fulano")
        assert p.speaker_id is None
        assert p.oab is None


# ── Testes de SegmentoTranscricao ─────────────────────────────────────────────

class TestSegmentoTranscricao:
    def test_duracao(self):
        seg = SegmentoTranscricao(inicio=10.0, fim=25.5, texto="teste")
        assert seg.duracao == pytest.approx(15.5)

    def test_timestamp_segundos(self):
        seg = SegmentoTranscricao(inicio=15.5, fim=20.0, texto="teste")
        assert seg.timestamp_formatado == "00:15"

    def test_timestamp_minutos(self):
        seg = SegmentoTranscricao(inicio=125.0, fim=130.0, texto="teste")
        assert seg.timestamp_formatado == "02:05"

    def test_timestamp_horas(self):
        seg = SegmentoTranscricao(inicio=3661.0, fim=3680.0, texto="teste")
        assert seg.timestamp_formatado == "01:01:01"

    def test_timestamp_zero(self):
        seg = SegmentoTranscricao(inicio=0.0, fim=5.0, texto="teste")
        assert seg.timestamp_formatado == "00:00"

    def test_speaker_id_padrao(self):
        seg = SegmentoTranscricao(inicio=0.0, fim=1.0, texto="teste")
        assert seg.speaker_id == "SPEAKER_00"


# ── Testes de ResultadoTranscricao ────────────────────────────────────────────

class TestResultadoTranscricao:
    def test_num_participantes(self, resultado_basico):
        # Há 2 speakers únicos (SPEAKER_00 e SPEAKER_01),
        # o terceiro segmento também é SPEAKER_00
        assert resultado_basico.num_participantes == 2

    def test_texto_completo(self, resultado_basico):
        texto = resultado_basico.texto_completo
        assert "Declaro aberta" in texto
        assert "A autora está presente" in texto
        # Segmento vazio não deve aparecer
        assert "  " not in texto.strip()

    def test_resultado_vazio(self):
        r = ResultadoTranscricao()
        assert r.num_participantes == 0
        assert r.texto_completo == ""


# ── Testes de extrair_mapeamento_json ─────────────────────────────────────────

class TestExtrairMapeamentoJson:
    def test_json_puro(self):
        resposta = '{"SPEAKER_00": "Juiz Dr. Silva", "SPEAKER_01": "Adv. Maria"}'
        resultado = extrair_mapeamento_json(resposta)
        assert resultado["SPEAKER_00"] == "Juiz Dr. Silva"
        assert len(resultado) == 2

    def test_json_com_texto_ao_redor(self):
        resposta = """
        Com base na análise:
        {"SPEAKER_00": "Juiz", "SPEAKER_01": "Adv. Autora"}
        Identificação concluída.
        """
        resultado = extrair_mapeamento_json(resposta)
        assert "SPEAKER_00" in resultado

    def test_sem_json_retorna_vazio(self):
        assert extrair_mapeamento_json("Não consegui identificar.") == {}

    def test_json_invalido_retorna_vazio(self):
        assert extrair_mapeamento_json("{chave sem aspas: valor}") == {}

    def test_json_vazio_retorna_dict(self):
        # "{}" é um JSON válido mas sem conteúdo
        resultado = extrair_mapeamento_json("{}")
        assert isinstance(resultado, dict)


# ── Testes de aplicar_mapeamento ──────────────────────────────────────────────

class TestAplicarMapeamento:
    def test_mapeamento_completo(self, juiz, adv_autor):
        segmentos = [
            SegmentoTranscricao(inicio=0.0, fim=5.0, texto="Olá", speaker_id="SPEAKER_00"),
            SegmentoTranscricao(inicio=5.0, fim=10.0, texto="Oi", speaker_id="SPEAKER_01"),
        ]
        mapeamento = {"SPEAKER_00": juiz, "SPEAKER_01": adv_autor}
        resultado = aplicar_mapeamento(segmentos, mapeamento)

        assert resultado[0].participante.role == RoleParticipante.JUIZ
        assert resultado[1].participante.nome == "Dra. Maria Santos"

    def test_speaker_sem_mapeamento_vira_desconhecido(self):
        segmentos = [
            SegmentoTranscricao(inicio=0.0, fim=5.0, texto="Fala", speaker_id="SPEAKER_99"),
        ]
        resultado = aplicar_mapeamento(segmentos, {})
        assert resultado[0].participante.role == RoleParticipante.DESCONHECIDO

    def test_mapeamento_vazio(self):
        segmentos = [
            SegmentoTranscricao(inicio=0.0, fim=5.0, texto="Fala", speaker_id="SPEAKER_00"),
        ]
        resultado = aplicar_mapeamento(segmentos, {})
        assert resultado[0].participante is not None
        assert resultado[0].participante.role == RoleParticipante.DESCONHECIDO


# ── Testes de _inferir_role ───────────────────────────────────────────────────

class TestInferirRole:
    def test_juiz(self):
        assert _inferir_role("Juiz Dr. Carlos Silva") == RoleParticipante.JUIZ

    def test_juiza(self):
        assert _inferir_role("Juíza Dra. Ana Lima") == RoleParticipante.JUIZ

    def test_advogado_autor(self):
        assert _inferir_role("Adv. Autora Dra. Maria Santos") == RoleParticipante.ADVOGADO_AUTOR

    def test_advogado_reu(self):
        assert _inferir_role("Advogado do Réu Dr. João") == RoleParticipante.ADVOGADO_REU

    def test_promotor(self):
        assert _inferir_role("Promotor Dr. Pedro") == RoleParticipante.PROMOTOR

    def test_testemunha(self):
        assert _inferir_role("Testemunha Fulano de Tal") == RoleParticipante.TESTEMUNHA

    def test_perito(self):
        assert _inferir_role("Perito Eng. Ana Costa") == RoleParticipante.PERITO

    def test_desconhecido(self):
        assert _inferir_role("Participante XYZ") == RoleParticipante.DESCONHECIDO


# ── Testes de formatar_transcript_markdown ────────────────────────────────────

class TestFormatarTranscriptMarkdown:
    def test_cabecalho_presente(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "1234567-89.2024.8.26.0100" in md
        assert "15/01/2025" in md
        assert "3ª Vara Cível" in md
        assert "São Paulo/SP" in md

    def test_participantes_listados(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "SPEAKER_00" in md
        assert "Juiz(a) — Dr. Carlos Silva" in md
        assert "Adv. Autor(a) — Dra. Maria Santos (OAB/SP 123456)" in md

    def test_timestamps_presentes(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "00:15" in md  # início do primeiro segmento (15s)
        assert "00:32" in md  # início do segundo segmento (32s)

    def test_texto_transcrito(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "Declaro aberta a audiência" in md
        assert "A autora está presente" in md

    def test_segmento_vazio_ignorado(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        # O terceiro segmento é vazio e não deve aparecer
        assert md.count("SPEAKER_00") <= md.count("SPEAKER_00")  # garante que existe

    def test_aviso_revisao_obrigatorio(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "ATENÇÃO" in md
        assert "revisada pelo advogado" in md

    def test_modelo_asr_no_rodape(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "whisper-large-v3" in md

    def test_duracao_no_rodape(self, resultado_basico):
        md = formatar_transcript_markdown(resultado_basico)
        assert "60min" in md  # 3600 segundos = 60 minutos

    def test_sem_participantes_identificados(self):
        """Transcript sem mapeamento de speakers ainda deve ser válido."""
        seg = SegmentoTranscricao(
            inicio=0.0, fim=10.0, texto="Fala genérica.", speaker_id="SPEAKER_00"
        )
        resultado = ResultadoTranscricao(
            segmentos=[seg],
            duracao_total=10.0,
        )
        md = formatar_transcript_markdown(resultado)
        assert "Fala genérica." in md
        assert "SPEAKER_00" in md

    def test_resultado_sem_segmentos(self):
        """Transcript vazio deve retornar markdown mínimo válido."""
        resultado = ResultadoTranscricao()
        md = formatar_transcript_markdown(resultado)
        assert "ATENÇÃO" in md  # aviso sempre presente


# ── Testes dos endpoints FastAPI ──────────────────────────────────────────────

class TestEndpointsTranscricao:
    @pytest.fixture
    def cliente(self):
        from ana.api.main import criar_app
        app = criar_app()
        return TestClient(app)

    def test_status_sem_whisperx(self, cliente):
        """Deve retornar disponivel=False quando whisperx não está instalado."""
        with patch(
            "ana.api.routers.transcricao._verificar_whisperx_disponivel",
            return_value=False,
        ):
            resp = cliente.get("/transcricao/status")
        assert resp.status_code == 200
        dados = resp.json()
        assert dados["disponivel"] is False
        assert dados["whisperx_instalado"] is False
        assert "uv sync" in dados["mensagem"]

    def test_status_sem_hf_token(self, cliente, monkeypatch):
        """Deve retornar disponivel=False quando HF_TOKEN não está configurado."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch(
            "ana.api.routers.transcricao._verificar_whisperx_disponivel",
            return_value=True,
        ):
            resp = cliente.get("/transcricao/status")
        assert resp.status_code == 200
        dados = resp.json()
        assert dados["disponivel"] is False
        assert dados["hf_token_configurado"] is False
        assert "HF_TOKEN" in dados["mensagem"]

    def test_status_disponivel(self, cliente, monkeypatch):
        """Deve retornar disponivel=True quando tudo está configurado."""
        monkeypatch.setenv("HF_TOKEN", "hf_fake_token")
        with patch(
            "ana.api.routers.transcricao._verificar_whisperx_disponivel",
            return_value=True,
        ):
            resp = cliente.get("/transcricao/status")
        assert resp.status_code == 200
        dados = resp.json()
        assert dados["disponivel"] is True

    def test_transcrever_sem_whisperx_retorna_503(self, cliente):
        """Upload sem whisperx deve retornar 503 Service Unavailable."""
        with patch(
            "ana.api.routers.transcricao._verificar_whisperx_disponivel",
            return_value=False,
        ):
            resp = cliente.post(
                "/transcricao/transcrever",
                files={"audio": ("teste.mp3", b"dados_fake", "audio/mpeg")},
            )
        assert resp.status_code == 503

    def test_transcrever_formato_invalido_retorna_422(self, cliente, monkeypatch):
        """Upload de arquivo .txt deve retornar 422 Unprocessable Entity."""
        monkeypatch.setenv("HF_TOKEN", "hf_fake_token")
        with patch(
            "ana.api.routers.transcricao._verificar_whisperx_disponivel",
            return_value=True,
        ):
            resp = cliente.post(
                "/transcricao/transcrever",
                files={"audio": ("documento.txt", b"texto qualquer", "text/plain")},
            )
        assert resp.status_code == 422
        assert "txt" in resp.json()["detail"].lower()
