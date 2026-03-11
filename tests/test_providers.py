"""Testes para o pacote ana.providers.

Testa:
1. OllamaLLMProvider.invocar() com mock de httpx
2. Conformidade estrutural (@runtime_checkable isinstance) das classes existentes
3. LLMProvider como mock direto na identificação de participantes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ana.providers import (
    ASRProvider,
    DiarizacaoProvider,
    EmbeddingProvider,
    LLMProvider,
    OllamaLLMProvider,
    RerankerProvider,
    VectorStoreProvider,
)


# ── OllamaLLMProvider ─────────────────────────────────────────────────────────

class TestOllamaLLMProvider:
    def test_invocar_retorna_resposta(self):
        """invocar() deve retornar o texto da chave 'response' do JSON Ollama."""
        provedor = OllamaLLMProvider(modelo="qwen2.5:7b", host="http://localhost:11434")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Resposta do modelo"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            resultado = provedor.invocar("Qual é o juiz?")

        assert resultado == "Resposta do modelo"

    def test_invocar_envia_temperatura(self):
        """invocar() deve enviar a temperatura correta no payload."""
        provedor = OllamaLLMProvider(modelo="llama3:8b")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            provedor.invocar("prompt", temperatura=0.5)

            _, call_kwargs = mock_client.post.call_args
            payload = call_kwargs["json"]

        assert payload["options"]["temperature"] == 0.5

    def test_invocar_usa_modelo_correto(self):
        """invocar() deve usar o modelo configurado no construtor."""
        provedor = OllamaLLMProvider(modelo="gemma2:9b")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            provedor.invocar("prompt")

            _, call_kwargs = mock_client.post.call_args
            assert call_kwargs["json"]["model"] == "gemma2:9b"

    def test_invocar_resposta_ausente_retorna_string_vazia(self):
        """invocar() deve retornar '' se 'response' não estiver no JSON."""
        provedor = OllamaLLMProvider(modelo="qwen2.5:7b")

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response

            resultado = provedor.invocar("prompt")

        assert resultado == ""

    def test_ollama_conforma_llmprovider(self):
        """OllamaLLMProvider deve conformar com LLMProvider via isinstance."""
        provedor = OllamaLLMProvider(modelo="qwen2.5:7b")
        assert isinstance(provedor, LLMProvider)


# ── Conformidade estrutural dos Protocols ─────────────────────────────────────

class TestConformidadeEstrutural:
    """Verifica que as classes existentes conformam com os Protocols."""

    def test_gerador_embeddings_conforma_embedding_provider(self):
        """GeradorEmbeddings deve conformar com EmbeddingProvider."""
        from unittest.mock import MagicMock, patch

        # Evita carregar sentence-transformers pesado: mocka apenas o __init__
        with patch("ana.config_modelos.obter_modelos") as mock_obter:
            mock_config = MagicMock()
            mock_config.ativo.embeddings.modelo = "test-model"
            mock_config.ativo.embeddings.dispositivo = "cpu"
            mock_config.ativo.embeddings.batch_size = 32
            mock_config.ativo.embeddings.dimensao = 1024
            mock_obter.return_value = mock_config

            from ana.rag.embeddings import GeradorEmbeddings
            gerador = GeradorEmbeddings.__new__(GeradorEmbeddings)
            gerador.modelo_nome = "test"
            gerador.dispositivo = "cpu"
            gerador.batch_size = 32
            gerador.dimensao = 1024
            gerador._modelo = None

        assert isinstance(gerador, EmbeddingProvider)

    def test_classe_com_invocar_conforma_llmprovider(self):
        """Qualquer classe com método invocar() deve conformar com LLMProvider."""
        class FakeLLM:
            def invocar(self, prompt: str, temperatura: float = 0.1, **kwargs: Any) -> str:
                return "resposta fake"

        assert isinstance(FakeLLM(), LLMProvider)

    def test_objeto_sem_invocar_nao_conforma_llmprovider(self):
        """Objeto sem método invocar() não deve conformar com LLMProvider."""
        class SemInvocar:
            def chamar(self, prompt: str) -> str:
                return ""

        assert not isinstance(SemInvocar(), LLMProvider)

    def test_objeto_sem_gerar_nao_conforma_embedding_provider(self):
        """Objeto sem método gerar() não deve conformar com EmbeddingProvider."""
        class Incompleto:
            def gerar_batch(self, textos: list[str]) -> list[list[float]]:
                return []
            # faltam gerar() e gerar_query()

        assert not isinstance(Incompleto(), EmbeddingProvider)

    def test_objeto_sem_rerankar_nao_conforma_reranker_provider(self):
        """Objeto sem método rerankar() não deve conformar com RerankerProvider."""
        class SemRerankar:
            pass

        assert not isinstance(SemRerankar(), RerankerProvider)


# ── LLMProvider como mock em identificar_participantes ────────────────────────

class TestIdentificacaoComMockLLM:
    """Testa identificar_participantes usando mock de LLMProvider."""

    @pytest.fixture
    def segmentos(self):
        from ana.transcricao.modelos import SegmentoTranscricao

        return [
            SegmentoTranscricao(
                inicio=0.0, fim=10.0,
                texto="Declaro aberta a audiência.",
                speaker_id="SPEAKER_00",
            ),
            SegmentoTranscricao(
                inicio=10.0, fim=20.0,
                texto="Sou a advogada da autora, Dra. Maria Santos, OAB/SP 123456.",
                speaker_id="SPEAKER_01",
            ),
        ]

    @pytest.fixture
    def metadata(self):
        from ana.transcricao.modelos import MetadataAudiencia

        return MetadataAudiencia(
            numero_processo="0001234-56.2024.8.26.0100",
            data="10/01/2025",
            tipo_audiencia="Instrução",
            vara="1ª Vara Cível",
            cidade_uf="São Paulo/SP",
        )

    def test_identificar_com_llm_retorna_mapeamento(self, segmentos, metadata):
        """identificar_participantes deve usar provedor_llm.invocar()."""
        from ana.transcricao.identificacao import identificar_participantes

        mock_llm = MagicMock()
        mock_llm.invocar.return_value = (
            '{"SPEAKER_00": "Juiz Dr. Fulano", "SPEAKER_01": "Adv. Autora Dra. Maria Santos"}'
        )

        segs, mapeamento = identificar_participantes(segmentos, metadata, mock_llm)

        mock_llm.invocar.assert_called_once()
        assert "SPEAKER_00" in mapeamento or "SPEAKER_01" in mapeamento

    def test_identificar_llm_com_falha_retorna_mapeamento_parcial(
        self, segmentos, metadata
    ):
        """Se LLM lançar exceção, deve retornar mapeamento da Camada 1."""
        from ana.transcricao.identificacao import identificar_participantes

        mock_llm = MagicMock()
        mock_llm.invocar.side_effect = RuntimeError("Ollama indisponível")

        segs, mapeamento = identificar_participantes(segmentos, metadata, mock_llm)

        # Não deve propagar a exceção; mapeamento pode ser vazio ou parcial
        assert isinstance(mapeamento, dict)
        assert isinstance(segs, list)

    def test_identificar_llm_json_invalido_gracioso(self, segmentos, metadata):
        """Se LLM retornar JSON inválido, deve continuar sem falhar."""
        from ana.transcricao.identificacao import identificar_participantes

        mock_llm = MagicMock()
        mock_llm.invocar.return_value = "Não consegui identificar os participantes."

        segs, mapeamento = identificar_participantes(segmentos, metadata, mock_llm)

        assert isinstance(mapeamento, dict)
