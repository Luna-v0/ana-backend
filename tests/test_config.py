"""Testes do módulo de configuração do sistema ANA.

Valida que os valores padrão estão corretos e que a estrutura
de dataclass funciona conforme o esperado.
"""

import pytest
from ana.config import ConfiguracaoANA, obter_configuracao


class TestConfiguracaoANA:
    """Testes para a classe ConfiguracaoANA."""

    def test_valores_padrao_ollama(self):
        config = ConfiguracaoANA()
        assert config.ollama_host == "http://localhost:11434"

    def test_valores_padrao_qdrant(self):
        config = ConfiguracaoANA()
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.qdrant_grpc_port == 6334

    def test_valores_padrao_colecoes(self):
        config = ConfiguracaoANA()
        assert config.colecao_legislacao == "legislacao_brasileira"
        assert config.prefixo_colecao_sessao == "sessao"

    def test_valores_padrao_servidor(self):
        config = ConfiguracaoANA()
        assert config.backend_host == "0.0.0.0"
        assert config.backend_port == 8000

    def test_debug_false_por_padrao(self):
        config = ConfiguracaoANA()
        assert config.debug is False

    def test_log_level_padrao(self):
        config = ConfiguracaoANA()
        assert config.log_level == "INFO"

    def test_cors_origins_apenas_localhost(self):
        """LGPD: somente origens localhost são permitidas."""
        config = ConfiguracaoANA()
        assert isinstance(config.cors_origins, list)
        assert len(config.cors_origins) >= 1
        for origem in config.cors_origins:
            assert "localhost" in origem, (
                f"LGPD: apenas origens localhost permitidas, encontrou: {origem}"
            )

    def test_alterar_valores_diretamente(self):
        """Confirma que a config é editável como dataclass Python."""
        config = ConfiguracaoANA(qdrant_port=9999, debug=True)
        assert config.qdrant_port == 9999
        assert config.debug is True

    def test_singleton_obter_configuracao(self):
        """obter_configuracao() retorna sempre a mesma instância."""
        a = obter_configuracao()
        b = obter_configuracao()
        assert a is b
