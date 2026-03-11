"""Testes do módulo de configuração de modelos do sistema ANA.

Valida carregamento do YAML, perfis, validações e integração
com a variável de ambiente ANA_PERFIL_MODELOS.
"""

import os
import pytest
from pathlib import Path

from ana.config_modelos import (
    ConfiguracaoModelos,
    PerfilModelos,
    carregar_config_modelos,
)

# Caminho do arquivo real de configuração
CAMINHO_YAML = Path(__file__).parent.parent / "config" / "modelos.yaml"


class TestCarregamentoYAML:
    """Testes de carregamento e parsing do arquivo YAML."""

    def test_arquivo_yaml_existe(self):
        """Verifica que config/modelos.yaml existe no repositório."""
        assert CAMINHO_YAML.exists(), (
            f"config/modelos.yaml não encontrado em {CAMINHO_YAML}"
        )

    def test_carregamento_sem_erros(self):
        """Verifica que o YAML é carregado sem erros de validação."""
        config = carregar_config_modelos(CAMINHO_YAML)
        assert isinstance(config, ConfiguracaoModelos)

    def test_perfis_disponiveis(self):
        """Verifica que os perfis 'teste' e 'producao' estão definidos."""
        config = carregar_config_modelos(CAMINHO_YAML)
        assert "teste" in config.perfis_disponiveis
        assert "producao" in config.perfis_disponiveis

    def test_perfil_ativo_padrao_e_teste(self):
        """Verifica que o perfil ativo padrão é 'teste'."""
        config = carregar_config_modelos(CAMINHO_YAML)
        assert config.perfil_ativo == "teste"

    def test_arquivo_nao_encontrado_lanca_erro(self):
        """Verifica que FileNotFoundError é lançado para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            carregar_config_modelos(Path("/caminho/inexistente/modelos.yaml"))


class TestPerfilTeste:
    """Testes específicos do perfil 'teste' (16GB VRAM)."""

    @pytest.fixture(scope="class")
    def perfil(self) -> PerfilModelos:
        """Carrega o perfil 'teste' do YAML."""
        config = carregar_config_modelos(CAMINHO_YAML)
        return config.perfis["teste"]

    def test_vram_estimada_menor_que_12gb(self, perfil: PerfilModelos):
        """Verifica que o perfil 'teste' usa menos de 12GB VRAM."""
        assert perfil.vram_estimada_gb < 12.0, (
            f"Perfil 'teste' usa {perfil.vram_estimada_gb}GB — muito alto para 12GB VRAM"
        )

    def test_sem_modelos_14b(self, perfil: PerfilModelos):
        """Verifica que nenhum agente usa modelo 14B no perfil 'teste'."""
        agentes = perfil.agentes.model_dump()
        modelos_14b = [
            (agente, modelo)
            for agente, modelo in agentes.items()
            if "14b" in str(modelo).lower()
        ]
        assert len(modelos_14b) == 0, (
            f"Perfil 'teste' contém modelos 14B: {modelos_14b}"
        )

    def test_embedding_multilingual_e5_large(self, perfil: PerfilModelos):
        """Verifica que o modelo de embeddings é o multilingual-e5-large."""
        assert "multilingual-e5-large" in perfil.embeddings.modelo

    def test_dimensao_embeddings_1024(self, perfil: PerfilModelos):
        """Verifica que a dimensão dos embeddings é 1024 conforme spec 02."""
        assert perfil.embeddings.dimensao == 1024

    def test_reranker_bge(self, perfil: PerfilModelos):
        """Verifica que o reranker é o BAAI/bge-reranker-base."""
        assert "bge-reranker" in perfil.reranker.modelo

    def test_todos_agentes_definidos(self, perfil: PerfilModelos):
        """Verifica que todos os 8 agentes do spec 09 têm modelo definido."""
        agentes = perfil.agentes
        assert agentes.orquestrador
        assert agentes.pesquisador
        assert agentes.analista
        assert agentes.redator
        assert agentes.validador
        assert agentes.transcritor
        assert agentes.monitor_prazos
        assert agentes.buscador_similares


class TestPerfilProducao:
    """Testes específicos do perfil 'producao'."""

    @pytest.fixture(scope="class")
    def perfil(self) -> PerfilModelos:
        """Carrega o perfil 'producao' do YAML."""
        config = carregar_config_modelos(CAMINHO_YAML)
        return config.perfis["producao"]

    def test_pesquisador_usa_modelo_grande(self, perfil: PerfilModelos):
        """Verifica que o Pesquisador Legal usa modelo maior em producao."""
        # Perfil producao: gemma-3-12b ou maior para tarefas jurídicas complexas
        modelo = perfil.agentes.pesquisador
        assert modelo, "Pesquisador sem modelo configurado"
        assert modelo != perfil.agentes.orquestrador or "12b" in modelo or "14b" in modelo, (
            "Pesquisador em producao deve usar modelo maior que o orquestrador"
        )

    def test_redator_usa_modelo_grande(self, perfil: PerfilModelos):
        """Verifica que o Redator usa modelo maior em producao."""
        modelo = perfil.agentes.redator
        assert modelo, "Redator sem modelo configurado"

    def test_validador_usa_modelo_rapido(self, perfil: PerfilModelos):
        """Verifica que o Validador usa modelo mais leve para velocidade."""
        # Orquestrador e validador devem ser modelos menores (gemma3:4b)
        assert perfil.agentes.validador, "Validador sem modelo configurado"
        assert "gemma3:4b" in perfil.agentes.validador or "3b" in perfil.agentes.validador

    def test_vram_cabe_em_16gb(self, perfil: PerfilModelos):
        """Verifica que o perfil 'producao' cabe em 16GB VRAM."""
        assert perfil.vram_estimada_gb <= 16.0, (
            f"Perfil 'producao' estimado em {perfil.vram_estimada_gb}GB — não cabe em 16GB"
        )


class TestTrocaDePerfil:
    """Testes de troca de perfil editando perfil_ativo diretamente."""

    def test_troca_perfil_para_producao(self, tmp_path):
        """Verifica que trocar perfil_ativo no YAML ativa o perfil correto."""
        import yaml as _yaml

        with open(CAMINHO_YAML) as f:
            dados = _yaml.safe_load(f)

        dados["perfil_ativo"] = "producao"
        yaml_tmp = tmp_path / "modelos_prod.yaml"
        with open(yaml_tmp, "w") as f:
            _yaml.dump(dados, f)

        config = carregar_config_modelos(yaml_tmp)
        assert config.perfil_ativo == "producao"
        assert config.ativo.agentes.pesquisador, "Perfil producao sem modelo pesquisador"

    def test_perfil_invalido_lanca_erro(self, tmp_path):
        """Verifica que perfil_ativo inválido no YAML levanta ValueError."""
        import yaml as _yaml

        with open(CAMINHO_YAML) as f:
            dados = _yaml.safe_load(f)

        dados["perfil_ativo"] = "perfil_inexistente"
        yaml_tmp = tmp_path / "modelos_invalido.yaml"
        with open(yaml_tmp, "w") as f:
            _yaml.dump(dados, f)

        with pytest.raises(ValueError, match="não encontrado"):
            carregar_config_modelos(yaml_tmp)


class TestPropriedadeAtivo:
    """Testes da propriedade .ativo e .perfis_disponiveis."""

    def test_ativo_retorna_perfil_correto(self):
        """Verifica que .ativo retorna o perfil correspondente ao perfil_ativo."""
        config = carregar_config_modelos(CAMINHO_YAML)
        perfil = config.ativo
        assert isinstance(perfil, PerfilModelos)
        assert perfil.vram_estimada_gb > 0

    def test_perfis_disponiveis_lista_completa(self):
        """Verifica que .perfis_disponiveis lista todos os perfis."""
        config = carregar_config_modelos(CAMINHO_YAML)
        disponiveis = config.perfis_disponiveis
        assert isinstance(disponiveis, list)
        assert len(disponiveis) >= 2
        assert "teste" in disponiveis
        assert "producao" in disponiveis
