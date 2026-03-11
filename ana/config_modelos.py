"""Módulo de configuração de modelos do sistema ANA.

Carrega e valida o arquivo config/modelos.yaml, que define os modelos
LLM, de embeddings e de reranking para cada perfil de execução.

A troca entre perfis é feita alterando 'perfil_ativo' diretamente no
arquivo config/modelos.yaml.

Exemplo de uso:
    >>> from ana.config_modelos import obter_modelos
    >>> modelos = obter_modelos()
    >>> print(modelos.ativo.agentes.pesquisador)
    'qwen2.5:7b'
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# Caminho padrão do arquivo de configuração de modelos
# flat layout: ana/config_modelos.py → .parent = ana/ → .parent = project root
_CAMINHO_PADRAO = Path(__file__).parent.parent / "config" / "modelos.yaml"


class ConfigEmbeddings(BaseModel):
    """Configuração do modelo de embeddings semânticos.

    Attributes:
        modelo: Nome do modelo no HuggingFace Hub.
        dispositivo: Dispositivo de execução ('cuda' ou 'cpu').
        dimensao: Dimensão dos vetores gerados pelo modelo.
        batch_size: Tamanho do batch para geração em lote.
    """

    modelo: str = Field(description="Nome do modelo no HuggingFace Hub")
    dispositivo: str = Field(
        default="cuda",
        description="Dispositivo de execução: 'cuda' ou 'cpu'",
    )
    dimensao: int = Field(
        default=1024,
        description="Dimensão dos vetores (1024 para multilingual-e5-large)",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size para geração de embeddings em lote",
    )


class ConfigReranker(BaseModel):
    """Configuração do modelo de reranking (CrossEncoder).

    Attributes:
        modelo: Nome do modelo CrossEncoder no HuggingFace Hub.
        dispositivo: Dispositivo de execução ('cuda' ou 'cpu').
    """

    modelo: str = Field(description="Nome do modelo CrossEncoder no HuggingFace Hub")
    dispositivo: str = Field(
        default="cuda",
        description="Dispositivo de execução: 'cuda' ou 'cpu'",
    )


class ConfigAgentes(BaseModel):
    """Modelos LLM atribuídos a cada agente do sistema.

    Segue a estratégia do spec 09: modelos maiores para tarefas
    complexas (pesquisa, redação) e menores para tarefas simples
    (classificação, validação binária).

    Attributes:
        orquestrador: SLM rápido para classificação de intenção.
        pesquisador: LLM para síntese e busca jurídica.
        analista: LLM para análise de processo e documentos.
        redator: LLM para geração de peças processuais.
        validador: SLM para validação de leis citadas.
        transcritor: LLM para identificação de participantes.
        monitor_prazos: SLM para extração de prazos.
        buscador_similares: SLM para análise de similaridade.
    """

    orquestrador: str
    pesquisador: str
    analista: str
    redator: str
    validador: str
    transcritor: str
    monitor_prazos: str
    buscador_similares: str


class PerfilModelos(BaseModel):
    """Perfil completo de modelos para um ambiente de execução.

    Attributes:
        descricao: Descrição do perfil e seu caso de uso.
        vram_estimada_gb: Estimativa de VRAM necessária em GB.
        agentes: Modelos por agente.
        embeddings: Configuração do modelo de embeddings.
        reranker: Configuração do modelo de reranking.
    """

    descricao: str
    vram_estimada_gb: float
    agentes: ConfigAgentes
    embeddings: ConfigEmbeddings
    reranker: ConfigReranker


class ConfiguracaoModelos(BaseModel):
    """Configuração completa de modelos com múltiplos perfis.

    Carregada do arquivo config/modelos.yaml. O perfil ativo pode ser
    sobrescrito pela variável de ambiente ANA_PERFIL_MODELOS.

    Attributes:
        perfil_ativo: Nome do perfil em uso.
        perfis: Dicionário de perfis disponíveis.
        storage_backend: Backend de armazenamento (``'qdrant'`` ou ``'postgres'``).
    """

    perfil_ativo: str = Field(
        default="teste",
        description="Nome do perfil de modelos ativo",
    )
    perfis: dict[str, PerfilModelos] = Field(
        description="Perfis de modelos disponíveis",
    )
    storage_backend: str = Field(
        default="qdrant",
        description="Backend de armazenamento: 'qdrant' (padrão) ou 'postgres'",
    )

    @model_validator(mode="after")
    def validar_perfil_ativo(self) -> "ConfiguracaoModelos":
        """Valida que o perfil ativo existe nos perfis definidos.

        Returns:
            Própria instância após validação.

        Raises:
            ValueError: Se o perfil ativo não estiver nos perfis definidos.
        """
        if self.perfil_ativo not in self.perfis:
            disponiveis = list(self.perfis.keys())
            raise ValueError(
                f"Perfil '{self.perfil_ativo}' não encontrado em config/modelos.yaml. "
                f"Perfis disponíveis: {disponiveis}"
            )
        return self

    @property
    def ativo(self) -> PerfilModelos:
        """Retorna o perfil de modelos atualmente ativo.

        Returns:
            Instância do PerfilModelos correspondente ao perfil_ativo.
        """
        return self.perfis[self.perfil_ativo]

    @property
    def perfis_disponiveis(self) -> list[str]:
        """Retorna lista de nomes de perfis disponíveis.

        Returns:
            Lista de strings com os nomes dos perfis definidos no YAML.
        """
        return list(self.perfis.keys())


def carregar_config_modelos(
    caminho: Path | None = None,
) -> ConfiguracaoModelos:
    """Carrega configuração de modelos a partir do arquivo YAML.

    A variável de ambiente ANA_PERFIL_MODELOS, se definida, sobrescreve
    o valor de 'perfil_ativo' do arquivo.

    Args:
        caminho: Caminho do arquivo YAML. Usa o padrão config/modelos.yaml
            se None.

    Returns:
        Instância validada de ConfiguracaoModelos.

    Raises:
        FileNotFoundError: Se o arquivo YAML não for encontrado.
        ValueError: Se o conteúdo do YAML for inválido.
    """
    caminho_efetivo = caminho or _CAMINHO_PADRAO

    if not caminho_efetivo.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração de modelos não encontrado: {caminho_efetivo}\n"
            "Copie config/modelos.yaml e ajuste conforme seu hardware."
        )

    with open(caminho_efetivo, encoding="utf-8") as arquivo:
        dados: dict[str, Any] = yaml.safe_load(arquivo)

    return ConfiguracaoModelos(**dados)


@lru_cache(maxsize=1)
def obter_modelos() -> ConfiguracaoModelos:
    """Retorna instância singleton da configuração de modelos (com cache).

    Returns:
        Instância única de ConfiguracaoModelos para o ciclo de vida da app.
    """
    return carregar_config_modelos()
