"""Contratos de input/output para o router /health."""

from pydantic import BaseModel


class StatusServico(BaseModel):
    """Status de um serviço de infraestrutura.

    Attributes:
        nome: Nome identificador do serviço.
        disponivel: True se o serviço está respondendo.
        url: URL ou endereço do serviço verificado.
        detalhes: Mensagem adicional (versão, erro, etc.).
    """

    nome: str
    disponivel: bool
    url: str
    detalhes: str = ""


class InfoModelos(BaseModel):
    """Informações sobre o perfil de modelos ativo.

    Attributes:
        perfil_ativo: Nome do perfil em uso ('teste' ou 'producao').
        vram_estimada_gb: Estimativa de VRAM necessária pelo perfil.
        pesquisador: Modelo do Agente Pesquisador Legal.
        orquestrador: Modelo do Orquestrador.
        embedding: Modelo de embeddings semânticos.
    """

    perfil_ativo: str
    vram_estimada_gb: float
    pesquisador: str
    orquestrador: str
    embedding: str


class RespostaHealth(BaseModel):
    """Resposta completa do endpoint de health check.

    Attributes:
        status: Status geral: 'ok' ou 'degradado'.
        versao: Versão do sistema ANA.
        servicos: Lista de status de cada serviço de infraestrutura.
        modelos: Informações do perfil de modelos ativo.
    """

    status: str
    versao: str
    servicos: list[StatusServico]
    modelos: InfoModelos
