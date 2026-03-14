"""Contratos de input/output para o router /scrapers."""

from pydantic import BaseModel, Field


# =============================================================================
# Coleta
# =============================================================================

class RequisicaoColeta(BaseModel):
    """Payload para disparar coleta de uma fonte.

    Attributes:
        fonte: Nome da fonte a coletar ('planalto', 'lexml', 'stf', 'stj').
    """

    fonte: str = Field(description="Nome da fonte: 'planalto', 'lexml', 'stf' ou 'stj'")


class RespostaColeta(BaseModel):
    """Resultado de uma operação de coleta.

    Attributes:
        fonte: Nome da fonte.
        documentos_novos: Documentos indexados nesta execução.
        documentos_ignorados: Documentos sem alteração (cache hit).
        erros: Lista de erros ocorridos.
        duracao_segundos: Tempo total de execução.
    """

    fonte: str
    documentos_novos: int
    documentos_ignorados: int
    erros: list[str]
    duracao_segundos: float


class RespostaAtualizarTudo(BaseModel):
    """Confirmação de início de atualização de todas as fontes.

    Attributes:
        mensagem: Confirmação de enfileiramento.
        fontes: Lista de fontes que serão atualizadas.
    """

    mensagem: str
    fontes: list[str]


# =============================================================================
# Status
# =============================================================================

class StatusFonte(BaseModel):
    """Status de uma fonte de scraping.

    Attributes:
        disponivel: True se a fonte está implementada e acessível.
        ultima_coleta: Timestamp ISO da última coleta bem-sucedida (ou None).
        documentos_indexados: Total de documentos indexados desta fonte.
        erro: Mensagem de erro da última tentativa (ou None).
    """

    disponivel: bool
    ultima_coleta: str | None = None
    documentos_indexados: int = 0
    erro: str | None = None


class StatusScrapers(BaseModel):
    """Status geral do pipeline de scrapers.

    Attributes:
        dependencias_instaladas: True se leis-br está instalado.
        fontes: Status individual de cada fonte.
        total_documentos_cache: Total de documentos em cache.
        erro: Mensagem de erro geral (ou None).
    """

    dependencias_instaladas: bool
    fontes: dict[str, StatusFonte]
    total_documentos_cache: int = 0
    erro: str | None = None


# =============================================================================
# Fontes
# =============================================================================

class InfoFonte(BaseModel):
    """Informações sobre uma fonte de scraping.

    Attributes:
        descricao: Descrição da fonte e conteúdo.
        tipo_documento: Tipo de documento coletado.
        intervalo_horas: Intervalo de atualização em horas.
        ultima_coleta: Timestamp ISO da última coleta (ou None).
        documentos_indexados: Total de documentos indexados.
    """

    descricao: str
    tipo_documento: str
    intervalo_horas: int
    ultima_coleta: str | None = None
    documentos_indexados: int = 0


class RespostaFontes(BaseModel):
    """Lista de fontes disponíveis e seus metadados.

    Attributes:
        fontes: Mapeamento nome → informações da fonte.
    """

    fontes: dict[str, InfoFonte]
