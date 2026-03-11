"""Router de health check para monitoramento dos serviços de infraestrutura.

Verifica disponibilidade de Ollama e Qdrant via HTTP, retornando
status detalhado de cada serviço conforme descrito no spec 01 e spec 10.
"""

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from ana.config import obter_configuracao

router = APIRouter(prefix="/health", tags=["Sistema"])


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
        status: Status geral: 'ok', 'degradado' ou 'offline'.
        versao: Versão do sistema ANA.
        servicos: Lista de status de cada serviço de infraestrutura.
        modelos: Informações do perfil de modelos ativo.
    """

    status: str
    versao: str
    servicos: list[StatusServico]
    modelos: InfoModelos


async def _verificar_ollama(host: str) -> StatusServico:
    """Verifica disponibilidade do Ollama via endpoint /api/tags.

    Args:
        host: URL base do Ollama (ex: 'http://localhost:11434').

    Returns:
        StatusServico com resultado da verificação.
    """
    url = f"{host}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as cliente:
            resposta = await cliente.get(url)
            if resposta.status_code == 200:
                dados = resposta.json()
                modelos = [m["name"] for m in dados.get("models", [])]
                detalhes = f"Modelos disponíveis: {', '.join(modelos) or 'nenhum'}"
                return StatusServico(
                    nome="ollama",
                    disponivel=True,
                    url=host,
                    detalhes=detalhes,
                )
    except Exception as erro:
        return StatusServico(
            nome="ollama",
            disponivel=False,
            url=host,
            detalhes=str(erro),
        )
    return StatusServico(
        nome="ollama",
        disponivel=False,
        url=host,
        detalhes="Resposta inesperada do serviço",
    )


async def _verificar_qdrant(host: str, porta: int) -> StatusServico:
    """Verifica disponibilidade do Qdrant via endpoint /collections.

    Args:
        host: Hostname do Qdrant.
        porta: Porta REST do Qdrant.

    Returns:
        StatusServico com resultado da verificação.
    """
    url_base = f"http://{host}:{porta}"
    url = f"{url_base}/collections"
    try:
        async with httpx.AsyncClient(timeout=5.0) as cliente:
            resposta = await cliente.get(url)
            if resposta.status_code == 200:
                dados = resposta.json()
                colecoes = [
                    c["name"]
                    for c in dados.get("result", {}).get("collections", [])
                ]
                detalhes = f"Collections: {', '.join(colecoes) or 'nenhuma'}"
                return StatusServico(
                    nome="qdrant",
                    disponivel=True,
                    url=url_base,
                    detalhes=detalhes,
                )
    except Exception as erro:
        return StatusServico(
            nome="qdrant",
            disponivel=False,
            url=url_base,
            detalhes=str(erro),
        )
    return StatusServico(
        nome="qdrant",
        disponivel=False,
        url=url_base,
        detalhes="Resposta inesperada do serviço",
    )


@router.get(
    "/",
    response_model=RespostaHealth,
    summary="Health check completo",
    description=(
        "Verifica a disponibilidade de todos os serviços de infraestrutura "
        "(Ollama, Qdrant). Retorna 'ok' quando todos estão disponíveis, "
        "'degradado' quando algum está offline."
    ),
)
async def health_check() -> RespostaHealth:
    """Verifica saúde de todos os serviços de infraestrutura.

    Returns:
        RespostaHealth com status de cada serviço, perfil de modelos ativo
        e status geral do sistema.
    """
    from ana import __version__
    from ana.config_modelos import obter_modelos

    config = obter_configuracao()
    modelos = obter_modelos()
    perfil = modelos.ativo

    servicos = [
        await _verificar_ollama(config.ollama_host),
        await _verificar_qdrant(config.qdrant_host, config.qdrant_port),
    ]

    todos_disponiveis = all(s.disponivel for s in servicos)
    status = "ok" if todos_disponiveis else "degradado"

    return RespostaHealth(
        status=status,
        versao=__version__,
        servicos=servicos,
        modelos=InfoModelos(
            perfil_ativo=modelos.perfil_ativo,
            vram_estimada_gb=perfil.vram_estimada_gb,
            pesquisador=perfil.agentes.pesquisador,
            orquestrador=perfil.agentes.orquestrador,
            embedding=perfil.embeddings.modelo,
        ),
    )
