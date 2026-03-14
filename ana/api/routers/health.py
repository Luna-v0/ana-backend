"""Router de health check para monitoramento dos serviços de infraestrutura.

Verifica disponibilidade de Ollama e PostgreSQL, retornando
status detalhado de cada serviço conforme descrito no spec 01 e spec 10.
"""

import httpx
import psycopg
from fastapi import APIRouter

from ana.config import obter_configuracao
from ana.api.schemas.health import InfoModelos, RespostaHealth, StatusServico

router = APIRouter(prefix="/health", tags=["Sistema"])


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


async def _verificar_postgres(dsn: str) -> StatusServico:
    """Verifica disponibilidade do PostgreSQL via conexão psycopg.

    Args:
        dsn: DSN de conexão PostgreSQL (ex: 'postgresql://user:pass@host:5432/db').

    Returns:
        StatusServico com resultado da verificação.
    """
    url_display = dsn.split("@")[-1] if "@" in dsn else dsn
    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            await conn.execute("SELECT 1")
        return StatusServico(
            nome="postgres",
            disponivel=True,
            url=url_display,
            detalhes="Conexão OK",
        )
    except Exception as erro:
        return StatusServico(
            nome="postgres",
            disponivel=False,
            url=url_display,
            detalhes=str(erro),
        )


@router.get(
    "/",
    response_model=RespostaHealth,
    summary="Health check completo",
    description=(
        "Verifica a disponibilidade de todos os serviços de infraestrutura "
        "(Ollama, PostgreSQL). Retorna 'ok' quando todos estão disponíveis, "
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
        await _verificar_postgres(config.postgres_dsn),
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
