"""Router FastAPI para o módulo de scrapers jurídicos.

Expõe endpoints para consultar status das fontes, disparar coletas
manuais e monitorar o agendador automático.

Endpoints:
    GET  /scrapers/status        — Status do pipeline e dependências
    GET  /scrapers/fontes        — Lista fontes e última coleta de cada
    POST /scrapers/coletar       — Dispara coleta completa de uma fonte
    POST /scrapers/atualizar     — Atualização incremental de uma fonte
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/scrapers", tags=["scrapers"])

# Pipeline singleton (criado lazy na primeira requisição)
_pipeline = None


def _obter_pipeline():
    """Retorna o pipeline de scrapers, criando-o se necessário."""
    global _pipeline
    if _pipeline is None:
        from ana.scrapers.pipeline import PipelineScrapers
        _pipeline = PipelineScrapers()
    return _pipeline


class RequisicaoColeta(BaseModel):
    """Payload para disparar coleta de uma fonte.

    Attributes:
        fonte: Nome da fonte a coletar ('planalto', 'lexml', 'stf', 'stj').
    """
    fonte: str


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


@router.get("/status")
async def status_scrapers() -> dict:
    """Retorna status do pipeline de scrapers e disponibilidade de dependências.

    Returns:
        Dicionário com status de cada fonte e flag de dependências instaladas.
    """
    try:
        pipeline = _obter_pipeline()
        return pipeline.status()
    except Exception as e:
        logger.error(f"Erro ao obter status dos scrapers: {e}")
        return {
            "fontes": {},
            "total_documentos_cache": 0,
            "dependencias_instaladas": False,
            "erro": str(e),
        }


@router.get("/fontes")
async def listar_fontes() -> dict:
    """Lista as fontes disponíveis e informações básicas de cada uma.

    Returns:
        Dicionário com configuração e status de cada fonte.
    """
    from ana.scrapers.agendador import INTERVALOS_HORAS

    fontes = {
        "planalto": {
            "descricao": "Legislação federal compilada (planalto.gov.br)",
            "tipo_documento": "lei_federal",
            "intervalo_horas": INTERVALOS_HORAS["planalto"],
        },
        "lexml": {
            "descricao": "Legislação estruturada em XML (Senado Federal / lexml.gov.br)",
            "tipo_documento": "lei_federal",
            "intervalo_horas": INTERVALOS_HORAS["lexml"],
        },
        "stf": {
            "descricao": "Súmulas ordinárias e vinculantes do STF",
            "tipo_documento": "sumula",
            "intervalo_horas": INTERVALOS_HORAS["stf"],
        },
        "stj": {
            "descricao": "Súmulas do Superior Tribunal de Justiça",
            "tipo_documento": "sumula",
            "intervalo_horas": INTERVALOS_HORAS["stj"],
        },
    }

    try:
        pipeline = _obter_pipeline()
        status = pipeline.status()
        for nome, info in fontes.items():
            info.update(status["fontes"].get(nome, {}))
    except Exception:
        pass

    return {"fontes": fontes}


def _executar_coleta_bg(fonte: str) -> None:
    """Executa coleta em background (chamado via BackgroundTasks)."""
    try:
        pipeline = _obter_pipeline()
        resultado = pipeline.coletar_fonte(fonte)
        logger.info(
            f"Coleta background '{fonte}' concluída: "
            f"{resultado.documentos_novos} novos"
        )
    except Exception as e:
        logger.error(f"Erro na coleta background de '{fonte}': {e}")


@router.post("/coletar", response_model=RespostaColeta)
async def coletar_fonte(
    req: RequisicaoColeta,
    background_tasks: BackgroundTasks,
) -> RespostaColeta:
    """Dispara coleta completa de uma fonte específica.

    A coleta é executada em background para não bloquear a resposta HTTP.
    Consulte /scrapers/status para acompanhar o progresso.

    Args:
        req: Fonte a coletar.
        background_tasks: Executor de tarefas em background do FastAPI.

    Returns:
        Confirmação de início da coleta (a coleta ocorre em background).

    Raises:
        HTTPException 422: Se o nome da fonte for inválido.
    """
    fontes_validas = {"planalto", "lexml", "stf", "stj"}
    if req.fonte not in fontes_validas:
        raise HTTPException(
            status_code=422,
            detail=f"Fonte '{req.fonte}' inválida. Use: {sorted(fontes_validas)}",
        )

    background_tasks.add_task(_executar_coleta_bg, req.fonte)
    logger.info(f"Coleta de '{req.fonte}' agendada em background")

    return RespostaColeta(
        fonte=req.fonte,
        documentos_novos=0,
        documentos_ignorados=0,
        erros=[],
        duracao_segundos=0.0,
    )


@router.post("/atualizar-tudo")
async def atualizar_tudo(background_tasks: BackgroundTasks) -> dict:
    """Dispara atualização incremental de todas as fontes em background.

    Coleta apenas documentos publicados após a última coleta de cada fonte.

    Returns:
        Confirmação de início das atualizações.
    """
    fontes = ["planalto", "lexml", "stf", "stj"]
    for fonte in fontes:
        background_tasks.add_task(_executar_coleta_bg, fonte)

    logger.info("Atualização incremental de todas as fontes agendada em background")
    return {
        "mensagem": "Atualização iniciada em background para todas as fontes",
        "fontes": fontes,
    }
