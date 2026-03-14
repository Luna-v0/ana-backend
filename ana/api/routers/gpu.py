"""Router FastAPI para diagnóstico e controle do GestorGPU.

Endpoints:
    GET  /gpu/status   — Estado atual dos modelos GPU (carregados, timers, VRAM)
    POST /gpu/liberar  — Descarrega um modelo específico sob demanda
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from ana.gpu import obter_gestor

router = APIRouter(prefix="/gpu", tags=["GPU"])


@router.get(
    "/status",
    summary="Status dos modelos GPU",
    description=(
        "Retorna o estado atual do GestorGPU: quais modelos estão carregados em VRAM, "
        "timers de idle ativos, VRAM estimada por modelo e configuração de timeout."
    ),
)
async def status_gpu() -> dict:
    """Retorna o estado atual do GestorGPU."""
    gestor = obter_gestor()
    return gestor.status()


@router.post(
    "/liberar/{nome}",
    summary="Descarregar modelo GPU sob demanda",
    description=(
        "Força o descarregamento imediato de um modelo da VRAM, "
        "limpando o cache CUDA. Útil antes de operações que exigem toda a VRAM disponível."
    ),
)
async def liberar_modelo(nome: str) -> dict:
    """Descarrega um modelo específico da VRAM.

    Args:
        nome: Nome do modelo a descarregar (ex: "embeddings", "reranker", "whisper").

    Returns:
        Confirmação com nome do modelo descarregado.

    Raises:
        HTTPException 404: Se o modelo não estiver registrado no GestorGPU.
    """
    gestor = obter_gestor()
    if nome not in gestor._registros:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{nome}' não registrado. "
                f"Disponíveis: {list(gestor._registros)}"
            ),
        )

    gestor._cancelar_timer(nome)
    async with gestor._gpu_lock:
        await gestor._descarregar(nome)

    logger.info(f"GestorGPU: '{nome}' descarregado via endpoint /gpu/liberar")
    return {"descarregado": nome, "mensagem": f"Modelo '{nome}' liberado da VRAM"}
