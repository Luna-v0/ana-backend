"""Registro dos modelos GPU do sistema ANA no GestorGPU.

Define as funções de criação e liberação para cada modelo gerenciado:
- "embeddings"  : intfloat/multilingual-e5-large  (~2.1 GB VRAM)
- "reranker"    : BAAI/bge-reranker-base           (~0.5 GB VRAM)
- "whisper"     : whisper-large-v3                 (~10.0 GB VRAM)
- "diarizacao"  : pyannote/speaker-diarization-3.1 (~2.0 GB VRAM)

Uso no startup da aplicação:
    from ana.gpu.registro import registrar_modelos
    from ana.gpu import obter_gestor
    registrar_modelos(obter_gestor())
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from ana.gpu.gestor import GestorGPU


# =============================================================================
# Helpers internos
# =============================================================================

def _dispositivo_cuda() -> str:
    """Retorna 'cuda' se GPU disponível, senão 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _limpar_modelo(modelo: Any, atributos: list[str]) -> None:
    """Seta atributos para None e chama gc.collect()."""
    for attr in atributos:
        setattr(modelo, attr, None)
    gc.collect()


# =============================================================================
# Embeddings — intfloat/multilingual-e5-large
# =============================================================================

def _criar_embeddings() -> Any:
    """Cria e carrega o GeradorEmbeddings em VRAM."""
    from ana.rag.embeddings import obter_gerador_embeddings
    gerador = obter_gerador_embeddings()
    gerador._carregar_modelo()  # força carregamento imediato
    return gerador


def _liberar_embeddings(gerador: Any) -> None:
    """Descarrega o modelo de embeddings da VRAM."""
    gerador._modelo = None
    gc.collect()
    logger.debug("embeddings: modelo descarregado da VRAM")


# =============================================================================
# Reranker — BAAI/bge-reranker-base
# =============================================================================

def _criar_reranker() -> Any:
    """Cria e carrega o Reranker CrossEncoder em VRAM."""
    from ana.rag.retrieval import obter_reranker
    reranker = obter_reranker()
    reranker._carregar_modelo()  # força carregamento imediato
    return reranker


def _liberar_reranker(reranker: Any) -> None:
    """Descarrega o modelo de reranking da VRAM."""
    reranker._modelo = None
    gc.collect()
    logger.debug("reranker: modelo descarregado da VRAM")


# =============================================================================
# Whisper — whisper-large-v3 via WhisperX
# =============================================================================

def _criar_whisper() -> Any:
    """Cria e carrega o MotorASR (WhisperX large-v3) em VRAM."""
    from ana.transcricao.transcricao import MotorASR
    dispositivo = _dispositivo_cuda()
    motor = MotorASR(
        modelo_nome="large-v3",
        dispositivo=dispositivo,
        idioma="pt",
        batch_size=16 if dispositivo == "cuda" else 4,
        compute_type="float16" if dispositivo == "cuda" else "int8",
    )
    motor._carregar_modelo()
    return motor


def _liberar_whisper(motor: Any) -> None:
    """Descarrega o MotorASR da VRAM."""
    motor.liberar_modelo()  # já faz gc.collect() + torch.cuda.empty_cache()


# =============================================================================
# Diarização — pyannote/speaker-diarization-3.1
# =============================================================================

def _criar_diarizacao() -> Any:
    """Cria e carrega o DiarizadorSpeaker (pyannote) em VRAM."""
    import os
    from ana.transcricao.diarizacao import DiarizadorSpeaker
    dispositivo = _dispositivo_cuda()
    diarizador = DiarizadorSpeaker(
        dispositivo=dispositivo,
        hf_token=os.environ.get("HF_TOKEN", ""),
    )
    diarizador._carregar_pipeline()
    return diarizador


def _liberar_diarizacao(diarizador: Any) -> None:
    """Descarrega o DiarizadorSpeaker da VRAM."""
    diarizador.liberar_pipeline()  # já faz gc.collect() + torch.cuda.empty_cache()


# =============================================================================
# Registro no GestorGPU
# =============================================================================

def registrar_modelos(gestor: "GestorGPU") -> None:
    """Registra todos os modelos GPU no GestorGPU fornecido.

    Deve ser chamado uma vez no startup da aplicação, após o gestor
    ser criado. Os modelos NÃO são carregados aqui — apenas registrados.
    O carregamento ocorre na primeira requisição de cada modelo.

    Args:
        gestor: Instância do GestorGPU onde registrar os modelos.
    """
    gestor.registrar(
        nome="embeddings",
        carregar=_criar_embeddings,
        descarregar=_liberar_embeddings,
        vram_gb=2.1,
    )
    gestor.registrar(
        nome="reranker",
        carregar=_criar_reranker,
        descarregar=_liberar_reranker,
        vram_gb=0.5,
    )
    gestor.registrar(
        nome="whisper",
        carregar=_criar_whisper,
        descarregar=_liberar_whisper,
        vram_gb=10.0,
    )
    gestor.registrar(
        nome="diarizacao",
        carregar=_criar_diarizacao,
        descarregar=_liberar_diarizacao,
        vram_gb=2.0,
    )
    logger.info(
        "GestorGPU: 4 modelos registrados "
        "(embeddings, reranker, whisper, diarizacao)"
    )
