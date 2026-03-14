"""Gerenciador centralizado do ciclo de vida de modelos GPU.

Responsabilidades:
- Carregamento sob demanda (lazy) de modelos em VRAM
- Fila de requisições por modelo (asyncio.Lock por slot)
- Descarregamento automático após idle timeout com limpeza do cache CUDA
- Troca de modelos com callback para workflows agênticos (LLM ↔ Whisper)

Exemplo básico:
    gestor = GestorGPU()
    gestor.registrar("embeddings", criar_fn, liberar_fn, vram_gb=2.1)

    async with gestor.usar("embeddings") as modelo:
        vetores = modelo.gerar_batch(textos)

Exemplo agêntico (troca LLM ↔ Whisper):
    resultado = await gestor.trocar_callback(
        liberar=["embeddings", "reranker"],
        carregar="whisper",
        callback=lambda motor: motor.transcrever(caminho_audio),
    )
"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# =============================================================================
# Utilidades CUDA
# =============================================================================

def _limpar_cache_cuda() -> None:
    """Limpa o cache do alocador CUDA e força garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# =============================================================================
# Estruturas internas
# =============================================================================

@dataclass
class _RegistroModelo:
    """Metadados e factories de um modelo gerenciado pelo GestorGPU."""
    nome: str
    carregar: Callable[[], Any]       # síncrona: cria e carrega o modelo em VRAM
    descarregar: Callable[[Any], None]  # síncrona: libera VRAM do modelo
    vram_gb: float = 0.0


@dataclass
class _SlotModelo:
    """Representa um modelo atualmente carregado."""
    instancia: Any
    ultimo_uso: float = field(default_factory=time.monotonic)


# =============================================================================
# GestorGPU
# =============================================================================

class GestorGPU:
    """Gerenciador centralizado do ciclo de vida de modelos GPU do backend ANA.

    Gerencia quais modelos estão carregados em VRAM, enfilera requisições
    concorrentes por modelo e descarrega modelos ociosos para liberar memória.

    Suporta troca com callback para workflows agênticos onde um modelo precisa
    ceder VRAM temporariamente para outro (ex: embeddings → Whisper → embeddings).

    Attributes:
        timeout_idle_s: Segundos de inatividade antes de descarregar o modelo.
    """

    def __init__(self, timeout_idle_s: float = 300.0) -> None:
        """Inicializa o gestor vazio (modelos devem ser registrados separadamente).

        Args:
            timeout_idle_s: Timeout de idle em segundos. Padrão: 5 minutos.
        """
        self._registros: dict[str, _RegistroModelo] = {}
        self._slots: dict[str, _SlotModelo] = {}       # modelos carregados
        self._locks: dict[str, asyncio.Lock] = {}      # fila por modelo
        self._gpu_lock = asyncio.Lock()                # serializa load/unload
        self._timers: dict[str, asyncio.Task] = {}     # timers de idle
        self.timeout_idle_s = timeout_idle_s

    # ─────────────────────────────── Registro ────────────────────────────────

    def registrar(
        self,
        nome: str,
        carregar: Callable[[], Any],
        descarregar: Callable[[Any], None],
        vram_gb: float = 0.0,
    ) -> None:
        """Registra um modelo para gerenciamento pelo GestorGPU.

        Args:
            nome: Identificador único do modelo (ex: "embeddings", "whisper").
            carregar: Função síncrona que cria/carrega o modelo em VRAM.
                Chamada dentro de asyncio.to_thread para não bloquear o loop.
            descarregar: Função síncrona que libera o modelo da VRAM.
                Recebe a instância retornada por `carregar`.
            vram_gb: Estimativa de VRAM consumida (informativo).
        """
        self._registros[nome] = _RegistroModelo(nome, carregar, descarregar, vram_gb)
        self._locks[nome] = asyncio.Lock()
        logger.debug(f"GestorGPU: modelo '{nome}' registrado (~{vram_gb:.1f} GB VRAM)")

    # ──────────────────────────── Load / Unload ──────────────────────────────

    async def _carregar(self, nome: str) -> Any:
        """Carrega modelo se não estiver em memória.

        Deve ser chamado com ``_gpu_lock`` já adquirido.

        Args:
            nome: Nome do modelo registrado.

        Returns:
            Instância do modelo carregado.
        """
        if nome in self._slots:
            return self._slots[nome].instancia

        registro = self._registros[nome]
        logger.info(f"GestorGPU: carregando '{nome}' (~{registro.vram_gb:.1f} GB VRAM)")
        instancia = await asyncio.to_thread(registro.carregar)
        self._slots[nome] = _SlotModelo(instancia=instancia)
        logger.info(f"GestorGPU: '{nome}' carregado com sucesso")
        return instancia

    async def _descarregar(self, nome: str) -> None:
        """Descarrega modelo e limpa cache CUDA.

        Deve ser chamado com ``_gpu_lock`` já adquirido.
        No-op se o modelo não estiver carregado.

        Args:
            nome: Nome do modelo a descarregar.
        """
        if nome not in self._slots:
            return

        registro = self._registros[nome]
        slot = self._slots.pop(nome)
        logger.info(f"GestorGPU: descarregando '{nome}'")
        await asyncio.to_thread(registro.descarregar, slot.instancia)
        await asyncio.to_thread(_limpar_cache_cuda)
        logger.info(f"GestorGPU: '{nome}' descarregado — cache CUDA limpo")

    # ───────────────────────────── Timers idle ───────────────────────────────

    def _cancelar_timer(self, nome: str) -> None:
        """Cancela o timer de idle de um modelo se existir."""
        timer = self._timers.pop(nome, None)
        if timer and not timer.done():
            timer.cancel()

    def _agendar_idle(self, nome: str) -> None:
        """Agenda descarregamento automático após ``timeout_idle_s`` segundos."""
        self._cancelar_timer(nome)

        async def _expirar() -> None:
            try:
                await asyncio.sleep(self.timeout_idle_s)
                logger.info(
                    f"GestorGPU: '{nome}' ocioso por {self.timeout_idle_s:.0f}s — descarregando"
                )
                async with self._gpu_lock:
                    await self._descarregar(nome)
            except asyncio.CancelledError:
                pass  # timer cancelado por novo uso ou descarregamento explícito

        self._timers[nome] = asyncio.ensure_future(_expirar())

    # ─────────────────────────── Context manager ─────────────────────────────

    @asynccontextmanager
    async def usar(self, nome: str) -> AsyncGenerator[Any, None]:
        """Adquire o modelo, carregando-o se necessário. Enfilera se ocupado.

        Cancela o timer de idle enquanto o modelo está em uso.
        Reagenda o timer de idle ao liberar.

        Args:
            nome: Nome do modelo registrado.

        Yields:
            Instância do modelo pronta para uso.

        Raises:
            KeyError: Se o modelo não estiver registrado.

        Example:
            async with gestor.usar("embeddings") as gerador:
                vetores = await asyncio.to_thread(gerador.gerar_batch, textos)
        """
        if nome not in self._registros:
            disponiveis = list(self._registros)
            raise KeyError(
                f"GestorGPU: modelo '{nome}' não registrado. "
                f"Disponíveis: {disponiveis}"
            )

        async with self._locks[nome]:
            self._cancelar_timer(nome)
            async with self._gpu_lock:
                instancia = await self._carregar(nome)
            try:
                yield instancia
            finally:
                if nome in self._slots:
                    self._slots[nome].ultimo_uso = time.monotonic()
                self._agendar_idle(nome)

    # ─────────────────────────── Troca agêntica ──────────────────────────────

    async def trocar_callback(
        self,
        liberar: list[str],
        carregar: str,
        callback: Callable[[Any], Any],
    ) -> Any:
        """Troca de contexto GPU para workflows agênticos.

        Descarrega os modelos em ``liberar``, carrega ``carregar``,
        executa ``callback(modelo)`` em thread separada, e descarrega
        ``carregar`` imediatamente após (sem agendar idle timer).

        Os modelos liberados voltam a ser carregados sob demanda na
        próxima requisição.

        Adquire os locks de todos os modelos envolvidos em ordem
        consistente para evitar deadlock.

        Args:
            liberar: Nomes dos modelos a descarregar antes do callback.
            carregar: Nome do modelo a carregar para o callback.
            callback: Função **síncrona** recebendo a instância do modelo
                carregado e retornando o resultado desejado.

        Returns:
            Resultado do callback.

        Example:
            # Dentro de um workflow agêntico: LLM cede VRAM para Whisper
            transcript = await gestor.trocar_callback(
                liberar=["embeddings", "reranker"],
                carregar="whisper",
                callback=lambda motor: motor.transcrever(caminho_audio),
            )
        """
        # Ordena nomes para adquirir locks em ordem determinística (evita deadlock)
        todos_nomes = sorted(set(liberar + [carregar]))
        todos_nomes = [n for n in todos_nomes if n in self._locks]
        locks_ordenados = [self._locks[n] for n in todos_nomes]

        # Adquire todos os locks antes de qualquer operação de memória
        for lock in locks_ordenados:
            await lock.acquire()

        try:
            async with self._gpu_lock:
                for nome in liberar:
                    self._cancelar_timer(nome)
                    await self._descarregar(nome)
                instancia = await self._carregar(carregar)

            try:
                resultado = await asyncio.to_thread(callback, instancia)
            finally:
                # Descarrega imediatamente — sem idle timer
                async with self._gpu_lock:
                    self._cancelar_timer(carregar)
                    await self._descarregar(carregar)

        finally:
            for lock in reversed(locks_ordenados):
                lock.release()

        return resultado

    # ──────────────────────────────── Status ─────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Retorna estado atual do gestor para diagnóstico e monitoramento.

        Returns:
            Dicionário com modelos carregados, registrados, timers ativos
            e configuração de timeout.
        """
        return {
            "modelos_carregados": {
                nome: {
                    "vram_gb_estimado": self._registros[nome].vram_gb,
                    "timer_idle_ativo": (
                        nome in self._timers and not self._timers[nome].done()
                    ),
                    "ultimo_uso_s_atras": round(
                        time.monotonic() - self._slots[nome].ultimo_uso, 1
                    ),
                }
                for nome in self._slots
                if nome in self._registros
            },
            "modelos_registrados": {
                nome: {"vram_gb_estimado": reg.vram_gb}
                for nome, reg in self._registros.items()
            },
            "timeout_idle_segundos": self.timeout_idle_s,
        }

    async def descarregar_tudo(self) -> None:
        """Descarrega todos os modelos ativos.

        Usado no shutdown da aplicação para liberar VRAM ordenadamente.
        """
        for nome in list(self._timers):
            self._cancelar_timer(nome)
        async with self._gpu_lock:
            for nome in list(self._slots):
                await self._descarregar(nome)
        logger.info("GestorGPU: todos os modelos descarregados")
