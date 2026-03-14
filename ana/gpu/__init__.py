"""Pacote de gerenciamento de recursos GPU do sistema ANA.

Expõe o GestorGPU e o singleton global obter_gestor().
"""

from __future__ import annotations

from ana.gpu.gestor import GestorGPU

_gestor: GestorGPU | None = None


def obter_gestor() -> GestorGPU:
    """Retorna o singleton global do GestorGPU.

    Criado na primeira chamada e reutilizado por toda a aplicação.
    Os modelos devem ser registrados via ``ana.gpu.registro.registrar_modelos()``
    antes do primeiro uso.

    Returns:
        Instância singleton de GestorGPU.
    """
    global _gestor
    if _gestor is None:
        _gestor = GestorGPU(timeout_idle_s=300.0)
    return _gestor


__all__ = ["GestorGPU", "obter_gestor"]
