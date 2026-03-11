"""Protocol para provedores de embeddings semânticos."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contrato para geradores de embeddings.

    Conforme: GeradorEmbeddings em rag/embeddings.py
    """

    def gerar(self, texto: str) -> list[float]: ...

    def gerar_batch(self, textos: list[str]) -> list[list[float]]: ...

    def gerar_query(self, query: str) -> list[float]: ...
