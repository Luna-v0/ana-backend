"""Protocol para provedores de reranking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RerankerProvider(Protocol):
    """Contrato para rerankers de resultados de busca.

    Conforme: Reranker em rag/retrieval.py
    """

    def rerankar(
        self,
        query: str,
        candidatos: list[tuple[str, str]],
        top_n: int | None = None,
    ) -> list[tuple[str, float]]: ...
