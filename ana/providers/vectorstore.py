"""Protocol para provedores de armazenamento vetorial."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ana.rag.modelos import ChunkJuridico, FiltrosBusca


@runtime_checkable
class VectorStoreProvider(Protocol):
    """Contrato para indexadores e buscadores vetoriais.

    Conforme: IndexadorQdrant em rag/indexador.py

    Nota: IndexadorQdrant usa `nome_colecao` e `limite` como nomes de parâmetro,
    e `busca_semantica` retorna `list[dict]`. O isinstance(@runtime_checkable)
    verifica apenas a existência dos métodos, portanto conforma em runtime.
    """

    def indexar_chunks(
        self, chunks: list[ChunkJuridico], colecao: str | None = None
    ) -> int: ...

    def busca_semantica(
        self,
        vetor_query: list[float],
        filtros: FiltrosBusca,
        top_k: int,
        colecao: str | None = None,
    ) -> list[Any]: ...

    def verificar_conexao(self) -> bool: ...

    def listar_colecoes(self) -> list[str]: ...
