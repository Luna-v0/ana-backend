"""Abstrações via typing.Protocol para os componentes do sistema ANA.

Cada Protocol define o contrato de um componente sem acoplar a implementações
concretas. As classes existentes conformam via duck typing (structural subtyping)
— não precisam herdar dos Protocols.

Exports:
    LLMProvider: Contrato para provedores de LLM.
    OllamaLLMProvider: Implementação Ollama via REST API.
    ASRProvider: Contrato para motores ASR.
    DiarizacaoProvider: Contrato para diarizadores de falantes.
    EmbeddingProvider: Contrato para geradores de embeddings.
    RerankerProvider: Contrato para rerankers.
    VectorStoreProvider: Contrato para indexadores vetoriais.
"""

from .asr import ASRProvider
from .diarizacao import DiarizacaoProvider
from .embeddings import EmbeddingProvider
from .llm import LLMProvider, OllamaLLMProvider
from .reranker import RerankerProvider
from .vectorstore import VectorStoreProvider

__all__ = [
    "LLMProvider",
    "OllamaLLMProvider",
    "ASRProvider",
    "DiarizacaoProvider",
    "EmbeddingProvider",
    "RerankerProvider",
    "VectorStoreProvider",
]
