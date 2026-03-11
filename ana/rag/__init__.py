"""Módulo RAG (Retrieval-Augmented Generation) do sistema ANA.

Implementa o pipeline de busca híbrida (semântica + BM25) sobre a
legislação brasileira conforme descrito no spec 02.

Submodules:
    modelos: Estruturas de dados para chunks e resultados de busca.
    ingestao: Pipeline de extração, chunking e indexação de documentos.
    embeddings: Geração de embeddings com multilingual-e5-large.
    indexador: Indexação no Qdrant com metadata jurídica.
    retrieval: Busca híbrida, reranking e MMR.

Nota (LGPD):
    Todo processamento é local. Documentos de processos nunca são
    enviados para serviços externos.
"""
