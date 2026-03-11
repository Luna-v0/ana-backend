"""Módulo de agentes LangGraph do sistema ANA.

Implementa a orquestração de agentes especializados via LangGraph,
conforme descrito no spec 09.

Submodules:
    estado: TypedDict EstadoJuridico compartilhado entre agentes.
    orquestrador: Grafo principal de orquestração LangGraph.
    pesquisador: Agente Pesquisador Legal (RAG + síntese).
    prompts: Templates de prompt para cada agente.
"""
