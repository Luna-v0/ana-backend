"""Pacote de agentes LangGraph do sistema ANA (Spec 10).

Implementa o grafo de orquestração jurídica com nós especializados:
- ``nos/classificar.py``: Orquestrador — classifica intenção do usuário
- ``nos/pesquisar.py``: Pesquisador — busca RAG na legislação
- ``nos/analisar.py``: Analista — análise de documentos do processo
- ``nos/resposta.py``: Sintetizador — gera resposta final e valida leis
- ``grafo.py``: StateGraph compilado com AsyncSqliteSaver
- ``prompts.py``: Templates de prompt por agente
- ``estado.py``: EstadoJuridico TypedDict compartilhado

Uso principal via ``POST /chat`` no router de chat.
"""
