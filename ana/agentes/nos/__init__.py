"""Pacote de nós individuais do grafo LangGraph do sistema ANA.

Cada módulo implementa um nó específico do StateGraph:
- ``classificar``  — Orquestrador: classifica intenção do usuário
- ``pesquisar``    — Pesquisador: busca RAG na legislação
- ``analisar``     — Analista: busca documentos do processo
- ``validar``      — Validador: verifica fontes dos chunks RAG
- ``reformular``   — Redator: enriquece prompt com instruções jurídicas formais
- ``resposta``     — Sintetizador: prepara prompt final ou gera resposta direta
"""
