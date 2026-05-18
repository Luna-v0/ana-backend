"""Definição do estado compartilhado entre agentes do sistema ANA.

O EstadoJuridico é o TypedDict passado entre todos os nós do grafo
LangGraph, conforme definido no spec 09.
"""

from typing import Optional, TypedDict


class EstadoJuridico(TypedDict, total=False):
    """Estado compartilhado entre todos os agentes do grafo LangGraph.

    Cada campo pode ser None quando ainda não foi preenchido pelo agente
    correspondente. Os campos 'mensagem_usuario' e 'sessao_id' são obrigatórios
    na entrada do grafo.

    Attributes:
        mensagem_usuario: Mensagem original enviada pelo usuário.
        sessao_id: ID da sessão de processo ativa (para RAG isolado).
        transcricao_anexada: Texto de transcrição de audiência anexado ao chat
            pelo usuário via frontend. Usado como contexto adicional pelos nós
            de pesquisa e análise.
        intencao: Intenção classificada pelo Orquestrador.
        contexto_rag: Lista de chunks recuperados pelo RAG.
        prompt_sintese: Prompt bruto de síntese (pré-reformulação).
        prompt_reformulacao: Prompt enriquecido com instruções de reformulação
            em português jurídico formal. Criado pelo nó reformular.
        rascunho: Resposta gerada antes da reformulação final.
        resposta: Resposta final gerada (pós-reformulação).
        descricao: Descrição curta (1–2 frases) do conteúdo respondido,
            usada como título do chat no frontend.
        validacao: Resultado da validação de leis pelo Validador.
        documentos_gerados: Lista de caminhos de documentos Word gerados.
        erro: Mensagem de erro caso ocorra falha em algum nó.
    """

    mensagem_usuario: str
    sessao_id: str
    transcricao_anexada: Optional[str]
    documento_processo: Optional[str]
    intencao: Optional[str]
    contexto_rag: Optional[list[dict]]
    prompt_sintese: Optional[str]
    prompt_reformulacao: Optional[str]
    rascunho: Optional[str]
    resposta: Optional[str]
    descricao: Optional[str]
    validacao: Optional[dict]
    documentos_gerados: Optional[list[str]]
    erro: Optional[str]


# Intenções reconhecidas pelo Orquestrador
INTENCOES_VALIDAS = frozenset({
    "pesquisa_legal",       # Busca em legislação, súmulas, jurisprudência
    "analise_processo",     # Perguntas sobre o processo/sessão atual
    "gerar_documento",      # Criar/editar peça processual Word
    "transcrever_audio",    # Transcrição de audiência
    "verificar_prazo",      # Consulta sobre prazos processuais
    "buscar_similar",       # Busca de processos similares
    "desconhecida",         # Intenção não reconhecida — resposta genérica
})
