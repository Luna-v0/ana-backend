"""Contratos de input/output para o router /sessoes."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Sessão — CRUD
# =============================================================================

class RequisicaoCriarSessao(BaseModel):
    """Payload para criação de uma nova sessão.

    Attributes:
        numero_processo: Número CNJ do processo.
        tipo_acao: Tipo da ação judicial.
        area: Área jurídica.
        vara: Vara responsável.
        cidade_uf: Cidade e UF do foro.
        partes: Partes do processo (autor, réu, etc.).
        prazos: Prazos processuais.
    """

    numero_processo: str = Field(description="Número CNJ do processo")
    tipo_acao: str = Field(description="Tipo da ação judicial")
    area: str = Field(default="civil", description="Área jurídica")
    vara: str = Field(default="", description="Vara responsável")
    cidade_uf: str = Field(default="", description="Cidade e UF do foro")
    partes: dict[str, Any] = Field(
        default_factory=dict,
        description="Partes do processo (autor, réu, etc.)",
    )
    prazos: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prazos processuais",
    )


class RequisicaoAtualizarSessao(BaseModel):
    """Campos atualizáveis de uma sessão (todos opcionais).

    Attributes:
        numero_processo: Novo número do processo.
        tipo_acao: Novo tipo de ação.
        area: Nova área jurídica.
        vara: Nova vara.
        cidade_uf: Nova cidade/UF.
        status: Novo status ('ativo', 'arquivado', etc.).
        partes: Novo mapeamento de partes.
        prazos: Nova lista de prazos.
    """

    numero_processo: str | None = None
    tipo_acao: str | None = None
    area: str | None = None
    vara: str | None = None
    cidade_uf: str | None = None
    status: str | None = None
    partes: dict[str, Any] | None = None
    prazos: list[dict[str, Any]] | None = None


class RespostaSessao(BaseModel):
    """Representação completa de uma sessão.

    Attributes:
        id: Identificador único da sessão (prefixo 'sess_').
        numero_processo: Número CNJ do processo.
        tipo_acao: Tipo da ação judicial.
        area: Área jurídica.
        vara: Vara responsável.
        cidade_uf: Cidade e UF do foro.
        status: Status atual da sessão.
        criado_em: Timestamp ISO de criação.
        atualizado_em: Timestamp ISO da última atualização.
        partes: Partes do processo.
        prazos: Prazos processuais.
    """

    id: str
    numero_processo: str
    tipo_acao: str
    area: str
    vara: str
    cidade_uf: str
    status: str
    criado_em: str
    atualizado_em: str
    partes: dict[str, Any]
    prazos: list[dict[str, Any]]


# =============================================================================
# Documentos
# =============================================================================

class RespostaDocumento(BaseModel):
    """Representação de um documento indexado em uma sessão.

    Attributes:
        id: Identificador único do documento.
        sessao_id: ID da sessão à qual pertence.
        nome: Nome original do arquivo.
        tipo: Extensão/tipo do arquivo (.pdf, .docx, .txt, .md).
        tamanho_bytes: Tamanho do arquivo em bytes.
        chunks_indexados: Quantidade de chunks gerados e indexados.
        criado_em: Timestamp ISO de upload.
    """

    id: str
    sessao_id: str
    nome: str
    tipo: str
    tamanho_bytes: int
    chunks_indexados: int
    criado_em: str


# =============================================================================
# Busca intra-sessão
# =============================================================================

class RequisicaoBuscaSessao(BaseModel):
    """Payload para busca vetorial em uma sessão.

    Attributes:
        query: Texto da busca.
        modo: Escopo da busca.
        top_k: Número máximo de resultados.
    """

    query: str = Field(description="Texto da busca")
    modo: Literal["intra", "global", "cross"] = Field(
        default="intra",
        description=(
            "Modo de busca: "
            "'intra' (só esta sessão), "
            "'global' (legislação + sessão), "
            "'cross' (outras sessões)"
        ),
    )
    top_k: int = Field(default=10, ge=1, le=50)


class ResultadoBuscaSessao(BaseModel):
    """Um resultado de busca vetorial em sessão.

    Attributes:
        id: ID do chunk.
        score: Score de relevância.
        texto: Conteúdo textual do chunk.
        fonte: Identificação da fonte.
        sessao_id: ID da sessão de origem (apenas em modo cross).
        artigo: Artigo referenciado (se aplicável).
    """

    id: str
    score: float
    texto: str
    fonte: str
    sessao_id: str | None = None
    artigo: str | None = None


class RespostaBuscaSessao(BaseModel):
    """Resposta de busca vetorial em sessão.

    Attributes:
        query: Query original.
        modo: Modo de busca utilizado.
        total: Total de resultados retornados.
        resultados: Lista de chunks relevantes.
    """

    query: str
    modo: str
    total: int
    resultados: list[ResultadoBuscaSessao]
