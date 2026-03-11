"""Modelos de dados para o sistema de sessões de processos jurídicos."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sessao:
    """Representa uma sessão de processo jurídico.

    Attributes:
        id: Identificador único no formato 'sess_XXXXXXXX'.
        numero_processo: Número CNJ do processo (ex: '0001234-12.2024.8.26.0100').
        tipo_acao: Tipo da ação judicial (ex: 'Ação de Indenização').
        area: Área jurídica principal.
        vara: Vara ou câmara responsável.
        cidade_uf: Cidade e UF do foro.
        status: Status atual ('em_andamento', 'encerrada', 'suspensa').
        criado_em: ISO8601 da criação.
        atualizado_em: ISO8601 da última atualização.
        partes: Dict com partes do processo (autor, réu, etc.).
        prazos: Lista de prazos processuais.
    """

    id: str
    numero_processo: str
    tipo_acao: str
    area: str = "civil"
    vara: str = ""
    cidade_uf: str = ""
    status: str = "em_andamento"
    criado_em: str = ""
    atualizado_em: str = ""
    partes: dict[str, Any] = field(default_factory=dict)
    prazos: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentoSessao:
    """Representa um documento indexado em uma sessão.

    Attributes:
        id: Identificador único do documento.
        sessao_id: ID da sessão a que pertence.
        nome: Nome original do arquivo.
        tipo: Tipo do arquivo ('pdf', 'docx', 'txt').
        tamanho_bytes: Tamanho do arquivo em bytes.
        chunks_indexados: Quantidade de chunks vetorizados.
        criado_em: ISO8601 da indexação.
    """

    id: str
    sessao_id: str
    nome: str
    tipo: str
    tamanho_bytes: int
    chunks_indexados: int
    criado_em: str
