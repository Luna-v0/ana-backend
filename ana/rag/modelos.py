"""Modelos de dados para o pipeline RAG do sistema ANA.

Define as estruturas de dados para chunks jurídicos, metadata
e resultados de busca, conforme o schema do spec 02.
"""

from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TipoDocumento(str, Enum):
    """Tipos de documento suportados pelo sistema RAG.

    Attributes:
        LEI_FEDERAL: Legislação federal (leis, decretos, etc.).
        LEI_ESTADUAL: Legislação estadual.
        LEI_MUNICIPAL: Legislação municipal.
        SUMULA: Súmulas do STF, STJ, TST e tribunais regionais.
        JURISPRUDENCIA: Decisões de tribunais superiores e regionais.
        DOCUMENTO_USUARIO: Documentos do usuário (contratos, peças, etc.).
    """

    LEI_FEDERAL = "lei_federal"
    LEI_ESTADUAL = "lei_estadual"
    LEI_MUNICIPAL = "lei_municipal"
    SUMULA = "sumula"
    JURISPRUDENCIA = "jurisprudencia"
    DOCUMENTO_USUARIO = "documento_usuario"


class AreaJuridica(str, Enum):
    """Áreas do direito suportadas para filtragem no RAG.

    Attributes:
        CIVIL: Direito civil.
        PENAL: Direito penal.
        TRABALHISTA: Direito do trabalho.
        TRIBUTARIO: Direito tributário.
        CONSUMIDOR: Direito do consumidor.
        DADOS: Proteção de dados (LGPD).
        ADMINISTRATIVO: Direito administrativo.
        CONSTITUCIONAL: Direito constitucional.
        PROCESSUAL_CIVIL: Processo civil.
        PROCESSUAL_PENAL: Processo penal.
    """

    CIVIL = "civil"
    PENAL = "penal"
    TRABALHISTA = "trabalhista"
    TRIBUTARIO = "tributario"
    CONSUMIDOR = "consumidor"
    DADOS = "dados"
    ADMINISTRATIVO = "administrativo"
    CONSTITUCIONAL = "constitucional"
    PROCESSUAL_CIVIL = "processual_civil"
    PROCESSUAL_PENAL = "processual_penal"


class VigenciaStatus(str, Enum):
    """Status de vigência de um documento legal.

    Attributes:
        ATIVA: Legislação em vigor.
        REVOGADA: Legislação completamente revogada.
        PARCIALMENTE_REVOGADA: Parte da legislação foi revogada.
    """

    ATIVA = "ativa"
    REVOGADA = "revogada"
    PARCIALMENTE_REVOGADA = "parcialmente_revogada"


class MetadataChunkJuridico(BaseModel):
    """Metadata rica de um chunk jurídico para filtragem no Qdrant.

    Armazena toda a hierarquia legal e informações de vigência
    para permitir filtros precisos durante a busca.

    Attributes:
        fonte: Identificação da lei ou fonte (ex: 'Lei 13.709/2018 (LGPD)').
        tipo: Tipo do documento jurídico.
        area: Área do direito principal.
        titulo: Título da estrutura hierárquica (quando aplicável).
        capitulo: Capítulo da estrutura hierárquica (quando aplicável).
        secao: Seção da estrutura hierárquica (quando aplicável).
        artigo: Artigo específico (ex: 'Art. 5').
        data_publicacao: Data de publicação da norma.
        vigencia: Status atual de vigência.
        orgao: Órgão emissor (Congresso, STF, STJ, TST, etc.).
        url_origem: URL da fonte original (planalto, lexml, etc.).
        sessao_id: ID da sessão de processo (apenas para documentos de usuário).
    """

    fonte: str = Field(description="Identificação da lei ou fonte")
    tipo: TipoDocumento = Field(description="Tipo do documento jurídico")
    area: Optional[AreaJuridica] = Field(
        default=None,
        description="Área do direito principal",
    )
    titulo: Optional[str] = Field(
        default=None,
        description="Título na hierarquia legal",
    )
    capitulo: Optional[str] = Field(
        default=None,
        description="Capítulo na hierarquia legal",
    )
    secao: Optional[str] = Field(
        default=None,
        description="Seção na hierarquia legal",
    )
    artigo: Optional[str] = Field(
        default=None,
        description="Artigo específico (ex: 'Art. 5')",
    )
    data_publicacao: Optional[date] = Field(
        default=None,
        description="Data de publicação da norma",
    )
    vigencia: VigenciaStatus = Field(
        default=VigenciaStatus.ATIVA,
        description="Status atual de vigência",
    )
    orgao: Optional[str] = Field(
        default=None,
        description="Órgão emissor da norma",
    )
    url_origem: Optional[str] = Field(
        default=None,
        description="URL da fonte original para referência",
    )
    sessao_id: Optional[str] = Field(
        default=None,
        description="ID da sessão (apenas para documentos de processo)",
    )


class ChunkJuridico(BaseModel):
    """Representa um chunk de legislação pronto para indexação no Qdrant.

    A unidade base de chunking é o artigo de lei, respeitando a
    estrutura hierárquica: Lei → Título → Capítulo → Seção → Artigo.

    Attributes:
        id: Identificador único do chunk (gerado durante indexação).
        texto: Conteúdo textual do chunk (artigo ou trecho).
        metadata: Metadata jurídica completa para filtragem.
        embedding: Vetor de embedding gerado pelo multilingual-e5-large.
    """

    id: Optional[str] = Field(
        default=None,
        description="ID único gerado pelo indexador",
    )
    texto: str = Field(description="Conteúdo textual do chunk jurídico")
    metadata: MetadataChunkJuridico = Field(
        description="Metadata jurídica completa",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vetor de embedding (1024 dimensões para e5-large)",
    )


class ResultadoBusca(BaseModel):
    """Resultado de uma busca no sistema RAG.

    Attributes:
        chunk: Chunk recuperado com seu conteúdo e metadata.
        score: Score de relevância combinado (semântico + BM25 via RRF).
        rank_semantico: Posição no ranking semântico original.
        rank_bm25: Posição no ranking BM25 original.
    """

    chunk: ChunkJuridico = Field(description="Chunk recuperado")
    score: float = Field(description="Score de relevância combinado (RRF)")
    rank_semantico: Optional[int] = Field(
        default=None,
        description="Posição no ranking semântico",
    )
    rank_bm25: Optional[int] = Field(
        default=None,
        description="Posição no ranking BM25",
    )


class FiltrosBusca(BaseModel):
    """Filtros para refinar a busca no Qdrant.

    Permite ao agente pesquisador filtrar por tipo, área, vigência
    e outros critérios para buscas mais precisas.

    Attributes:
        tipos: Lista de tipos de documento a incluir.
        areas: Lista de áreas jurídicas a incluir.
        vigencia: Filtrar por status de vigência.
        orgaos: Lista de órgãos emissores a incluir.
        sessao_id: Restringir busca a uma sessão específica.
    """

    tipos: Optional[list[TipoDocumento]] = Field(
        default=None,
        description="Filtrar por tipo de documento",
    )
    areas: Optional[list[AreaJuridica]] = Field(
        default=None,
        description="Filtrar por área jurídica",
    )
    vigencia: Optional[VigenciaStatus] = Field(
        default=VigenciaStatus.ATIVA,
        description="Filtrar por vigência (padrão: apenas ativos)",
    )
    orgaos: Optional[list[str]] = Field(
        default=None,
        description="Filtrar por órgão emissor",
    )
    sessao_id: Optional[str] = Field(
        default=None,
        description="Restringir busca a uma sessão específica de processo",
    )
