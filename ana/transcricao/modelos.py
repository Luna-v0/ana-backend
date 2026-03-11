"""Modelos de dados para transcrição de audiências judiciais.

Define as estruturas usadas em todo o pipeline de transcrição:
participantes, segmentos, metadados e resultado final.

Nota (LGPD):
    Arquivos de áudio de audiências contêm dados pessoais sensíveis.
    Todo o processamento é 100% local — nenhum áudio sai da máquina.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RoleParticipante(str, Enum):
    """Papel processual do participante na audiência judicial."""

    JUIZ = "juiz"
    ADVOGADO_AUTOR = "advogado_autor"
    ADVOGADO_REU = "advogado_reu"
    TESTEMUNHA = "testemunha"
    PERITO = "perito"
    PROMOTOR = "promotor"
    DEFENSOR = "defensor"
    PARTE_AUTORA = "parte_autora"
    PARTE_RE = "parte_re"
    ESCRIVAO = "escrivao"
    DESCONHECIDO = "desconhecido"


# Prefixos de exibição por role
_PREFIXOS_ROLE: dict[RoleParticipante, str] = {
    RoleParticipante.JUIZ: "Juiz(a)",
    RoleParticipante.ADVOGADO_AUTOR: "Adv. Autor(a)",
    RoleParticipante.ADVOGADO_REU: "Adv. Réu(ré)",
    RoleParticipante.TESTEMUNHA: "Testemunha",
    RoleParticipante.PERITO: "Perito(a)",
    RoleParticipante.PROMOTOR: "Promotor(a)",
    RoleParticipante.DEFENSOR: "Defensor(a) Público(a)",
    RoleParticipante.PARTE_AUTORA: "Autor(a)",
    RoleParticipante.PARTE_RE: "Réu(ré)",
    RoleParticipante.ESCRIVAO: "Escrivão/ã",
    RoleParticipante.DESCONHECIDO: "Participante",
}


@dataclass
class ParticipanteAudiencia:
    """Representa um participante identificado da audiência.

    Attributes:
        role: Papel processual do participante.
        nome: Nome completo do participante.
        oab: Número OAB (apenas para advogados).
        speaker_id: Label atribuído pelo diarizador (ex: SPEAKER_00).
        confianca: Score de confiança na identificação (0.0 a 1.0).
    """

    role: RoleParticipante
    nome: str
    oab: Optional[str] = None
    speaker_id: Optional[str] = None
    confianca: float = 1.0

    @property
    def label_formatado(self) -> str:
        """Rótulo formatado para exibição na transcrição."""
        prefixo = _PREFIXOS_ROLE.get(self.role, "Participante")
        label = f"{prefixo} — {self.nome}"
        if self.oab:
            label += f" ({self.oab})"
        return label


@dataclass
class SegmentoTranscricao:
    """Segmento individual da transcrição com metadados de tempo e locutor.

    Attributes:
        inicio: Timestamp de início em segundos.
        fim: Timestamp de fim em segundos.
        texto: Texto transcrito do segmento.
        speaker_id: Label do diarizador (ex: SPEAKER_00).
        participante: Participante identificado (após resolução do mapeamento).
        confianca_asr: Score de confiança da transcrição ASR (0.0 a 1.0).
    """

    inicio: float
    fim: float
    texto: str
    speaker_id: str = "SPEAKER_00"
    participante: Optional[ParticipanteAudiencia] = None
    confianca_asr: float = 1.0

    @property
    def duracao(self) -> float:
        """Duração do segmento em segundos."""
        return self.fim - self.inicio

    @property
    def timestamp_formatado(self) -> str:
        """Timestamp de início no formato MM:SS ou HH:MM:SS."""
        segundos_total = int(self.inicio)
        horas = segundos_total // 3600
        minutos = (segundos_total % 3600) // 60
        segundos = segundos_total % 60
        if horas > 0:
            return f"{horas:02d}:{minutos:02d}:{segundos:02d}"
        return f"{minutos:02d}:{segundos:02d}"


@dataclass
class MetadataAudiencia:
    """Metadados da audiência judicial fornecidos pelo advogado.

    Attributes:
        numero_processo: Número do processo no formato CNJ.
        data: Data da audiência no formato dd/mm/aaaa.
        tipo_audiencia: Tipo da audiência (instrução, conciliação, etc.).
        vara: Descrição da vara e foro.
        cidade_uf: Cidade e UF do foro.
        participantes_esperados: Mapeamento role_str → ParticipanteAudiencia.
    """

    numero_processo: str = ""
    data: str = ""
    tipo_audiencia: str = "Instrução e Julgamento"
    vara: str = ""
    cidade_uf: str = ""
    participantes_esperados: dict[str, ParticipanteAudiencia] = field(default_factory=dict)


@dataclass
class ResultadoTranscricao:
    """Resultado completo da transcrição de uma audiência judicial.

    Attributes:
        segmentos: Lista ordenada de segmentos transcritos.
        metadata: Metadados da audiência.
        mapeamento_speakers: Dicionário SPEAKER_XX → ParticipanteAudiencia.
        duracao_total: Duração total do áudio em segundos.
        modelo_asr: Nome do modelo de transcrição ASR usado.
        modelo_diarizacao: Nome do modelo de diarização usado.
        idioma: Código do idioma detectado (ex: 'pt').
        arquivo_origem: Nome do arquivo de áudio processado.
    """

    segmentos: list[SegmentoTranscricao] = field(default_factory=list)
    metadata: MetadataAudiencia = field(default_factory=MetadataAudiencia)
    mapeamento_speakers: dict[str, ParticipanteAudiencia] = field(default_factory=dict)
    duracao_total: float = 0.0
    modelo_asr: str = "whisper-large-v3"
    modelo_diarizacao: str = "pyannote/speaker-diarization-3.1"
    idioma: str = "pt"
    arquivo_origem: str = ""

    @property
    def num_participantes(self) -> int:
        """Número de speakers únicos identificados na transcrição."""
        return len({s.speaker_id for s in self.segmentos if s.speaker_id})

    @property
    def texto_completo(self) -> str:
        """Texto completo concatenado sem formatação de markdown."""
        return " ".join(s.texto.strip() for s in self.segmentos if s.texto.strip())


def aplicar_mapeamento(
    segmentos: list[SegmentoTranscricao],
    mapeamento: dict[str, ParticipanteAudiencia],
) -> list[SegmentoTranscricao]:
    """Aplica mapeamento speaker_id → participante nos segmentos.

    Segmentos sem speaker no mapeamento recebem um participante
    genérico com role DESCONHECIDO.

    Args:
        segmentos: Segmentos da transcrição com speaker_id definido.
        mapeamento: Dicionário SPEAKER_XX → ParticipanteAudiencia.

    Returns:
        Segmentos com campo participante preenchido.
    """
    participante_desconhecido = ParticipanteAudiencia(
        role=RoleParticipante.DESCONHECIDO,
        nome="Participante não identificado",
    )
    for seg in segmentos:
        seg.participante = mapeamento.get(seg.speaker_id, participante_desconhecido)
    return segmentos


def extrair_mapeamento_json(resposta_llm: str) -> dict[str, str]:
    """Extrai mapeamento SPEAKER_XX → nome do JSON na resposta do LLM.

    O LLM pode retornar texto explicativo antes e após o JSON.
    Esta função localiza e parseia apenas o bloco JSON.

    Args:
        resposta_llm: Texto completo da resposta do LLM.

    Returns:
        Dicionário SPEAKER_XX → nome do participante. Dict vazio se
        não houver JSON válido na resposta.
    """
    match = re.search(r"\{[^{}]+\}", resposta_llm, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except (ValueError, AttributeError):
        return {}
