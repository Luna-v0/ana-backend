"""Identificação dos participantes da audiência (4 camadas).

Pipeline de identificação de SPEAKER_XX → participante real:

Camada 1 — Metadata prévia: o advogado informa os participantes esperados.
Camada 2 — LLM por contexto: analisa as primeiras falas para mapear speakers.
Camada 3 — Voice enrollment: comparação com vozes cadastradas (opcional/futuro).
Camada 4 — Revisão manual: o advogado corrige atribuições incorretas.

Este módulo implementa as camadas 1 e 2 (as principais).
Camada 3 é marcada como futura (Fase 3). Camada 4 é responsabilidade do frontend.

Nota:
    A identificação por contexto usa os primeiros N segmentos da audiência,
    onde normalmente ocorrem as apresentações formais dos participantes.
"""

from __future__ import annotations

from loguru import logger

from ana.providers.llm import LLMProvider

from .modelos import (
    MetadataAudiencia,
    ParticipanteAudiencia,
    RoleParticipante,
    SegmentoTranscricao,
    aplicar_mapeamento,
    extrair_mapeamento_json,
)

# Número de segmentos iniciais a analisar para identificação por contexto
_SEGMENTOS_CONTEXTO = 20

# Prompt para identificação de speakers via LLM
_PROMPT_IDENTIFICACAO = """\
Analise o início desta transcrição de audiência judicial brasileira.
Identifique cada SPEAKER com base no que dizem na transcrição.

Dicas de identificação:
- O juiz(a) geralmente abre a audiência declarando-a aberta e conduz os trabalhos
- Advogados costumam se identificar com nome e número da OAB
- Testemunhas são chamadas pelo nome e prestam compromisso
- O promotor representa o Ministério Público
- Partes (autor/réu) respondem às perguntas do juiz

Participantes esperados pelo advogado:
{participantes_esperados}

Transcrição (primeiros segmentos):
{transcricao_inicial}

Retorne APENAS um JSON com o mapeamento SPEAKER_XX → nome completo do participante.
Exemplo: {{"SPEAKER_00": "Juiz Dr. Carlos Silva", "SPEAKER_01": "Adv. Dra. Maria Santos"}}

Se não conseguir identificar um speaker com segurança, omita-o do JSON.
Mapeamento:"""


def identificar_por_metadata(
    segmentos: list[SegmentoTranscricao],
    metadata: MetadataAudiencia,
) -> dict[str, ParticipanteAudiencia]:
    """Camada 1: tenta identificar speakers via metadata prévia.

    Quando há apenas 1 participante por role e o número de speakers
    na transcrição bate com o número de participantes esperados,
    faz a atribuição automática por posição de fala (quem fala primeiro
    é provavelmente o juiz, etc.).

    Esta é uma heurística simples. Casos ambíguos ficam para a Camada 2.

    Args:
        segmentos: Segmentos transcritos com speaker_id.
        metadata: Metadados com participantes_esperados fornecidos pelo advogado.

    Returns:
        Mapeamento parcial SPEAKER_XX → ParticipanteAudiencia (pode estar incompleto).
    """
    if not metadata.participantes_esperados:
        return {}

    # Coleta speakers únicos em ordem de aparição
    speakers_em_ordem: list[str] = []
    vistos: set[str] = set()
    for seg in segmentos:
        if seg.speaker_id and seg.speaker_id not in vistos:
            speakers_em_ordem.append(seg.speaker_id)
            vistos.add(seg.speaker_id)

    if not speakers_em_ordem:
        logger.debug("Camada 1: nenhum speaker detectado, mapeamento vazio")
        return {}

    # Ordena participantes por prioridade de aparição esperada na audiência
    _PRIORIDADE_ROLE = [
        RoleParticipante.JUIZ,
        RoleParticipante.ADVOGADO_AUTOR,
        RoleParticipante.ADVOGADO_REU,
        RoleParticipante.TESTEMUNHA,
        RoleParticipante.PERITO,
        RoleParticipante.PROMOTOR,
        RoleParticipante.DEFENSOR,
        RoleParticipante.PARTE_AUTORA,
        RoleParticipante.PARTE_RE,
        RoleParticipante.ESCRIVAO,
        RoleParticipante.DESCONHECIDO,
    ]
    participantes_ordenados = sorted(
        metadata.participantes_esperados.values(),
        key=lambda p: _PRIORIDADE_ROLE.index(p.role) if p.role in _PRIORIDADE_ROLE else len(_PRIORIDADE_ROLE),
    )

    mapeamento: dict[str, ParticipanteAudiencia] = {}

    if len(speakers_em_ordem) == len(participantes_ordenados):
        # Contagens batem: atribuição posicional completa
        for speaker_id, participante in zip(speakers_em_ordem, participantes_ordenados):
            participante.speaker_id = speaker_id
            mapeamento[speaker_id] = participante
            logger.debug(
                f"Camada 1 (posicional): {speaker_id} → {participante.nome} "
                f"(role: {participante.role.value})"
            )
        logger.info(
            f"Camada 1: atribuição posicional completa — "
            f"{len(mapeamento)} speaker(s) mapeado(s)"
        )
    else:
        # Contagens não batem: apenas mapeia o juiz (heurística: fala primeiro)
        juizes = [p for p in participantes_ordenados if p.role == RoleParticipante.JUIZ]
        if juizes:
            primeiro_speaker = speakers_em_ordem[0]
            juizes[0].speaker_id = primeiro_speaker
            mapeamento[primeiro_speaker] = juizes[0]
            logger.debug(
                f"Camada 1 (juiz-only): {primeiro_speaker} → {juizes[0].nome} "
                f"(heurística: fala primeiro)"
            )
        logger.info(
            f"Camada 1: contagens não batem "
            f"({len(speakers_em_ordem)} speakers vs {len(participantes_ordenados)} participantes) "
            f"— apenas juiz mapeado ({len(mapeamento)} total), restante para Camada 2"
        )

    return mapeamento


def identificar_por_contexto_llm(
    segmentos: list[SegmentoTranscricao],
    metadata: MetadataAudiencia,
    provedor_llm: LLMProvider,
    mapeamento_parcial: dict[str, ParticipanteAudiencia] | None = None,
) -> dict[str, ParticipanteAudiencia]:
    """Camada 2: usa LLM para identificar speakers por contexto textual.

    Analisa os primeiros segmentos da audiência (onde ocorrem as apresentações)
    e solicita ao LLM que mapeie cada SPEAKER_XX ao participante correspondente.

    Args:
        segmentos: Segmentos transcritos com speaker_id.
        metadata: Metadados com participantes esperados.
        provedor_llm: Provedor LLM para geração de texto.
        mapeamento_parcial: Mapeamento já obtido pela Camada 1 (para evitar redundância).

    Returns:
        Mapeamento SPEAKER_XX → ParticipanteAudiencia enriquecido com identificações do LLM.
    """
    mapeamento: dict[str, ParticipanteAudiencia] = dict(mapeamento_parcial or {})

    # Identifica speakers ainda sem identificação
    speakers_sem_id = {
        seg.speaker_id
        for seg in segmentos
        if seg.speaker_id and seg.speaker_id not in mapeamento
    }
    if not speakers_sem_id:
        logger.info("Todos os speakers já identificados pela Camada 1")
        return mapeamento

    # Prepara trecho inicial da transcrição para o LLM
    segmentos_iniciais = segmentos[:_SEGMENTOS_CONTEXTO]
    linhas_transcricao = [
        f'{seg.speaker_id}: "{seg.texto.strip()}"'
        for seg in segmentos_iniciais
        if seg.texto.strip()
    ]
    transcricao_inicial = "\n".join(linhas_transcricao)

    # Prepara lista de participantes esperados
    linhas_participantes = [
        f"- {role}: {p.nome}" + (f" ({p.oab})" if p.oab else "")
        for role, p in metadata.participantes_esperados.items()
    ] or ["- (nenhum participante informado)"]

    prompt = _PROMPT_IDENTIFICACAO.format(
        participantes_esperados="\n".join(linhas_participantes),
        transcricao_inicial=transcricao_inicial,
    )

    logger.info("Identificando speakers via LLM")
    try:
        texto_resposta = provedor_llm.invocar(prompt, temperatura=0.1)
    except Exception as e:
        logger.warning(f"Falha na identificação via LLM: {e}")
        return mapeamento

    # Extrai mapeamento JSON da resposta
    mapeamento_nomes = extrair_mapeamento_json(texto_resposta)
    if not mapeamento_nomes:
        logger.warning("LLM não retornou JSON válido para mapeamento de speakers")
        return mapeamento

    # Converte nomes para ParticipanteAudiencia (confiança reduzida = LLM)
    for speaker_id, nome_identificado in mapeamento_nomes.items():
        if speaker_id in mapeamento:
            continue  # já identificado pela Camada 1
        participante = ParticipanteAudiencia(
            role=_inferir_role(nome_identificado),
            nome=nome_identificado,
            speaker_id=speaker_id,
            confianca=0.75,  # confiança reduzida para identificação automática
        )
        mapeamento[speaker_id] = participante
        logger.debug(f"LLM: {speaker_id} → {nome_identificado}")

    logger.info(f"Identificação LLM: {len(mapeamento_nomes)} speaker(s) mapeado(s)")
    return mapeamento


def _inferir_role(nome_identificado: str) -> RoleParticipante:
    """Infere o role do participante com base no nome retornado pelo LLM.

    Aplica heurísticas simples: verifica se o nome começa com palavras
    indicativas de role (ex: "Juiz", "Adv.", "Testemunha").

    Args:
        nome_identificado: Nome/título retornado pelo LLM.

    Returns:
        RoleParticipante correspondente (DESCONHECIDO se não identificado).
    """
    nome_lower = nome_identificado.lower()

    if any(p in nome_lower for p in ["juiz", "juíza", "excelência", "mm."]):
        return RoleParticipante.JUIZ
    if "promotor" in nome_lower or "ministério público" in nome_lower:
        return RoleParticipante.PROMOTOR
    if "defensor" in nome_lower:
        return RoleParticipante.DEFENSOR
    if "perito" in nome_lower:
        return RoleParticipante.PERITO
    if "testemunha" in nome_lower:
        return RoleParticipante.TESTEMUNHA
    # Verificar advogado ANTES de parte, pois "Adv. Autora" indica advogado, não parte
    if any(p in nome_lower for p in ["adv.", "advogado", "advogada", "oab"]):
        if any(p in nome_lower for p in ["autor", "autora"]):
            return RoleParticipante.ADVOGADO_AUTOR
        if any(p in nome_lower for p in ["réu", "ré", "reo"]):
            return RoleParticipante.ADVOGADO_REU
        return RoleParticipante.ADVOGADO_AUTOR  # padrão para advogado sem especificação
    if any(p in nome_lower for p in ["autor", "requerente"]):
        return RoleParticipante.PARTE_AUTORA
    if any(p in nome_lower for p in ["réu", "ré", "requerido"]):
        return RoleParticipante.PARTE_RE

    return RoleParticipante.DESCONHECIDO


def identificar_participantes(
    segmentos: list[SegmentoTranscricao],
    metadata: MetadataAudiencia,
    provedor_llm: LLMProvider,
) -> tuple[list[SegmentoTranscricao], dict[str, ParticipanteAudiencia]]:
    """Pipeline completo de identificação de participantes (Camadas 1 e 2).

    Executa sequencialmente:
    1. Identificação por metadata (heurística de ordem de fala)
    2. Identificação por contexto LLM (para speakers não identificados)
    3. Aplica o mapeamento final nos segmentos

    Args:
        segmentos: Segmentos transcritos com speaker_id da diarização.
        metadata: Metadados da audiência com participantes esperados.
        provedor_llm: Provedor LLM para identificação por contexto.

    Returns:
        Tupla (segmentos_com_participantes, mapeamento_final).
        segmentos_com_participantes: segmentos com campo participante preenchido.
        mapeamento_final: dict SPEAKER_XX → ParticipanteAudiencia.
    """
    logger.info("Iniciando identificação de participantes")

    # Camada 1: metadata + heurística
    mapeamento = identificar_por_metadata(segmentos, metadata)
    logger.info(f"Camada 1 (metadata): {len(mapeamento)} speaker(s) identificado(s)")

    # Camada 2: LLM por contexto (para os não identificados)
    mapeamento = identificar_por_contexto_llm(
        segmentos,
        metadata,
        provedor_llm,
        mapeamento_parcial=mapeamento,
    )
    total_identificados = len(mapeamento)
    logger.info(f"Camada 2 (LLM): {total_identificados} speaker(s) identificado(s) no total")

    # Aplica o mapeamento nos segmentos
    segmentos = aplicar_mapeamento(segmentos, mapeamento)

    return segmentos, mapeamento
