"""Formatação da transcrição de audiências como markdown estruturado.

Gera o documento final com cabeçalho, lista de participantes,
transcript com timestamps e rodapé com metadados técnicos.

Nota:
    O aviso de revisão obrigatória ao final é mandatório — transcrições
    automáticas nunca devem ser usadas processualmente sem revisão.
"""

from __future__ import annotations

from .modelos import ResultadoTranscricao, RoleParticipante


_CABECALHO_TEMPLATE = """\
# Transcrição — Audiência de {tipo_audiencia}
**Processo**: {numero_processo}
**Data**: {data}
**Vara**: {vara}
**Local**: {cidade_uf}

---

"""

_PARTICIPANTES_TEMPLATE = """\
## Participantes Identificados

{lista_participantes}

---

"""


def formatar_transcript_markdown(resultado: ResultadoTranscricao) -> str:
    """Formata o resultado da transcrição como markdown estruturado.

    Gera documento com:
    - Cabeçalho com metadados da audiência
    - Lista de participantes identificados com scores de confiança
    - Transcript segmento a segmento com timestamps e labels
    - Rodapé com informações técnicas e aviso obrigatório de revisão

    Args:
        resultado: Resultado completo da transcrição com segmentos e metadata.

    Returns:
        String markdown pronta para exibição no Gradio ou exportação.
    """
    partes: list[str] = []

    # Cabeçalho com metadados
    meta = resultado.metadata
    partes.append(_CABECALHO_TEMPLATE.format(
        tipo_audiencia=meta.tipo_audiencia or "Audiência",
        numero_processo=meta.numero_processo or "Não informado",
        data=meta.data or "Não informada",
        vara=meta.vara or "Não informada",
        cidade_uf=meta.cidade_uf or "Não informado",
    ))

    # Lista de participantes identificados
    if resultado.mapeamento_speakers:
        itens: list[str] = []
        for speaker_id, participante in sorted(resultado.mapeamento_speakers.items()):
            confianca_str = ""
            if participante.confianca < 1.0:
                confianca_str = f" _(confiança: {participante.confianca:.0%})_"
            itens.append(
                f"- **{speaker_id}**: {participante.label_formatado}{confianca_str}"
            )
        partes.append(_PARTICIPANTES_TEMPLATE.format(
            lista_participantes="\n".join(itens)
        ))

    # Corpo da transcrição
    partes.append("## Transcrição\n\n")

    speaker_anterior: str | None = None
    for seg in resultado.segmentos:
        if not seg.texto.strip():
            continue

        # Determina label do speaker para exibição
        if seg.participante and seg.participante.role != RoleParticipante.DESCONHECIDO:
            label = seg.participante.label_formatado
        elif seg.speaker_id in resultado.mapeamento_speakers:
            label = resultado.mapeamento_speakers[seg.speaker_id].label_formatado
        else:
            label = f"Participante ({seg.speaker_id})"

        # Novo bloco de fala quando o speaker muda
        if seg.speaker_id != speaker_anterior:
            partes.append(f"\n**[{seg.timestamp_formatado}] {label}:**\n")
            speaker_anterior = seg.speaker_id

        partes.append(f"{seg.texto.strip()} ")

    # Rodapé técnico com aviso obrigatório de revisão
    duracao_min = int(resultado.duracao_total // 60)
    duracao_seg = int(resultado.duracao_total % 60)
    partes.append(f"\n\n---\n\n")
    partes.append(
        f"_Gerado automaticamente — "
        f"Modelo ASR: `{resultado.modelo_asr}` | "
        f"Diarização: `{resultado.modelo_diarizacao}` | "
        f"Duração: {duracao_min}min {duracao_seg}s_\n"
    )
    partes.append(
        "_**ATENÇÃO**: Esta transcrição é gerada automaticamente e deve ser "
        "revisada pelo advogado antes de qualquer uso processual._\n"
    )

    return "".join(partes)
