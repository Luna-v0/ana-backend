"""Router de reescrita formal para o sistema ANA.

Expõe endpoint para reformulação de textos em português jurídico
brasileiro usando o modelo redator configurado no perfil ativo.
"""

from fastapi import APIRouter, HTTPException

from ana.api.schemas.redacao import RequisicaoReformular, RespostaReformular

router = APIRouter(prefix="/redacao", tags=["Redação"])


@router.post("/reformular", response_model=RespostaReformular)
async def reformular(requisicao: RequisicaoReformular) -> RespostaReformular:
    """Reformula texto para português jurídico brasileiro formal.

    Usa o modelo redator configurado no perfil ativo de modelos.

    Args:
        requisicao: Texto original e instruções opcionais.

    Returns:
        Texto reformulado e nome do modelo usado.

    Raises:
        HTTPException 422: Se o texto estiver vazio.
        HTTPException 503: Se o modelo falhar.
    """
    if not requisicao.texto.strip():
        raise HTTPException(status_code=422, detail="O campo 'texto' não pode estar vazio.")

    from ana.config import obter_configuracao
    from ana.config_modelos import obter_modelos
    from ana.providers.llm import OllamaLLMProvider

    config = obter_configuracao()
    modelos = obter_modelos()
    modelo = modelos.ativo.agentes.redator

    prompt_base = (
        "Você é um advogado brasileiro especialista em redação jurídica formal. "
        "Reescreva o texto a seguir em português jurídico brasileiro formal e técnico, "
        "mantendo o sentido original com precisão. "
        "Use linguagem adequada para peças processuais e documentos jurídicos. "
        "Retorne apenas o texto reformulado, sem explicações adicionais."
    )

    if requisicao.instrucoes.strip():
        prompt_base += f"\n\nInstruções adicionais: {requisicao.instrucoes.strip()}"

    prompt = f"{prompt_base}\n\nTexto original:\n{requisicao.texto}\n\nTexto reformulado:"

    try:
        llm = OllamaLLMProvider(modelo=modelo, host=config.ollama_host)
        texto_reformulado = llm.invocar(prompt, temperatura=0.3)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erro ao invocar modelo '{modelo}': {e}",
        )

    return RespostaReformular(
        texto_reformulado=texto_reformulado.strip(),
        modelo_usado=modelo,
    )
