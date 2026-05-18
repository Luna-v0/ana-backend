"""Router de geração de documentos jurídicos (peças processuais).

Expõe endpoint para geração de petições, contestações e recursos
como arquivos .docx, fundamentados em RAG + LLM redator.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ana.documentos.modelos import RequisicaoGerarDocumento, TipoPeca, label_peca

router = APIRouter(prefix="/documentos", tags=["Documentos"])

_MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@router.post("/gerar")
async def gerar_documento(requisicao: RequisicaoGerarDocumento) -> Response:
    """Gera uma peça processual como documento Word (.docx).

    Busca legislação e jurisprudência relevante via RAG e usa o LLM
    redator para compor o conteúdo de cada seção da peça.

    Args:
        requisicao: ID da sessão, tipo de peça e instruções opcionais.

    Returns:
        Arquivo .docx para download.

    Raises:
        HTTPException 404: Se a sessão não existir.
        HTTPException 503: Se o LLM ou RAG falharem.
    """
    from ana.documentos.gerador import gerar_documento as _gerar

    try:
        docx_bytes, nome_arquivo = _gerar(
            sessao_id=requisicao.sessao_id,
            tipo_peca=requisicao.tipo_peca,
            instrucoes=requisicao.instrucoes,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erro ao gerar documento: {e}",
        )

    return Response(
        content=docx_bytes,
        media_type=_MIME_DOCX,
        headers={"Content-Disposition": f'attachment; filename="{nome_arquivo}"'},
    )


@router.get("/tipos")
async def listar_tipos() -> list[dict]:
    """Lista os tipos de peças suportados."""
    return [
        {"tipo": t.value, "label": label_peca(t)}
        for t in TipoPeca
    ]
