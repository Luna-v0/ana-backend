"""Router de endpoints de sessões de processos jurídicos (Spec 06).

Gerencia o ciclo de vida de sessões: criação, listagem, upload de documentos
e busca vetorial intra-sessão, global e cross-sessão.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from ana.api.schemas.sessoes import (
    RequisicaoAtualizarSessao,
    RequisicaoBuscaSessao,
    RequisicaoCriarSessao,
    RespostaBuscaSessao,
    RespostaDocumento,
    RespostaSessao,
    ResultadoBuscaSessao,
)

router = APIRouter(prefix="/sessoes", tags=["Sessões"])

_EXTENSOES_SUPORTADAS = {".pdf", ".docx", ".txt", ".md"}


# =============================================================================
# Helpers
# =============================================================================

def _sessao_para_resposta(sessao) -> RespostaSessao:
    return RespostaSessao(
        id=sessao.id,
        numero_processo=sessao.numero_processo,
        tipo_acao=sessao.tipo_acao,
        area=sessao.area,
        vara=sessao.vara,
        cidade_uf=sessao.cidade_uf,
        status=sessao.status,
        criado_em=sessao.criado_em,
        atualizado_em=sessao.atualizado_em,
        partes=sessao.partes,
        prazos=sessao.prazos,
    )


def _doc_para_resposta(doc) -> RespostaDocumento:
    return RespostaDocumento(
        id=doc.id,
        sessao_id=doc.sessao_id,
        nome=doc.nome,
        tipo=doc.tipo,
        tamanho_bytes=doc.tamanho_bytes,
        chunks_indexados=doc.chunks_indexados,
        criado_em=doc.criado_em,
    )


def _gerar_sessao_id() -> str:
    return "sess_" + uuid.uuid4().hex[:8]


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/", response_model=RespostaSessao, status_code=201)
async def criar_sessao(req: RequisicaoCriarSessao) -> RespostaSessao:
    """Cria uma nova sessão de processo jurídico."""
    from ana.sessoes.modelos import Sessao
    from ana.sessoes.repositorio import criar_sessao as _criar, inicializar_banco

    inicializar_banco()

    sessao = Sessao(
        id=_gerar_sessao_id(),
        numero_processo=req.numero_processo,
        tipo_acao=req.tipo_acao,
        area=req.area,
        vara=req.vara,
        cidade_uf=req.cidade_uf,
        partes=req.partes,
        prazos=req.prazos,
    )
    criada = _criar(sessao)
    return _sessao_para_resposta(criada)


@router.get("/", response_model=list[RespostaSessao])
async def listar_sessoes() -> list[RespostaSessao]:
    """Lista todas as sessões existentes."""
    from ana.sessoes.repositorio import listar_sessoes as _listar, inicializar_banco

    inicializar_banco()
    return [_sessao_para_resposta(s) for s in _listar()]


@router.get("/{sessao_id}", response_model=RespostaSessao)
async def obter_sessao(sessao_id: str) -> RespostaSessao:
    """Retorna uma sessão pelo ID."""
    from ana.sessoes.repositorio import obter_sessao as _obter

    sessao = _obter(sessao_id)
    if sessao is None:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")
    return _sessao_para_resposta(sessao)


@router.patch("/{sessao_id}", response_model=RespostaSessao)
async def atualizar_sessao(sessao_id: str, req: RequisicaoAtualizarSessao) -> RespostaSessao:
    """Atualiza campos de uma sessão."""
    from ana.sessoes.repositorio import atualizar_sessao as _atualizar

    campos = {k: v for k, v in req.model_dump().items() if v is not None}
    sessao = _atualizar(sessao_id, campos)
    if sessao is None:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")
    return _sessao_para_resposta(sessao)


@router.delete("/{sessao_id}", status_code=204)
async def deletar_sessao(sessao_id: str) -> None:
    """Remove sessão, seus documentos no SQLite e os chunks no pgvector."""
    from ana.sessoes.repositorio import deletar_sessao as _deletar
    from ana.sessoes.ingestao import remover_todos_documentos_sessao

    try:
        remover_todos_documentos_sessao(sessao_id)
    except Exception:
        pass

    removido = _deletar(sessao_id)
    if not removido:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")


@router.post("/{sessao_id}/documentos", response_model=RespostaDocumento, status_code=201)
async def upload_documento(
    sessao_id: str,
    arquivo: UploadFile = File(...),
) -> RespostaDocumento:
    """Faz upload e indexa um documento na sessão."""
    from ana.sessoes.repositorio import obter_sessao, criar_documento, inicializar_banco
    from ana.sessoes.ingestao import ingerir_documento_sessao

    inicializar_banco()

    if obter_sessao(sessao_id) is None:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")

    nome = arquivo.filename or "documento"
    sufixo = Path(nome).suffix.lower()

    if sufixo not in _EXTENSOES_SUPORTADAS:
        raise HTTPException(
            status_code=422,
            detail=f"Formato '{sufixo}' não suportado. Use: {', '.join(_EXTENSOES_SUPORTADAS)}",
        )

    conteudo = await arquivo.read()
    if not conteudo:
        raise HTTPException(status_code=422, detail="Arquivo vazio")

    try:
        doc = ingerir_documento_sessao(
            sessao_id=sessao_id,
            nome=nome,
            conteudo=conteudo,
            sufixo=sufixo,
        )
    except (ValueError, ImportError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na indexação: {e}") from e

    doc_persistido = criar_documento(doc)
    return _doc_para_resposta(doc_persistido)


@router.get("/{sessao_id}/documentos", response_model=list[RespostaDocumento])
async def listar_documentos(sessao_id: str) -> list[RespostaDocumento]:
    """Lista documentos indexados de uma sessão."""
    from ana.sessoes.repositorio import obter_sessao, listar_documentos as _listar

    if obter_sessao(sessao_id) is None:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")

    return [_doc_para_resposta(d) for d in _listar(sessao_id)]


@router.delete("/{sessao_id}/documentos/{doc_id}", status_code=204)
async def remover_documento(sessao_id: str, doc_id: str) -> None:
    """Remove um documento da sessão (SQLite + chunks pgvector)."""
    from ana.sessoes.repositorio import deletar_documento
    from ana.sessoes.ingestao import remover_documento_sessao

    try:
        remover_documento_sessao(sessao_id, doc_id)
    except Exception:
        pass

    removido = deletar_documento(doc_id)
    if not removido:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado")


@router.post("/{sessao_id}/buscar", response_model=RespostaBuscaSessao)
async def buscar_em_sessao(
    sessao_id: str,
    req: RequisicaoBuscaSessao,
) -> RespostaBuscaSessao:
    """Executa busca vetorial em documentos da sessão.

    Modos:
    - **intra**: Apenas documentos desta sessão
    - **global**: Legislação + documentos desta sessão
    - **cross**: Documentos de outras sessões (exceto esta)
    """
    from ana.sessoes.repositorio import obter_sessao
    from ana.sessoes.busca import (
        buscar_intra_sessao,
        buscar_global,
        buscar_cross_sessao,
    )

    if obter_sessao(sessao_id) is None:
        raise HTTPException(status_code=404, detail=f"Sessão '{sessao_id}' não encontrada")

    try:
        if req.modo == "intra":
            brutos = buscar_intra_sessao(req.query, sessao_id, req.top_k)
        elif req.modo == "global":
            brutos = buscar_global(req.query, sessao_id, req.top_k)
        else:
            brutos = buscar_cross_sessao(req.query, sessao_id, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {e}") from e

    resultados = [
        ResultadoBuscaSessao(
            id=r["id"],
            score=r["score"],
            texto=r["payload"].get("texto", ""),
            fonte=r["payload"].get("fonte", ""),
            sessao_id=r["payload"].get("sessao_id"),
            artigo=r["payload"].get("artigo"),
        )
        for r in brutos
    ]

    return RespostaBuscaSessao(
        query=req.query,
        modo=req.modo,
        total=len(resultados),
        resultados=resultados,
    )
