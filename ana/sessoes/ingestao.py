"""Ingestão de documentos para sessões de processos.

Extrai texto de PDFs, DOCX e TXT, chunkeia via pipeline RAG existente
e indexa na tabela 'processos' do PostgreSQL com sessao_id preenchido.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from ana.sessoes.modelos import DocumentoSessao


def extrair_texto(conteudo: bytes, sufixo: str, nome: str = "") -> str:
    """Extrai texto plano de bytes de um documento.

    Suporta PDF (pdfminer.six), DOCX (python-docx) e TXT/MD.

    Args:
        conteudo: Bytes brutos do arquivo.
        sufixo: Extensão do arquivo (ex: '.pdf', '.docx', '.txt').
        nome: Nome do arquivo para logging.

    Returns:
        Texto extraído como string.

    Raises:
        ValueError: Se o formato não for suportado.
    """
    sufixo = sufixo.lower()

    if sufixo == ".pdf":
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io

            entrada = io.BytesIO(conteudo)
            saida = io.StringIO()
            extract_text_to_fp(entrada, saida, laparams=LAParams(), output_type="text", codec="utf-8")
            texto = saida.getvalue()
        except ImportError:
            # Fallback para PyMuPDF se pdfminer não estiver disponível
            try:
                import fitz
                import io

                with fitz.open(stream=conteudo, filetype="pdf") as doc:
                    texto = "\n".join(pagina.get_text() for pagina in doc)
            except ImportError as e:
                raise ImportError(
                    "Instale pdfminer.six ou pymupdf: uv sync --group documentos"
                ) from e

    elif sufixo in (".docx",):
        try:
            from docx import Document
            import io

            doc = Document(io.BytesIO(conteudo))
            texto = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError as e:
            raise ImportError(
                "Instale python-docx: uv sync --group documentos"
            ) from e

    elif sufixo in (".txt", ".md", ".text"):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                texto = conteudo.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            texto = conteudo.decode("utf-8", errors="replace")

    else:
        raise ValueError(
            f"Formato não suportado: '{sufixo}'. Use PDF, DOCX ou TXT."
        )

    logger.info(f"Texto extraído de '{nome}': {len(texto)} chars")
    return texto


def ingerir_documento_sessao(
    sessao_id: str,
    nome: str,
    conteudo: bytes,
    sufixo: str,
    tipo: str = "documento_usuario",
) -> DocumentoSessao:
    """Ingere documento de sessão: extrai, chunkeia, gera embeddings, indexa.

    Args:
        sessao_id: ID da sessão de processo.
        nome: Nome original do arquivo.
        conteudo: Bytes do arquivo.
        sufixo: Extensão do arquivo (ex: '.pdf').
        tipo: Tipo semântico do documento.

    Returns:
        DocumentoSessao com chunks_indexados preenchido.

    Raises:
        ValueError: Se a extração ou chunking falharem.
    """
    from ana.rag.ingestao import processar_documento
    from ana.rag.embeddings import GeradorEmbeddings
    from ana.rag.modelos import TipoDocumento
    from ana.storage.pgvector_store import IndexadorPgVector

    # 1. Extrai texto
    texto = extrair_texto(conteudo, sufixo, nome)
    if not texto.strip():
        raise ValueError(f"Documento '{nome}' não contém texto extraível.")

    # 2. Chunkeia via pipeline RAG
    chunks = processar_documento(
        texto=texto,
        fonte=nome,
        tipo=TipoDocumento.DOCUMENTO_USUARIO,
        sessao_id=sessao_id,
    )

    if not chunks:
        # Para documentos sem estrutura de artigos, cria um único chunk
        from ana.rag.modelos import ChunkJuridico, MetadataChunkJuridico, VigenciaStatus
        meta = MetadataChunkJuridico(
            fonte=nome,
            tipo=TipoDocumento.DOCUMENTO_USUARIO,
            sessao_id=sessao_id,
            vigencia=VigenciaStatus.ATIVA,
        )
        chunks = [ChunkJuridico(texto=texto[:8000], metadata=meta)]
        logger.info(f"Documento '{nome}' sem artigos — indexado como chunk único")

    # 3. Gera embeddings
    gerador = GeradorEmbeddings()
    textos = [c.texto for c in chunks]
    embeddings = gerador.gerar_batch(textos)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # 4. Indexa na tabela 'processos'
    indexador = IndexadorPgVector()
    indexador.criar_colecao_processos()
    total = indexador.indexar_chunks(chunks, nome_colecao="processos")

    doc_id = str(uuid.uuid4())
    return DocumentoSessao(
        id=doc_id,
        sessao_id=sessao_id,
        nome=nome,
        tipo=sufixo.lstrip(".").lower(),
        tamanho_bytes=len(conteudo),
        chunks_indexados=total,
        criado_em="",
    )


def remover_documento_sessao(sessao_id: str, doc_id: str) -> int:
    """Remove chunks de um documento da tabela 'processos' no PostgreSQL.

    Args:
        sessao_id: ID da sessão.
        doc_id: ID do documento a remover (campo fonte no payload).

    Returns:
        Número de chunks removidos.
    """
    from ana.storage.pgvector_store import IndexadorPgVector

    indexador = IndexadorPgVector()
    conn = indexador._get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM processos WHERE sessao_id = %s AND payload->>'fonte' = %s",
            (sessao_id, doc_id),
        )
        removidos = cursor.rowcount
    finally:
        conn.close()

    logger.info(f"Removidos {removidos} chunks (sessao={sessao_id}, doc={doc_id})")
    return removidos


def remover_todos_documentos_sessao(sessao_id: str) -> int:
    """Remove todos os chunks de uma sessão da tabela 'processos'.

    Args:
        sessao_id: ID da sessão.

    Returns:
        Número de chunks removidos.
    """
    from ana.storage.pgvector_store import IndexadorPgVector

    indexador = IndexadorPgVector()
    conn = indexador._get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM processos WHERE sessao_id = %s",
            (sessao_id,),
        )
        removidos = cursor.rowcount
    finally:
        conn.close()

    logger.info(f"Removidos {removidos} chunks da sessão '{sessao_id}'")
    return removidos
