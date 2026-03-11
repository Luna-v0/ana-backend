"""Bridge: conecta leis-br ao pipeline RAG do ANA.

O pacote ``leis-br`` cuida de scraping, caching e deduplicação.
Este módulo injeta a lógica de chunking + embeddings + indexação
do ANA como callback ``ingestor``.
"""

import os
from pathlib import Path

from loguru import logger

try:
    from leis_br import PipelineScrapers as _LeisBrPipeline
    from leis_br.modelos import DocumentoColetado, ResultadoColeta  # re-export
    _LEIS_BR_OK = True
except ImportError:
    _LeisBrPipeline = object  # type: ignore[assignment,misc]
    DocumentoColetado = None  # type: ignore[assignment]
    ResultadoColeta = None  # type: ignore[assignment]
    _LEIS_BR_OK = False
from ana.rag.embeddings import GeradorEmbeddings
from ana.rag.ingestao import processar_documento
from ana.rag.modelos import AreaJuridica, TipoDocumento, VigenciaStatus
from ana.storage import obter_cache, obter_vector_store


def _caminho_log() -> Path:
    if os.path.exists("/.dockerenv"):
        return Path("/app/data/update_log.json")
    return Path.home() / ".local" / "share" / "ana" / "update_log.json"


def _criar_ingestor(indexador, embeddings):
    """Cria o callback de ingestão que une chunking + embeddings + indexação."""

    def _ingerir(doc: DocumentoColetado) -> int:
        try:
            tipo = TipoDocumento(doc.tipo)
        except ValueError:
            tipo = TipoDocumento.LEI_FEDERAL

        area: AreaJuridica | None = None
        if doc.area:
            try:
                area = AreaJuridica(doc.area)
            except ValueError:
                pass

        try:
            vigencia = VigenciaStatus(doc.vigencia)
        except ValueError:
            vigencia = VigenciaStatus.ATIVA

        chunks = processar_documento(
            texto=doc.texto,
            fonte=doc.fonte,
            tipo=tipo,
            area=area,
            vigencia=vigencia,
            orgao=doc.orgao or None,
            url_origem=doc.url_origem,
        )

        if not chunks:
            logger.warning(f"Nenhum chunk gerado para '{doc.titulo}'")
            return 0

        textos = [c.texto for c in chunks]
        vetores = embeddings.gerar_batch(textos)
        for chunk, vetor in zip(chunks, vetores):
            chunk.embedding = vetor

        return indexador.indexar_chunks(chunks)

    return _ingerir


class PipelineScrapers(_LeisBrPipeline):  # type: ignore[misc]
    """PipelineScrapers do ANA: leis-br + ingestão RAG completa.

    Subclasse de :class:`leis_br.PipelineScrapers` que injeta
    automaticamente chunking, embeddings e indexação Qdrant/pgvector.
    """

    def __init__(self) -> None:
        indexador = obter_vector_store()
        cache = obter_cache()
        embeddings = GeradorEmbeddings()
        indexador.criar_colecao_legislacao()

        super().__init__(
            ingestor=_criar_ingestor(indexador, embeddings),
            cache=cache,
            caminho_log=_caminho_log(),
        )


def _dependencias_ok() -> bool:
    try:
        import bs4  # noqa: F401
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def executar_cli() -> None:
    """Ponto de entrada da CLI ``ana-scraper``."""
    import argparse
    import sys
    from datetime import datetime, timedelta

    parser = argparse.ArgumentParser(
        prog="ana-scraper",
        description="Coleta legislação pública e indexa no Qdrant/pgvector.",
    )
    parser.add_argument("--force", action="store_true", help="Força a coleta agora.")
    parser.add_argument("--fonte", metavar="NOME",
                        help="Coleta apenas esta fonte (planalto, lexml, stf, stj, tst).")
    parser.add_argument("--status", action="store_true",
                        help="Mostra o status das fontes e sai sem coletar.")
    args = parser.parse_args()

    if not _dependencias_ok():
        print(
            "❌ Dependências de scraping não instaladas.\n"
            "   Execute: uv sync --group scrapers",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline = PipelineScrapers()
    agora = datetime.now()
    intervalo = timedelta(days=7)

    st = pipeline.status()
    fontes_info = st["fontes"]

    print("\n📚 Status das fontes:\n")
    print(f"  {'Fonte':<12} {'Documentos':>12}  {'Última coleta':<22}  {'Situação'}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*22}  {'-'*20}")
    for nome, info in fontes_info.items():
        ultima_str = info.get("ultima_coleta") or "nunca"
        n_docs = info.get("documentos_no_cache", 0)
        ultima_dt = None
        if info.get("ultima_coleta"):
            ultima_dt = datetime.fromisoformat(info["ultima_coleta"])
            dias = (agora - ultima_dt).days
            situacao = f"✅ ok ({dias}d atrás)" if dias < 7 else f"⚠️  desatualizado ({dias}d)"
        else:
            situacao = "🔴 nunca coletado"
        print(f"  {nome:<12}  {n_docs:>12}  {ultima_str[:22]:<22}  {situacao}")
    print()

    if args.status:
        return

    todas_as_fontes = list(fontes_info.keys())

    if args.fonte:
        if args.fonte not in todas_as_fontes:
            print(
                f"❌ Fonte '{args.fonte}' desconhecida. "
                f"Disponíveis: {', '.join(todas_as_fontes)}",
                file=sys.stderr,
            )
            sys.exit(1)
        fontes_para_coletar = [args.fonte]
    elif args.force:
        fontes_para_coletar = todas_as_fontes
    else:
        fontes_para_coletar = []
        for nome, info in fontes_info.items():
            ultima_dt = None
            if info.get("ultima_coleta"):
                ultima_dt = datetime.fromisoformat(info["ultima_coleta"])
            if ultima_dt is None or (agora - ultima_dt) > intervalo:
                fontes_para_coletar.append(nome)

    if not fontes_para_coletar:
        print("✅ Todas as fontes foram atualizadas nos últimos 7 dias. Nada a fazer.")
        print("   Use --force para forçar a recoleta.\n")
        return

    print(f"🔄 Coletando: {', '.join(fontes_para_coletar)}\n")

    total_novos = 0
    total_erros = 0
    for nome in fontes_para_coletar:
        print(f"  ⬇️  {nome}...", end=" ", flush=True)
        resultado = pipeline.coletar_fonte(nome)
        total_novos += resultado.documentos_novos
        total_erros += len(resultado.erros)
        print(
            f"{resultado.documentos_novos} novos, "
            f"{resultado.documentos_ignorados} iguais, "
            f"{len(resultado.erros)} erros "
            f"({resultado.duracao_segundos:.1f}s)"
        )
        for erro in resultado.erros:
            print(f"     ⚠️  {erro}")

    print(f"\n✅ Concluído — {total_novos} documentos novos indexados, {total_erros} erros.")
    log = _caminho_log()
    if log.exists():
        print(f"   Log: {log}\n")
