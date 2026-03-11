"""Script de ingestão de legislação federal brasileira no Qdrant.

Processa textos de leis, gera embeddings e indexa no Qdrant.
Constrói também o índice BM25 em memória (serializado para reuso).

Uso:
    uv run python scripts/ingerir_legislacao.py --arquivo lei_lgpd.txt --fonte "Lei 13.709/2018 (LGPD)"
    uv run python scripts/ingerir_legislacao.py --help
"""

import argparse
import sys
from pathlib import Path

# Adiciona src/ ao path para importar o pacote ana
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ana.rag.ingestao import processar_documento
from ana.rag.embeddings import GeradorEmbeddings
from ana.rag.indexador import IndexadorQdrant
from ana.rag.modelos import AreaJuridica, TipoDocumento, VigenciaStatus

from loguru import logger


MAPEAMENTO_AREAS = {
    "civil": AreaJuridica.CIVIL,
    "penal": AreaJuridica.PENAL,
    "trabalhista": AreaJuridica.TRABALHISTA,
    "tributario": AreaJuridica.TRIBUTARIO,
    "consumidor": AreaJuridica.CONSUMIDOR,
    "dados": AreaJuridica.DADOS,
    "administrativo": AreaJuridica.ADMINISTRATIVO,
    "constitucional": AreaJuridica.CONSTITUCIONAL,
    "processual_civil": AreaJuridica.PROCESSUAL_CIVIL,
    "processual_penal": AreaJuridica.PROCESSUAL_PENAL,
}

MAPEAMENTO_TIPOS = {
    "lei_federal": TipoDocumento.LEI_FEDERAL,
    "lei_estadual": TipoDocumento.LEI_ESTADUAL,
    "sumula": TipoDocumento.SUMULA,
    "jurisprudencia": TipoDocumento.JURISPRUDENCIA,
}


def main():
    """Ponto de entrada do script de ingestão."""
    parser = argparse.ArgumentParser(
        description="Ingere documentos jurídicos no pipeline RAG do sistema ANA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Ingerir LGPD de arquivo .txt
  uv run python scripts/ingerir_legislacao.py \\
      --arquivo ~/docs/lgpd.txt \\
      --fonte "Lei 13.709/2018 (LGPD)" \\
      --area dados

  # Ingerir Código Civil com vigência
  uv run python scripts/ingerir_legislacao.py \\
      --arquivo ~/docs/codigo_civil.txt \\
      --fonte "Lei 10.406/2002 (Código Civil)" \\
      --tipo lei_federal \\
      --area civil \\
      --orgao "Congresso Nacional"
        """,
    )
    parser.add_argument(
        "--arquivo",
        required=True,
        help="Caminho para o arquivo .txt com o texto da lei",
    )
    parser.add_argument(
        "--fonte",
        required=True,
        help="Identificação da fonte (ex: 'Lei 13.709/2018 (LGPD)')",
    )
    parser.add_argument(
        "--tipo",
        default="lei_federal",
        choices=list(MAPEAMENTO_TIPOS.keys()),
        help="Tipo do documento jurídico (padrão: lei_federal)",
    )
    parser.add_argument(
        "--area",
        choices=list(MAPEAMENTO_AREAS.keys()),
        help="Área do direito",
    )
    parser.add_argument(
        "--orgao",
        help="Órgão emissor (ex: 'Congresso Nacional', 'STF')",
    )
    parser.add_argument(
        "--url",
        help="URL da fonte original",
    )
    parser.add_argument(
        "--batch-embeddings",
        type=int,
        default=32,
        help="Tamanho do batch para geração de embeddings (padrão: 32)",
    )
    parser.add_argument(
        "--recriar-colecao",
        action="store_true",
        help="Recria a collection no Qdrant do zero",
    )

    args = parser.parse_args()

    caminho = Path(args.arquivo)
    if not caminho.exists():
        logger.error(f"Arquivo não encontrado: {caminho}")
        sys.exit(1)

    logger.info(f"Iniciando ingestão: {args.fonte}")
    logger.info(f"Arquivo: {caminho}")

    # 1. Ler arquivo
    texto = caminho.read_text(encoding="utf-8")
    logger.info(f"Texto lido: {len(texto)} caracteres")

    # 2. Chunking jurídico
    tipo = MAPEAMENTO_TIPOS[args.tipo]
    area = MAPEAMENTO_AREAS.get(args.area) if args.area else None

    chunks = processar_documento(
        texto=texto,
        fonte=args.fonte,
        tipo=tipo,
        area=area,
        orgao=args.orgao,
        url_origem=args.url,
    )

    if not chunks:
        logger.error(f"Nenhum artigo encontrado em '{caminho}'. Verifique o formato.")
        sys.exit(1)

    logger.info(f"Chunks gerados: {len(chunks)}")

    # 3. Gerar embeddings
    logger.info("Gerando embeddings (pode demorar na primeira execução)...")
    gerador = GeradorEmbeddings(batch_size=args.batch_embeddings)
    textos = [c.texto for c in chunks]
    embeddings = gerador.gerar_batch(textos)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    logger.info(f"Embeddings gerados: {len(embeddings)} vetores de 1024 dims")

    # 4. Indexar no Qdrant
    indexador = IndexadorQdrant()
    if not indexador.verificar_conexao():
        logger.error("Qdrant não disponível. Execute: docker compose up -d qdrant")
        sys.exit(1)

    indexador.criar_colecao_legislacao(recriar=args.recriar_colecao)
    total = indexador.indexar_chunks(chunks)

    logger.success(
        f"Ingestão concluída!\n"
        f"  Fonte       : {args.fonte}\n"
        f"  Chunks      : {total} artigos indexados\n"
        f"  Collection  : {indexador.colecao_legislacao}\n"
        f"  Collections : {indexador.listar_colecoes()}"
    )


if __name__ == "__main__":
    main()
