#!/usr/bin/env python3
"""Script para construir o dicionário local de leis a partir do PostgreSQL.

Varre a tabela de legislação indexada no PostgreSQL e popula o dicionário
SQLite local (~/.local/share/ana/dicionario_leis.db) com leis e artigos.

Deve ser executado após indexar a legislação com o scraper Planalto.

Uso:
    uv run python scripts/construir_dicionario.py
    uv run python scripts/construir_dicionario.py --colecao minha_colecao
"""

import argparse
import sys
from pathlib import Path

# Garante que o pacote ana seja encontrado quando chamado da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrói o dicionário de leis a partir do PostgreSQL"
    )
    parser.add_argument(
        "--colecao",
        default=None,
        help="Nome da tabela PostgreSQL a varrer (padrão: legislacao_brasileira)",
    )
    args = parser.parse_args()

    from loguru import logger

    logger.info("=== ANA: Construindo dicionário de leis ===")

    try:
        from ana.validacao.dicionario import DicionarioLeis

        dic = DicionarioLeis()

        logger.info("Varrendo tabela de legislação no PostgreSQL...")
        total = dic.construir_de_postgres(nome_colecao=args.colecao)

        logger.info(f"Dicionário concluído: {total} leis indexadas")
        logger.info(f"Total no dicionário: {dic.total_leis()} leis")
        logger.info(f"Salvo em: {dic.caminho}")

    except ImportError as e:
        logger.error(f"Dependência não instalada: {e}")
        logger.error("Execute: uv sync")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro ao construir dicionário: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
