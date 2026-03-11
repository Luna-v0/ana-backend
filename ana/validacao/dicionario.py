"""Dicionário de leis para validação de existência e vigência.

Mantém um SQLite local (dicionario_leis.db) populado a partir da tabela
'legislacao_brasileira' no PostgreSQL. A verificação é offline/instantânea.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger


def _caminho_db() -> Path:
    if os.path.exists("/.dockerenv"):
        caminho = Path("/app/data/dicionario_leis.db")
    else:
        caminho = Path.home() / ".local" / "share" / "ana" / "dicionario_leis.db"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def _get_conn(caminho: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(caminho or _caminho_db()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# Mapeamento de apelidos comuns para número de lei
_ALIASES: dict[str, str] = {
    "cf": "constituicao_federal_1988",
    "cf/88": "constituicao_federal_1988",
    "constituição federal": "constituicao_federal_1988",
    "clt": "5452/1943",
    "eca": "8069/1990",
    "lgpd": "13709/2018",
    "código civil": "10406/2002",
    "código penal": "2848/1940",
    "código de processo civil": "13105/2015",
    "código de processo penal": "3689/1941",
    "código tributário nacional": "5172/1966",
    "código de defesa do consumidor": "8078/1990",
    "código eleitoral": "4737/1965",
}


class DicionarioLeis:
    """Dicionário local de leis e artigos para validação rápida.

    Attributes:
        caminho: Path para o SQLite do dicionário.
    """

    def __init__(self, caminho: Path | None = None) -> None:
        self.caminho = caminho or _caminho_db()
        self._inicializar_schema()

    def _inicializar_schema(self) -> None:
        """Cria as tabelas do dicionário se não existirem."""
        conn = _get_conn(self.caminho)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leis (
                    chave           TEXT PRIMARY KEY,
                    nome            TEXT NOT NULL,
                    vigencia        TEXT NOT NULL DEFAULT 'ativa',
                    total_artigos   INTEGER NOT NULL DEFAULT 0,
                    data_publicacao TEXT,
                    revogada_por    TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artigos (
                    lei_chave   TEXT NOT NULL REFERENCES leis(chave) ON DELETE CASCADE,
                    numero      TEXT NOT NULL,
                    ementa      TEXT,
                    PRIMARY KEY (lei_chave, numero)
                )
            """)
        conn.close()

    def _normalizar_chave(self, lei: str) -> str:
        """Normaliza referência de lei para chave de lookup."""
        lei_lower = lei.lower().strip()
        if lei_lower in _ALIASES:
            return _ALIASES[lei_lower]
        # Tenta extrair número/ano: "13.709/2018" → "13709/2018"
        sem_pontos = re.sub(r"\.", "", lei_lower)
        return sem_pontos

    def construir_de_postgres(self, nome_colecao: str | None = None) -> int:
        """Varre tabela do PostgreSQL e popula o dicionário SQLite.

        Extrai fontes únicas (nome da lei) e artigos do campo 'artigo'
        no payload. Útil após re-indexação da legislação.

        Args:
            nome_colecao: Tabela a varrer. Usa legislacao_brasileira se None.

        Returns:
            Número de leis inseridas/atualizadas.
        """
        from ana.config import obter_configuracao
        from ana.storage.pgvector_store import IndexadorPgVector

        config = obter_configuracao()
        colecao = nome_colecao or config.colecao_legislacao
        indexador = IndexadorPgVector()

        conn_pg = indexador._get_conn()
        try:
            rows = conn_pg.execute(
                f"SELECT fonte, artigo, vigencia FROM {colecao} WHERE fonte IS NOT NULL"
            ).fetchall()
        finally:
            conn_pg.close()

        # Agrega por fonte
        fontes: dict[str, dict[str, Any]] = {}
        for row in rows:
            fonte = row[0]
            artigo = row[1]
            vigencia = row[2] or "ativa"
            if fonte not in fontes:
                fontes[fonte] = {"vigencia": vigencia, "artigos": set()}
            if artigo:
                fontes[fonte]["artigos"].add(artigo)

        conn_sq = _get_conn(self.caminho)
        total = 0
        with conn_sq:
            for fonte, dados in fontes.items():
                chave = self._normalizar_chave(fonte)
                conn_sq.execute(
                    """INSERT OR REPLACE INTO leis (chave, nome, vigencia, total_artigos)
                       VALUES (?, ?, ?, ?)""",
                    (chave, fonte, dados["vigencia"], len(dados["artigos"])),
                )
                for artigo in dados["artigos"]:
                    conn_sq.execute(
                        "INSERT OR IGNORE INTO artigos (lei_chave, numero) VALUES (?, ?)",
                        (chave, artigo),
                    )
                total += 1
        conn_sq.close()

        logger.info(f"Dicionário construído: {total} leis, {len(rows)} chunks varridos")
        return total

    def validar_existencia(
        self, lei: str, artigo: str | None = None
    ) -> dict[str, Any]:
        """Verifica se uma lei (e opcionalmente um artigo) existe no dicionário.

        Args:
            lei: Referência à lei (ex: '13.709/2018', 'LGPD', 'Código Civil').
            artigo: Número do artigo (ex: '7', 'Art. 7').

        Returns:
            Dicionário com campos:
            - ``status``: 'EXISTE_E_VIGENTE' | 'LEI_NAO_ENCONTRADA' |
                'ARTIGO_NAO_EXISTE' | 'LEI_REVOGADA'
            - ``lei``: Nome completo encontrado (se disponível).
            - ``artigo``: Artigo validado.
            - ``detalhe``: Mensagem informativa.
        """
        chave = self._normalizar_chave(lei)
        conn = _get_conn(self.caminho)
        try:
            row_lei = conn.execute(
                "SELECT * FROM leis WHERE chave = ? OR chave LIKE ?",
                (chave, f"%{chave}%"),
            ).fetchone()
        finally:
            conn.close()

        if row_lei is None:
            return {
                "status": "LEI_NAO_ENCONTRADA",
                "lei": lei,
                "artigo": artigo,
                "detalhe": f"Lei '{lei}' não encontrada no dicionário local.",
            }

        if row_lei["vigencia"] == "revogada":
            return {
                "status": "LEI_REVOGADA",
                "lei": row_lei["nome"],
                "artigo": artigo,
                "detalhe": (
                    f"Lei '{row_lei['nome']}' foi revogada"
                    + (f" por {row_lei['revogada_por']}" if row_lei["revogada_por"] else "") + "."
                ),
            }

        if artigo is not None:
            # Normaliza número do artigo
            num_artigo = re.sub(r"[Aa]rt(?:igo)?\.?\s*", "", artigo).strip().rstrip("°º")
            conn = _get_conn(self.caminho)
            try:
                row_art = conn.execute(
                    "SELECT * FROM artigos WHERE lei_chave = ? AND numero LIKE ?",
                    (row_lei["chave"], f"%{num_artigo}%"),
                ).fetchone()
            finally:
                conn.close()

            if row_art is None:
                return {
                    "status": "ARTIGO_NAO_EXISTE",
                    "lei": row_lei["nome"],
                    "artigo": artigo,
                    "detalhe": (
                        f"Art. {num_artigo} não encontrado em '{row_lei['nome']}' "
                        f"(total indexado: {row_lei['total_artigos']} artigos)."
                    ),
                }

        return {
            "status": "EXISTE_E_VIGENTE",
            "lei": row_lei["nome"],
            "artigo": artigo,
            "detalhe": f"'{row_lei['nome']}' está vigente no dicionário.",
        }

    def total_leis(self) -> int:
        """Retorna o número de leis no dicionário."""
        conn = _get_conn(self.caminho)
        total = conn.execute("SELECT COUNT(*) FROM leis").fetchone()[0]
        conn.close()
        return total
