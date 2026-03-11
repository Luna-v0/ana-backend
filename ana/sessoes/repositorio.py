"""Repositório SQLite para metadata de sessões e documentos.

Armazena apenas metadata (IDs, nomes, datas, partes, prazos).
Os vetores ficam no PostgreSQL/pgvector via IndexadorPgVector.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ana.sessoes.modelos import DocumentoSessao, Sessao


def _caminho_db() -> Path:
    """Resolve o caminho do SQLite de sessões conforme ambiente."""
    if os.path.exists("/.dockerenv"):
        caminho = Path("/app/data/sessoes.db")
    else:
        caminho = Path.home() / ".local" / "share" / "ana" / "sessoes.db"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho


def _get_conn(caminho: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(caminho or _caminho_db()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def inicializar_banco() -> None:
    """Cria as tabelas do banco SQLite de sessões se não existirem."""
    conn = _get_conn()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessoes (
                id              TEXT PRIMARY KEY,
                numero_processo TEXT NOT NULL,
                tipo_acao       TEXT NOT NULL,
                area            TEXT NOT NULL DEFAULT 'civil',
                vara            TEXT NOT NULL DEFAULT '',
                cidade_uf       TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'em_andamento',
                partes          TEXT NOT NULL DEFAULT '{}',
                prazos          TEXT NOT NULL DEFAULT '[]',
                criado_em       TEXT NOT NULL,
                atualizado_em   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documentos (
                id              TEXT PRIMARY KEY,
                sessao_id       TEXT NOT NULL REFERENCES sessoes(id) ON DELETE CASCADE,
                nome            TEXT NOT NULL,
                tipo            TEXT NOT NULL,
                tamanho_bytes   INTEGER NOT NULL DEFAULT 0,
                chunks_indexados INTEGER NOT NULL DEFAULT 0,
                criado_em       TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS documentos_sessao_idx ON documentos(sessao_id)"
        )
    conn.close()


def _agora() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _row_para_sessao(row: sqlite3.Row) -> Sessao:
    return Sessao(
        id=row["id"],
        numero_processo=row["numero_processo"],
        tipo_acao=row["tipo_acao"],
        area=row["area"],
        vara=row["vara"],
        cidade_uf=row["cidade_uf"],
        status=row["status"],
        criado_em=row["criado_em"],
        atualizado_em=row["atualizado_em"],
        partes=json.loads(row["partes"]),
        prazos=json.loads(row["prazos"]),
    )


def _row_para_doc(row: sqlite3.Row) -> DocumentoSessao:
    return DocumentoSessao(
        id=row["id"],
        sessao_id=row["sessao_id"],
        nome=row["nome"],
        tipo=row["tipo"],
        tamanho_bytes=row["tamanho_bytes"],
        chunks_indexados=row["chunks_indexados"],
        criado_em=row["criado_em"],
    )


# ---------------------------------------------------------------------------
# CRUD de sessões
# ---------------------------------------------------------------------------

def criar_sessao(sessao: Sessao) -> Sessao:
    """Persiste uma nova sessão no banco."""
    agora = _agora()
    sessao.criado_em = agora
    sessao.atualizado_em = agora
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT INTO sessoes
               (id, numero_processo, tipo_acao, area, vara, cidade_uf,
                status, partes, prazos, criado_em, atualizado_em)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sessao.id,
                sessao.numero_processo,
                sessao.tipo_acao,
                sessao.area,
                sessao.vara,
                sessao.cidade_uf,
                sessao.status,
                json.dumps(sessao.partes, ensure_ascii=False),
                json.dumps(sessao.prazos, ensure_ascii=False),
                sessao.criado_em,
                sessao.atualizado_em,
            ),
        )
    conn.close()
    return sessao


def obter_sessao(sessao_id: str) -> Sessao | None:
    """Retorna uma sessão pelo ID, ou None se não encontrada."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessoes WHERE id = ?", (sessao_id,)
    ).fetchone()
    conn.close()
    return _row_para_sessao(row) if row else None


def listar_sessoes() -> list[Sessao]:
    """Retorna todas as sessões ordenadas por criação decrescente."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sessoes ORDER BY criado_em DESC"
    ).fetchall()
    conn.close()
    return [_row_para_sessao(r) for r in rows]


def atualizar_sessao(sessao_id: str, campos: dict) -> Sessao | None:
    """Atualiza campos de uma sessão e retorna a versão atualizada."""
    sessao = obter_sessao(sessao_id)
    if sessao is None:
        return None

    for campo, valor in campos.items():
        if hasattr(sessao, campo):
            setattr(sessao, campo, valor)
    sessao.atualizado_em = _agora()

    conn = _get_conn()
    with conn:
        conn.execute(
            """UPDATE sessoes SET
               numero_processo=?, tipo_acao=?, area=?, vara=?, cidade_uf=?,
               status=?, partes=?, prazos=?, atualizado_em=?
               WHERE id=?""",
            (
                sessao.numero_processo,
                sessao.tipo_acao,
                sessao.area,
                sessao.vara,
                sessao.cidade_uf,
                sessao.status,
                json.dumps(sessao.partes, ensure_ascii=False),
                json.dumps(sessao.prazos, ensure_ascii=False),
                sessao.atualizado_em,
                sessao_id,
            ),
        )
    conn.close()
    return sessao


def deletar_sessao(sessao_id: str) -> bool:
    """Remove sessão e seus documentos (ON DELETE CASCADE no SQLite).

    Returns:
        True se a sessão foi removida, False se não existia.
    """
    conn = _get_conn()
    with conn:
        cursor = conn.execute(
            "DELETE FROM sessoes WHERE id = ?", (sessao_id,)
        )
    conn.close()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# CRUD de documentos
# ---------------------------------------------------------------------------

def criar_documento(doc: DocumentoSessao) -> DocumentoSessao:
    """Persiste metadados de um documento indexado."""
    doc.criado_em = _agora()
    conn = _get_conn()
    with conn:
        conn.execute(
            """INSERT INTO documentos
               (id, sessao_id, nome, tipo, tamanho_bytes, chunks_indexados, criado_em)
               VALUES (?,?,?,?,?,?,?)""",
            (
                doc.id,
                doc.sessao_id,
                doc.nome,
                doc.tipo,
                doc.tamanho_bytes,
                doc.chunks_indexados,
                doc.criado_em,
            ),
        )
    conn.close()
    return doc


def listar_documentos(sessao_id: str) -> list[DocumentoSessao]:
    """Retorna todos os documentos de uma sessão."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM documentos WHERE sessao_id = ? ORDER BY criado_em",
        (sessao_id,),
    ).fetchall()
    conn.close()
    return [_row_para_doc(r) for r in rows]


def deletar_documento(doc_id: str) -> bool:
    """Remove um documento pelo ID.

    Returns:
        True se removido, False se não encontrado.
    """
    conn = _get_conn()
    with conn:
        cursor = conn.execute(
            "DELETE FROM documentos WHERE id = ?", (doc_id,)
        )
    conn.close()
    return cursor.rowcount > 0
