"""Configuração central do sistema ANA.

O sistema detecta automaticamente se está rodando dentro de um
container Docker (via /.dockerenv) e ajusta os hostnames para
os nomes dos serviços do docker-compose.

Ambientes:
    Local (fora do Docker):  Ollama em localhost:11434, PostgreSQL em localhost:5432
    Docker (compose):        Ollama em ollama:11434,    PostgreSQL em postgres:5432

A variável de ambiente DATABASE_URL sobrescreve o DSN padrão quando presente.
Para tokens de acesso a serviços externos (HuggingFace, etc.),
use o arquivo .env (veja .env.example).

Configuração de modelos LLM → config/modelos.yaml
Tokens de acesso            → .env

Exemplo de uso:
    >>> from ana.config import obter_configuracao
    >>> config = obter_configuracao()
    >>> config.ollama_host
    'http://localhost:11434'  # local
    'http://ollama:11434'     # dentro do Docker
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _em_docker() -> bool:
    """Detecta automaticamente se está rodando dentro de um container Docker.

    O Docker cria o arquivo /.dockerenv em todos os containers ao iniciar.
    Não requer variáveis de ambiente — zero configuração.

    Returns:
        True se executando dentro de um container Docker.
    """
    return os.path.exists("/.dockerenv")


@dataclass
class ConfiguracaoANA:
    """Configuração central do sistema Attorney Normative Assistent.

    Os hostnames de infraestrutura são ajustados automaticamente conforme
    o ambiente (local vs Docker). Edite os demais valores diretamente aqui.

    Attributes:
        ollama_host: URL completa do serviço Ollama.
        postgres_dsn: DSN de conexão PostgreSQL + pgvector.
            Lido de DATABASE_URL (env var) se presente; caso contrário,
            usa postgres:5432 (Docker) ou localhost:5432 (local).
        colecao_legislacao: Nome da tabela global de legislação.
        prefixo_colecao_sessao: Prefixo para tabelas de sessão por processo.
        backend_host: Host de bind da API FastAPI.
        backend_port: Porta da API FastAPI.
        cors_origins: Origens CORS permitidas (somente localhost, por LGPD).
        log_level: Nível de log do sistema.
        debug: Ativa modo debug com reload automático e logs detalhados.
    """

    # --- Ollama (LLM local) ---
    # Docker: http://ollama:11434 | Local: http://localhost:11434
    ollama_host: str = field(
        default_factory=lambda: (
            "http://ollama:11434" if _em_docker() else "http://localhost:11434"
        )
    )

    # --- PostgreSQL + pgvector (backend padrão) ---
    # DATABASE_URL env var sobrescreve o padrão quando presente
    postgres_dsn: str = field(
        default_factory=lambda: (
            os.environ.get("DATABASE_URL")
            or (
                "postgresql://ana:ana@postgres:5432/ana"
                if _em_docker()
                else "postgresql://ana:ana@localhost:5432/ana"
            )
        )
    )

    # --- Collections do Qdrant / tabelas pgvector ---
    colecao_legislacao: str = "legislacao_brasileira"
    prefixo_colecao_sessao: str = "sessao"

    # --- Servidor FastAPI ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:7860",   # Frontend Docker (porta exposta padrão)
        "http://localhost:3000",   # Alt dev
        "http://localhost:8080",   # Alt dev
        "http://127.0.0.1:5173",
        "http://127.0.0.1:7860",
    ])

    # --- Logging ---
    log_level: str = "INFO"
    debug: bool = False


@lru_cache(maxsize=1)
def obter_configuracao() -> ConfiguracaoANA:
    """Retorna instância singleton da configuração (com cache).

    Returns:
        Instância única de ConfiguracaoANA para o ciclo de vida da aplicação.
    """
    return ConfiguracaoANA()


# Instância global de conveniência
configuracao = obter_configuracao()
