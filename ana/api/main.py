"""Aplicação principal FastAPI do sistema ANA.

Configura a instância FastAPI com middlewares, routers e eventos
de ciclo de vida (startup/shutdown) para o sistema ANA.

Exemplo de uso:
    $ uv run uvicorn ana.api.main:app --reload --host 0.0.0.0 --port 8000
    $ uv run ana-api
"""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ana.config import obter_configuracao
from ana.api.routers import health, rag, transcricao, scrapers, redacao, sessoes, gpu, chat, documentos
from ana.scrapers.agendador import AgendadorScrapers
from ana.gpu import obter_gestor
from ana.gpu.registro import registrar_modelos

# Instância global do agendador (iniciado no lifespan)
_agendador = AgendadorScrapers()


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Executado na inicialização e encerramento do servidor.
    Loga configuração ativa e libera recursos ao encerrar.

    Args:
        app: Instância da aplicação FastAPI.

    Yields:
        Controla o contexto de vida da aplicação.
    """
    config = obter_configuracao()
    logger.info("Iniciando ANA — Attorney Normative Assistent v0.1.0")
    logger.info(f"Ollama: {config.ollama_host}")
    logger.info(f"PostgreSQL: {config.postgres_dsn}")
    logger.info(f"Debug: {config.debug}")
    _agendador.iniciar()

    # Registra modelos GPU no gestor (sem carregá-los ainda — lazy)
    registrar_modelos(obter_gestor())

    # Garante que a extensão pgvector e as tabelas necessárias existem
    try:
        from ana.storage.pgvector_store import IndexadorPgVector
        indexador = IndexadorPgVector()
        await asyncio.to_thread(indexador.criar_colecao_legislacao)
        await asyncio.to_thread(indexador.criar_colecao_processos)
        logger.info("Tabelas pgvector verificadas/criadas")
    except Exception as e:
        logger.warning(f"Falha ao criar tabelas pgvector no startup: {e}")

    # Inicializa índice BM25 a partir da tabela de legislação no PostgreSQL
    try:
        from ana.rag.retrieval import obter_pipeline_retrieval
        pipeline = obter_pipeline_retrieval()
        await asyncio.to_thread(pipeline.inicializar_bm25)
    except Exception as e:
        logger.warning(f"BM25 não inicializado no startup: {e}")

    logger.info("API pronta para receber requisições")
    yield
    _agendador.parar()
    await obter_gestor().descarregar_tudo()
    logger.info("Encerrando ANA — limpando recursos")


def criar_app() -> FastAPI:
    """Cria e configura a instância da aplicação FastAPI.

    Registra middlewares de CORS, inclui os routers de cada domínio
    e configura o gerenciador de ciclo de vida (lifespan).

    Returns:
        Instância configurada da aplicação FastAPI.
    """
    config = obter_configuracao()

    app = FastAPI(
        title="ANA — Attorney Normative Assistent",
        description=(
            "Plataforma de assistência jurídica com IA local para advogados brasileiros. "
            "Todos os dados de processos são processados localmente conforme a LGPD."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=ciclo_de_vida,
        contact={
            "name": "ANA Team",
        },
        license_info={
            "name": "MIT",
        },
    )

    # --- CORS: apenas origens locais (LGPD: dados não saem da máquina) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(rag.router)
    app.include_router(transcricao.router)
    app.include_router(scrapers.router)
    app.include_router(redacao.router)
    app.include_router(documentos.router)
    app.include_router(sessoes.router)
    app.include_router(gpu.router)

    # --- Endpoint raiz ---
    @app.get("/", include_in_schema=False)
    async def raiz() -> dict:
        """Retorna mensagem de boas-vindas e link para documentação."""
        return {
            "sistema": "ANA — Attorney Normative Assistent",
            "versao": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Instância global da aplicação
app = criar_app()


def iniciar_servidor() -> None:
    """Inicia o servidor uvicorn (entry point do script ana-api).

    Utiliza as configurações de host e porta definidas nas variáveis
    de ambiente ou valores padrão do config.
    """
    config = obter_configuracao()
    uvicorn.run(
        "ana.api.main:app",
        host=config.backend_host,
        port=config.backend_port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    iniciar_servidor()
