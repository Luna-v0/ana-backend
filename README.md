# ana-backend

Backend do **ANA — Attorney Normative Assistent**, plataforma de assistência jurídica para advogados brasileiros, executada majoritariamente em ambiente local para preservar a confidencialidade dos dados de clientes.

> Para a arquitetura completa do ecossistema, ver [`ana-arch`](https://github.com/Luna-v0/ana-arch).

## Capacidades

- **RAG híbrido** (semântico + sintático) sobre legislação brasileira
- **Transcrição** de audiências com identificação de participantes (WhisperX)
- **Geração e edição** de documentos Word com formatação jurídica
- **Sessões isoladas** por caso, com RAG dedicado
- **Validação automática** de citações legais para mitigar alucinações
- **Orquestração de agentes** modular via LangGraph

## Stack

| Camada | Tecnologia |
|--------|------------|
| API | FastAPI |
| LLM | Ollama |
| Vector DB | Qdrant |
| Transcrição | WhisperX |
| Orquestração | LangGraph |
| Checkpointing | SQLite (`AsyncSqliteSaver`) |
| Gerenciador | uv |

## Layout

```
ana/        # código da aplicação (rotas, agentes, RAG, sessões)
config/     # configuração de modelos e prompts
data/       # bases locais (sqlite, índices)
scripts/    # utilitários (indexação, manutenção)
tests/      # testes de integração (LangGraph end-to-end)
notebooks/  # exploração e protótipos
```

## Desenvolvimento

```bash
uv sync                              # instala dependências
uv run uvicorn ana.main:app --reload # API em http://localhost:8000
uv run pytest                        # testes
```

A documentação detalhada está em `docs/` (português).
