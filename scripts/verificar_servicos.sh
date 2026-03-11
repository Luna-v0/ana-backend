#!/bin/bash
# Verifica disponibilidade de todos os serviços da infraestrutura ANA
# Equivalente ao checklist do spec 10, seção 10.3

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
BACKEND_HOST="${BACKEND_HOST:-localhost}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

verde='\033[0;32m'
vermelho='\033[0;31m'
amarelo='\033[1;33m'
reset='\033[0m'

ok() { echo -e "${verde}[OK]${reset} $1"; }
erro() { echo -e "${vermelho}[ERRO]${reset} $1"; }
aviso() { echo -e "${amarelo}[AVISO]${reset} $1"; }

echo "=== ANA — Verificação de Serviços ==="
echo ""

# Verificar Ollama
echo "Verificando Ollama ($OLLAMA_HOST)..."
if curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    MODELOS=$(curl -s "${OLLAMA_HOST}/api/tags" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models',[])) or 'nenhum')")
    ok "Ollama disponível — Modelos: $MODELOS"
else
    erro "Ollama não disponível em $OLLAMA_HOST"
    aviso "Execute: docker compose up -d ollama"
fi

# Verificar Qdrant
echo "Verificando Qdrant ($QDRANT_HOST:$QDRANT_PORT)..."
if curl -sf "http://${QDRANT_HOST}:${QDRANT_PORT}/readyz" > /dev/null 2>&1; then
    COLECOES=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(', '.join(c['name'] for c in d.get('result',{}).get('collections',[])) or 'nenhuma')")
    ok "Qdrant disponível — Collections: $COLECOES"
else
    erro "Qdrant não disponível em $QDRANT_HOST:$QDRANT_PORT"
    aviso "Execute: docker compose up -d qdrant"
fi

# Verificar Backend ANA
echo "Verificando Backend ANA ($BACKEND_HOST:$BACKEND_PORT)..."
if curl -sf "http://${BACKEND_HOST}:${BACKEND_PORT}/health" > /dev/null 2>&1; then
    STATUS=$(curl -s "http://${BACKEND_HOST}:${BACKEND_PORT}/health" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('status','desconhecido'))")
    ok "Backend disponível — Status: $STATUS"
else
    erro "Backend ANA não disponível em $BACKEND_HOST:$BACKEND_PORT"
    aviso "Execute: uv run uvicorn ana.api.main:app --reload"
fi

echo ""
echo "=== Verificação concluída ==="
