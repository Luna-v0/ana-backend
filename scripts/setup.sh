#!/bin/bash
# Setup inicial do sistema ANA
# Executa as etapas de configuração descritas no spec 10, seção 10.3

set -euo pipefail

verde='\033[0;32m'
amarelo='\033[1;33m'
reset='\033[0m'

ok() { echo -e "${verde}[OK]${reset} $1"; }
passo() { echo -e "${amarelo}[PASSO]${reset} $1"; }

echo "=== ANA — Setup Inicial ==="
echo ""

# 1. Verificar uv
passo "1. Verificando uv..."
if ! command -v uv &> /dev/null; then
    echo "uv não encontrado. Instale em: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
ok "uv $(uv --version) encontrado"

# 2. Copiar .env se não existir
passo "2. Configurando variáveis de ambiente..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    ok ".env criado a partir de .env.example"
    echo "   IMPORTANTE: Edite o arquivo .env antes de continuar"
else
    ok ".env já existe"
fi

# 3. Sincronizar dependências
passo "3. Instalando dependências Python com uv..."
uv sync
ok "Dependências instaladas"

# 4. Subir infraestrutura Docker
passo "4. Iniciando serviços Docker (Ollama + Qdrant)..."
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    docker compose up -d ollama qdrant
    ok "Serviços Docker iniciados"
    echo "   Aguardando serviços ficarem prontos..."
    sleep 10
else
    echo "   Docker/Docker Compose não encontrado — serviços precisam ser iniciados manualmente"
fi

# 5. Baixar modelos Ollama
passo "5. Baixando modelos Ollama..."
echo "   Isso pode demorar dependendo da sua conexão..."
if docker compose ps ollama 2>/dev/null | grep -q "running"; then
    docker exec ana-ollama ollama pull qwen2.5:3b && ok "qwen2.5:3b baixado"
    docker exec ana-ollama ollama pull qwen2.5:7b && ok "qwen2.5:7b baixado"
    echo "   Para o modelo completo (14B), execute:"
    echo "   docker exec ana-ollama ollama pull qwen2.5:14b"
else
    echo "   Ollama não está em execução — baixe os modelos manualmente:"
    echo "   ollama pull qwen2.5:3b"
    echo "   ollama pull qwen2.5:7b"
    echo "   ollama pull qwen2.5:14b"
fi

echo ""
echo "=== Setup concluído! ==="
echo ""
echo "Para iniciar o backend:"
echo "   uv run uvicorn ana.api.main:app --reload"
echo ""
echo "Para verificar serviços:"
echo "   bash scripts/verificar_servicos.sh"
echo ""
echo "Documentação da API:"
echo "   http://localhost:8000/docs"
