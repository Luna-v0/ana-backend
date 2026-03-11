"""Testes do backend FastAPI do sistema ANA.

Valida endpoints de health check, CORS e estrutura de resposta
conforme definido no spec 01.
"""

import pytest
from fastapi.testclient import TestClient

from ana.api.main import criar_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Cria cliente de teste FastAPI para o módulo.

    Returns:
        TestClient configurado com a aplicação ANA.
    """
    app = criar_app()
    return TestClient(app, raise_server_exceptions=True)


class TestEndpointRaiz:
    """Testes para o endpoint raiz (GET /)."""

    def test_raiz_retorna_200(self, client: TestClient):
        """Verifica que a raiz retorna HTTP 200."""
        resposta = client.get("/")
        assert resposta.status_code == 200

    def test_raiz_retorna_nome_sistema(self, client: TestClient):
        """Verifica que a raiz identifica o sistema ANA."""
        resposta = client.get("/")
        dados = resposta.json()
        assert "ANA" in dados["sistema"]

    def test_raiz_referencia_docs(self, client: TestClient):
        """Verifica que a raiz indica o caminho para a documentação."""
        resposta = client.get("/")
        dados = resposta.json()
        assert dados.get("docs") == "/docs"

    def test_raiz_referencia_health(self, client: TestClient):
        """Verifica que a raiz indica o caminho para o health check."""
        resposta = client.get("/")
        dados = resposta.json()
        assert dados.get("health") == "/health"


class TestEndpointHealth:
    """Testes para o endpoint de health check (GET /health)."""

    def test_health_retorna_200(self, client: TestClient):
        """Verifica que o health check retorna HTTP 200."""
        resposta = client.get("/health")
        assert resposta.status_code == 200

    def test_health_retorna_campo_status(self, client: TestClient):
        """Verifica que o health check retorna campo 'status'."""
        resposta = client.get("/health")
        dados = resposta.json()
        assert "status" in dados
        assert dados["status"] in ("ok", "degradado", "offline")

    def test_health_retorna_campo_versao(self, client: TestClient):
        """Verifica que o health check retorna campo 'versao'."""
        resposta = client.get("/health")
        dados = resposta.json()
        assert "versao" in dados
        assert dados["versao"] == "0.1.0"

    def test_health_retorna_lista_servicos(self, client: TestClient):
        """Verifica que o health check retorna lista de serviços."""
        resposta = client.get("/health")
        dados = resposta.json()
        assert "servicos" in dados
        assert isinstance(dados["servicos"], list)
        assert len(dados["servicos"]) >= 2

    def test_health_servicos_tem_campos_corretos(self, client: TestClient):
        """Verifica que cada serviço tem campos 'nome', 'disponivel' e 'url'."""
        resposta = client.get("/health")
        dados = resposta.json()
        for servico in dados["servicos"]:
            assert "nome" in servico
            assert "disponivel" in servico
            assert "url" in servico
            assert isinstance(servico["disponivel"], bool)

    def test_health_inclui_ollama(self, client: TestClient):
        """Verifica que Ollama está na lista de serviços monitorados."""
        resposta = client.get("/health")
        dados = resposta.json()
        nomes = [s["nome"] for s in dados["servicos"]]
        assert "ollama" in nomes

    def test_health_inclui_qdrant(self, client: TestClient):
        """Verifica que Qdrant está na lista de serviços monitorados."""
        resposta = client.get("/health")
        dados = resposta.json()
        nomes = [s["nome"] for s in dados["servicos"]]
        assert "qdrant" in nomes

    def test_health_status_degradado_sem_servicos(self, client: TestClient):
        """Verifica que o status é 'degradado' quando serviços não estão disponíveis."""
        resposta = client.get("/health")
        dados = resposta.json()
        # Em ambiente de teste sem Docker, serviços estarão offline
        todos_disponiveis = all(s["disponivel"] for s in dados["servicos"])
        if not todos_disponiveis:
            assert dados["status"] == "degradado"

    def test_health_retorna_info_modelos(self, client: TestClient):
        """Verifica que o health check retorna informações do perfil de modelos ativo."""
        resposta = client.get("/health")
        dados = resposta.json()
        assert "modelos" in dados
        modelos = dados["modelos"]
        assert "perfil_ativo" in modelos
        assert "vram_estimada_gb" in modelos
        assert "pesquisador" in modelos
        assert "orquestrador" in modelos
        assert "embedding" in modelos

    def test_health_perfil_teste_por_padrao(self, client: TestClient):
        """Verifica que o perfil padrão é 'teste' (para 16GB VRAM)."""
        resposta = client.get("/health")
        dados = resposta.json()
        perfil = dados["modelos"]["perfil_ativo"]
        # Perfil padrão é 'teste' conforme config/modelos.yaml
        assert perfil == "teste"

    def test_health_perfil_teste_sem_modelos_14b(self, client: TestClient):
        """Verifica que o perfil 'teste' não usa modelos 14B."""
        resposta = client.get("/health")
        dados = resposta.json()
        modelos = dados["modelos"]
        assert "14b" not in modelos["pesquisador"]
        assert "14b" not in modelos["orquestrador"]


class TestDocumentacao:
    """Testes para os endpoints de documentação automática."""

    def test_swagger_ui_disponivel(self, client: TestClient):
        """Verifica que a UI do Swagger está acessível."""
        resposta = client.get("/docs")
        assert resposta.status_code == 200

    def test_redoc_disponivel(self, client: TestClient):
        """Verifica que o ReDoc está acessível."""
        resposta = client.get("/redoc")
        assert resposta.status_code == 200

    def test_openapi_json_disponivel(self, client: TestClient):
        """Verifica que o schema OpenAPI JSON está acessível."""
        resposta = client.get("/openapi.json")
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados["info"]["title"] == "ANA — Attorney Normative Assistent"


class TestRagModelos:
    """Testes para os modelos de dados do módulo RAG."""

    def test_chunk_juridico_criacao(self):
        """Verifica que ChunkJuridico é criado corretamente."""
        from ana.rag.modelos import (
            ChunkJuridico,
            MetadataChunkJuridico,
            TipoDocumento,
            AreaJuridica,
            VigenciaStatus,
        )
        chunk = ChunkJuridico(
            texto="Art. 5º Todos são iguais perante a lei.",
            metadata=MetadataChunkJuridico(
                fonte="Constituição Federal/1988",
                tipo=TipoDocumento.LEI_FEDERAL,
                area=AreaJuridica.CONSTITUCIONAL,
                artigo="Art. 5",
                vigencia=VigenciaStatus.ATIVA,
            ),
        )
        assert chunk.texto.startswith("Art. 5")
        assert chunk.metadata.tipo == TipoDocumento.LEI_FEDERAL
        assert chunk.metadata.vigencia == VigenciaStatus.ATIVA
        assert chunk.id is None  # Ainda não indexado
        assert chunk.embedding is None  # Ainda não gerado

    def test_filtros_busca_valores_padrao(self):
        """Verifica valores padrão de FiltrosBusca."""
        from ana.rag.modelos import FiltrosBusca, VigenciaStatus
        filtros = FiltrosBusca()
        # Por padrão, busca apenas legislação ativa (previne citar leis revogadas)
        assert filtros.vigencia == VigenciaStatus.ATIVA
        assert filtros.tipos is None
        assert filtros.sessao_id is None


class TestAgentesEstado:
    """Testes para o estado compartilhado dos agentes."""

    def test_intencoes_validas_completas(self):
        """Verifica que todas as intenções do spec 09 estão definidas."""
        from ana.agentes.estado import INTENCOES_VALIDAS
        intencoes_esperadas = {
            "pesquisa_legal",
            "analise_processo",
            "gerar_documento",
            "transcrever_audio",
            "verificar_prazo",
            "buscar_similar",
            "desconhecida",
        }
        assert intencoes_esperadas.issubset(INTENCOES_VALIDAS)

    def test_estado_juridico_tipado(self):
        """Verifica que EstadoJuridico aceita os campos esperados."""
        from ana.agentes.estado import EstadoJuridico
        estado: EstadoJuridico = {
            "mensagem_usuario": "Qual o prazo para contestação no CPC?",
            "sessao_id": "sessao-001",
            "intencao": "pesquisa_legal",
            "contexto_rag": [],
            "resposta": None,
            "validacao": None,
            "documentos_gerados": [],
            "erro": None,
        }
        assert estado["intencao"] == "pesquisa_legal"
        assert estado["sessao_id"] == "sessao-001"
