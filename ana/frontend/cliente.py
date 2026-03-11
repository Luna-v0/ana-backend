"""Cliente HTTP para comunicação com o backend FastAPI do sistema ANA.

Encapsula todas as chamadas ao backend, tratando erros de conexão
e retornando respostas estruturadas para o frontend Gradio.

Nota (LGPD):
    Todas as requisições são para localhost. Dados nunca saem da máquina.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from loguru import logger


# Timeout padrão para requisições (embeddings podem demorar na primeira chamada)
_TIMEOUT_PADRAO = httpx.Timeout(120.0, connect=5.0)


@dataclass
class RespostaCliente:
    """Resultado padronizado de uma chamada ao backend.

    Attributes:
        sucesso: True se a requisição foi bem-sucedida.
        dados: Dados retornados pelo backend (se sucesso).
        erro: Mensagem de erro (se falha).
        codigo_http: Código de status HTTP recebido.
    """

    sucesso: bool
    dados: Any = None
    erro: str = ""
    codigo_http: int = 0


class ClienteANA:
    """Cliente HTTP síncrono para o backend FastAPI do sistema ANA.

    Attributes:
        url_base: URL base do backend (ex: 'http://localhost:8000').
    """

    def __init__(self, url_base: str = "http://localhost:8000") -> None:
        """Inicializa o cliente com a URL base do backend.

        Args:
            url_base: URL base do backend FastAPI.
        """
        self.url_base = url_base.rstrip("/")

    def _get(self, caminho: str) -> RespostaCliente:
        """Executa requisição GET ao backend.

        Args:
            caminho: Caminho relativo do endpoint (ex: '/health').

        Returns:
            RespostaCliente com dados ou erro.
        """
        url = f"{self.url_base}{caminho}"
        try:
            with httpx.Client(timeout=_TIMEOUT_PADRAO) as cliente:
                resp = cliente.get(url)
                resp.raise_for_status()
                return RespostaCliente(sucesso=True, dados=resp.json(), codigo_http=resp.status_code)
        except httpx.ConnectError:
            msg = f"Backend offline — inicie com: uv run uvicorn ana.api.main:app --reload"
            logger.warning(msg)
            return RespostaCliente(sucesso=False, erro=msg)
        except httpx.HTTPStatusError as e:
            msg = f"Erro HTTP {e.response.status_code}: {e.response.text[:200]}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg, codigo_http=e.response.status_code)
        except Exception as e:
            msg = f"Erro inesperado: {type(e).__name__}: {e}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg)

    def _post(self, caminho: str, payload: dict) -> RespostaCliente:
        """Executa requisição POST ao backend.

        Args:
            caminho: Caminho relativo do endpoint.
            payload: Dicionário a enviar como JSON.

        Returns:
            RespostaCliente com dados ou erro.
        """
        url = f"{self.url_base}{caminho}"
        try:
            with httpx.Client(timeout=_TIMEOUT_PADRAO) as cliente:
                resp = cliente.post(url, json=payload)
                resp.raise_for_status()
                return RespostaCliente(sucesso=True, dados=resp.json(), codigo_http=resp.status_code)
        except httpx.ConnectError:
            msg = "Backend offline — inicie com: uv run uvicorn ana.api.main:app --reload"
            logger.warning(msg)
            return RespostaCliente(sucesso=False, erro=msg)
        except httpx.HTTPStatusError as e:
            try:
                detalhe = e.response.json().get("detail", e.response.text[:300])
            except Exception:
                detalhe = e.response.text[:300]
            msg = f"Erro {e.response.status_code}: {detalhe}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg, codigo_http=e.response.status_code)
        except Exception as e:
            msg = f"Erro inesperado: {type(e).__name__}: {e}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg)

    def health(self) -> RespostaCliente:
        """Verifica saúde do backend e serviços de infraestrutura.

        Returns:
            RespostaCliente com dados de saúde de todos os serviços.
        """
        return self._get("/health")

    def status_rag(self) -> RespostaCliente:
        """Retorna status do pipeline RAG (Qdrant, BM25, modelos).

        Returns:
            RespostaCliente com status do pipeline RAG.
        """
        return self._get("/rag/status")

    def ingerir(
        self,
        texto: str,
        fonte: str,
        tipo: str = "lei_federal",
        area: Optional[str] = None,
        vigencia: str = "ativa",
        orgao: Optional[str] = None,
        url_origem: Optional[str] = None,
    ) -> RespostaCliente:
        """Ingere documento jurídico no pipeline RAG.

        Args:
            texto: Texto completo do documento.
            fonte: Identificação da fonte.
            tipo: Tipo do documento jurídico.
            area: Área do direito (opcional).
            vigencia: Status de vigência.
            orgao: Órgão emissor (opcional).
            url_origem: URL da fonte (opcional).

        Returns:
            RespostaCliente com chunks gerados e indexados.
        """
        payload: dict = {
            "texto": texto,
            "fonte": fonte,
            "tipo": tipo,
            "vigencia": vigencia,
        }
        if area:
            payload["area"] = area
        if orgao:
            payload["orgao"] = orgao
        if url_origem:
            payload["url_origem"] = url_origem

        return self._post("/rag/ingerir", payload)

    def buscar(
        self,
        query: str,
        area: Optional[str] = None,
        top_k: int = 5,
        usar_reranker: bool = True,
        usar_mmr: bool = True,
        vigencia: Optional[str] = "ativa",
    ) -> RespostaCliente:
        """Executa busca híbrida na legislação indexada.

        Args:
            query: Consulta jurídica em linguagem natural.
            area: Filtro por área do direito (opcional).
            top_k: Número máximo de resultados.
            usar_reranker: Ativa reranking com CrossEncoder.
            usar_mmr: Ativa diversidade com MMR.
            vigencia: Status de vigência a filtrar ('ativa', 'revogada', ou None=todas).

        Returns:
            RespostaCliente com lista de chunks relevantes.
        """
        filtros: dict = {}
        if vigencia:
            filtros["vigencia"] = vigencia
        if area and area != "Todas":
            filtros["areas"] = [area]

        payload = {
            "query": query,
            "filtros": filtros,
            "top_k": top_k,
            "usar_reranker": usar_reranker,
            "usar_mmr": usar_mmr,
        }
        return self._post("/rag/buscar", payload)

    def status_transcricao(self) -> RespostaCliente:
        """Retorna disponibilidade do módulo de transcrição (whisperx/pyannote).

        Returns:
            RespostaCliente com status, dispositivo e mensagem de instalação.
        """
        return self._get("/transcricao/status")

    def transcrever(
        self,
        caminho_audio: str,
        numero_processo: str = "",
        data_audiencia: str = "",
        tipo_audiencia: str = "Instrução e Julgamento",
        vara: str = "",
        cidade_uf: str = "",
        min_speakers: int = 2,
        max_speakers: int = 6,
        identificar_por_llm: bool = True,
    ) -> RespostaCliente:
        """Envia arquivo de áudio para transcrição com diarização.

        Args:
            caminho_audio: Caminho local para o arquivo de áudio.
            numero_processo: Número CNJ do processo (opcional).
            data_audiencia: Data da audiência dd/mm/aaaa (opcional).
            tipo_audiencia: Tipo da audiência.
            vara: Vara e foro da audiência (opcional).
            cidade_uf: Cidade e UF (opcional).
            min_speakers: Mínimo de falantes esperados.
            max_speakers: Máximo de falantes esperados.
            identificar_por_llm: Usa LLM para identificar speakers por contexto.

        Returns:
            RespostaCliente com markdown da transcrição e metadados.
        """
        import os
        url = f"{self.url_base}/transcricao/transcrever"
        try:
            with httpx.Client(timeout=_TIMEOUT_PADRAO) as cliente:
                with open(caminho_audio, "rb") as f:
                    nome_arquivo = os.path.basename(caminho_audio)
                    resp = cliente.post(
                        url,
                        files={"audio": (nome_arquivo, f, "audio/mpeg")},
                        data={
                            "numero_processo": numero_processo,
                            "data_audiencia": data_audiencia,
                            "tipo_audiencia": tipo_audiencia,
                            "vara": vara,
                            "cidade_uf": cidade_uf,
                            "min_speakers": str(min_speakers),
                            "max_speakers": str(max_speakers),
                            "identificar_por_llm": "true" if identificar_por_llm else "false",
                        },
                    )
                    resp.raise_for_status()
                    return RespostaCliente(sucesso=True, dados=resp.json(), codigo_http=resp.status_code)
        except httpx.ConnectError:
            msg = "Backend offline — inicie com: uv run uvicorn ana.api.main:app --reload"
            logger.warning(msg)
            return RespostaCliente(sucesso=False, erro=msg)
        except httpx.HTTPStatusError as e:
            try:
                detalhe = e.response.json().get("detail", e.response.text[:300])
            except Exception:
                detalhe = e.response.text[:300]
            msg = f"Erro {e.response.status_code}: {detalhe}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg, codigo_http=e.response.status_code)
        except Exception as e:
            msg = f"Erro inesperado: {type(e).__name__}: {e}"
            logger.error(msg)
            return RespostaCliente(sucesso=False, erro=msg)

    def status_scrapers(self) -> RespostaCliente:
        """Retorna status do pipeline de scrapers e cache.

        Returns:
            RespostaCliente com fontes disponíveis e documentos no cache.
        """
        return self._get("/scrapers/status")

    def listar_fontes_scrapers(self) -> RespostaCliente:
        """Lista as fontes de scraping configuradas.

        Returns:
            RespostaCliente com lista de fontes disponíveis.
        """
        return self._get("/scrapers/fontes")

    def coletar_fonte(self, fonte: str) -> RespostaCliente:
        """Dispara coleta completa de uma fonte em background.

        Args:
            fonte: Nome da fonte (planalto, lexml, stf, stj).

        Returns:
            RespostaCliente confirmando agendamento da coleta.
        """
        return self._post("/scrapers/coletar", {"fonte": fonte})

    def atualizar_tudo(self) -> RespostaCliente:
        """Dispara atualização incremental de todas as fontes em background.

        Returns:
            RespostaCliente confirmando agendamento.
        """
        return self._post("/scrapers/atualizar-tudo", {})

    def reformular(self, texto: str, instrucoes: str = "") -> RespostaCliente:
        """Reformula texto em português jurídico brasileiro formal.

        Args:
            texto: Texto a ser reformulado.
            instrucoes: Instruções adicionais opcionais para guiar a reescrita.

        Returns:
            RespostaCliente com texto_reformulado e modelo_usado.
        """
        payload: dict = {"texto": texto}
        if instrucoes:
            payload["instrucoes"] = instrucoes
        return self._post("/redacao/reformular", payload)
