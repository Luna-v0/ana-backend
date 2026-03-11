"""Protocol e implementação concreta para provedores de LLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Contrato para provedores de LLM."""

    def invocar(self, prompt: str, temperatura: float = 0.1, **kwargs: Any) -> str: ...


class OllamaLLMProvider:
    """Implementação concreta: Ollama via REST API (httpx)."""

    def __init__(self, modelo: str, host: str = "http://localhost:11434") -> None:
        self.modelo = modelo
        self.host = host

    def invocar(self, prompt: str, temperatura: float = 0.1, **kwargs: Any) -> str:
        import httpx

        with httpx.Client(timeout=60.0) as cliente:
            resposta = cliente.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.modelo,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperatura},
                    **kwargs,
                },
            )
            resposta.raise_for_status()
            return resposta.json().get("response", "")

    def invocar_stream(
        self, prompt: str, temperatura: float = 0.1, **kwargs: Any
    ) -> Iterator[str]:
        """Gera tokens incrementalmente via Ollama streaming.

        Yields:
            Fragmentos de texto conforme o modelo os produz.
        """
        import httpx
        import json

        with httpx.Client(timeout=120.0) as cliente:
            with cliente.stream(
                "POST",
                f"{self.host}/api/generate",
                json={
                    "model": self.modelo,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": temperatura},
                    **kwargs,
                },
            ) as resposta:
                resposta.raise_for_status()
                for linha in resposta.iter_lines():
                    if not linha:
                        continue
                    dados = json.loads(linha)
                    token = dados.get("response", "")
                    if token:
                        yield token
                    if dados.get("done"):
                        break
