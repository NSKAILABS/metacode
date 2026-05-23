"""Ollama provider — local LLM + local embeddings."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from metaopticsai.configs.settings import settings
from metaopticsai.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ):
        self.model = model or settings.llm.ollama_model
        self.base_url = base_url or settings.llm.ollama_base_url
        self.temperature = (
            temperature if temperature is not None else settings.llm.temperature
        )

    @property
    def supports_embeddings(self) -> bool:
        return True

    def chat_model(self, **overrides: Any) -> BaseChatModel:
        return ChatOllama(
            model=overrides.get("model", self.model),
            base_url=self.base_url,
            temperature=overrides.get("temperature", self.temperature),
            num_predict=overrides.get("num_predict", settings.llm.num_predict),
        )

    def embeddings(self, **overrides: Any) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=overrides.get("model", settings.llm.ollama_embed_model),
            base_url=self.base_url,
        )

    def is_available(self) -> bool:
        import requests
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False