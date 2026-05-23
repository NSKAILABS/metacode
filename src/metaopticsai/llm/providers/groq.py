"""Groq provider — cloud, fast Llama-style models with native tool calling."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from metaopticsai.configs.settings import settings
from metaopticsai.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    name = "groq"
    supports_tools = True

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm.groq_model
        self.api_key = api_key or settings.llm.groq_api_key

    def chat_model(self, **overrides: Any) -> BaseChatModel:
        return ChatGroq(
            model=overrides.get("model", self.model),
            api_key=self.api_key,
            temperature=overrides.get("temperature", settings.llm.temperature),
        )

    def is_available(self) -> bool:
        return bool(self.api_key)