"""Anthropic provider — Claude models, native tool calling."""
from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from metaopticsai.configs.settings import settings
from metaopticsai.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_tools = True

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm.anthropic_model
        self.api_key = api_key or settings.llm.anthropic_api_key

    def chat_model(self, **overrides: Any) -> BaseChatModel:
        return ChatAnthropic(
            model=overrides.get("model", self.model),
            api_key=self.api_key,
            temperature=overrides.get("temperature", settings.llm.temperature),
            max_tokens=overrides.get("max_tokens", settings.llm.num_predict),
        )

    def is_available(self) -> bool:
        return bool(self.api_key)