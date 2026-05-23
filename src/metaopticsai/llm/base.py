"""LLM provider abstraction. Returns LangChain-compatible runnables so
LangGraph nodes can stay provider-agnostic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(ABC):
    """Wraps a chat-style LLM with a uniform API."""

    name: str
    model: str

    @abstractmethod
    def chat_model(self, **overrides: Any) -> BaseChatModel:
        """Return a LangChain ChatModel — usable in LCEL chains."""

    @abstractmethod
    def is_available(self) -> bool: ...

    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports native tool/function calling."""
        return False

    @property
    def supports_embeddings(self) -> bool:
        """Whether this provider also exposes an embeddings model."""
        return False

    def embeddings(self, **overrides: Any):
        """Return a LangChain Embeddings model. Default: raise."""
        raise NotImplementedError(
            f"{self.name!r} provider does not support embeddings."
        )