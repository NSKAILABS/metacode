"""Provider registry — pick an LLM by name without hard-importing every dep."""
from __future__ import annotations

from typing import Callable

from metaopticsai.configs.settings import settings
from metaopticsai.llm.base import LLMProvider

# Lazy-construction map so missing optional deps don't break imports.
_BUILDERS: dict[str, Callable[..., LLMProvider]] = {}


def _register():
    """Populate builders, swallowing ImportError per provider."""
    if not _BUILDERS:
        try:
            from metaopticsai.llm.providers.ollama import OllamaProvider
            _BUILDERS["ollama"] = OllamaProvider
        except ImportError:
            pass
        try:
            from metaopticsai.llm.providers.groq import GroqProvider
            _BUILDERS["groq"] = GroqProvider
        except ImportError:
            pass
        try:
            from metaopticsai.llm.providers.anthropic import AnthropicProvider
            _BUILDERS["anthropic"] = AnthropicProvider
        except ImportError:
            pass


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """Return an LLMProvider by name (default = settings.llm.default_provider)."""
    _register()
    n = (name or settings.llm.default_provider).lower()
    if n not in _BUILDERS:
        raise ValueError(
            f"Unknown LLM provider {n!r}. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[n](**kwargs)


def available_providers() -> list[str]:
    _register()
    return sorted(_BUILDERS)