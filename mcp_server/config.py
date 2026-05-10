"""
mcp_server.config — centralized configuration.

All knobs are read from environment variables (with `.env` autoloaded by
python-dotenv). Defaults are tuned for a fully local, Ollama-first deploy:
nothing in this module reaches the public internet unless the user explicitly
sets `ANTHROPIC_API_KEY` or points to a remote Ollama host.

Use:
    from mcp_server.config import settings
    settings.OLLAMA_BASE_URL        # 'http://localhost:11434'
    settings.OLLAMA_MODEL           # e.g. 'deepseek-r1:7b'
    settings.OUTPUT_DIR             # Path
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    OLLAMA_BASE_URL: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL",
                                                              "http://localhost:11434"))
    OLLAMA_MODEL: str = field(default_factory=lambda: _env("OLLAMA_MODEL",
                                                           "deepseek-r1:7b"))
    OLLAMA_EMBED_MODEL: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL",
                                                                 "nomic-embed-text"))
    OLLAMA_TEMPERATURE: float = field(default_factory=lambda: float(_env("OLLAMA_TEMPERATURE", "0.1")))
    OLLAMA_NUM_PREDICT: int = field(default_factory=lambda: int(_env("OLLAMA_NUM_PREDICT", "1024")))

    ANTHROPIC_API_KEY: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_MODEL: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-7"))

    PROJECT_ROOT: Path = field(default_factory=lambda: _ROOT)
    OUTPUT_DIR: Path = field(default_factory=lambda: _env_path(
        "METAOPTICS_OUTPUT_DIR", _ROOT / "outputs"))
    RESOURCE_DIR: Path = field(default_factory=lambda: _env_path(
        "METAOPTICS_RESOURCE_DIR", _ROOT / "mcp_server" / "resources"))
    PAPERS_DIR: Path = field(default_factory=lambda: _env_path(
        "METAOPTICS_PAPERS_DIR", _ROOT / "papers"))
    METABOX_MATERIAL_DIR: Path = field(default_factory=lambda: _env_path(
        "METAOPTICS_MATERIAL_DIR", _ROOT / "metabox3" / "material_data"))

    SERVER_HOST: str = field(default_factory=lambda: _env("MCP_HOST", "127.0.0.1"))
    SERVER_PORT: int = field(default_factory=lambda: int(_env("MCP_PORT", "8765")))

    DEFAULT_HARMONICS: int = field(default_factory=lambda: int(_env("RCWA_HARMONICS", "3")))
    DEFAULT_RESOLUTION: int = field(default_factory=lambda: int(_env("RCWA_RESOLUTION", "64")))
    DEFAULT_MINIBATCH: int = field(default_factory=lambda: int(_env("RCWA_MINIBATCH", "20")))
    HEAVY_JOBS_ENABLED: bool = field(default_factory=lambda: _env_bool("HEAVY_JOBS", True))

    TF_CPP_MIN_LOG_LEVEL: str = field(default_factory=lambda: _env("TF_CPP_MIN_LOG_LEVEL", "3"))


settings = Settings()

settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", settings.TF_CPP_MIN_LOG_LEVEL)