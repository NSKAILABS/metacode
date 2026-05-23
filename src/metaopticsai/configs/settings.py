"""Centralized Pydantic Settings — single import point: `from metaopticsai.configs.settings import settings`.

All groups are independently loadable from environment variables and `.env`.
The Settings composite object is built once at import time; never mutate.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root from this file's location: configs/settings.py → root
# is 4 levels up: configs → metaopticsai → src → metaopticsai (the repo).
_ROOT = Path(__file__).resolve().parents[3]
_ENV = str(_ROOT / ".env")


# ── 1. LLM provider configuration ────────────────────────────────────────────

class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=_ENV, extra="ignore")

    default_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    ollama_embed_model: str = "nomic-embed-text"

    groq_model: str = "llama-3.1-70b-versatile"
    groq_api_key: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-7"

    temperature: float = 0.1
    num_predict: int = 1024


# ── 2. RCWA simulation defaults ──────────────────────────────────────────────

class RCWAConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RCWA_", env_file=_ENV, extra="ignore")

    harmonics: int = 3
    resolution: int = 64
    minibatch_size: int = 20


# ── 3. Self-RAG workflow ─────────────────────────────────────────────────────

class WorkflowConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKFLOW_", env_file=_ENV, extra="ignore")

    max_rag_retries: int = 3
    max_optimize_iters: int = 10
    fom_target: float = 0.85
    doc_grade_cutoff: float = 0.6


# ── 4. MCP server ────────────────────────────────────────────────────────────

class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_file=_ENV, extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8765
    heavy_jobs_enabled: bool = True


# ── 5. Paths ─────────────────────────────────────────────────────────────────

class PathsConfig:
    """Not a Pydantic model — paths are deterministic, derived from project root."""

    def __init__(self):
        self.project_root: Path = _ROOT
        self.output_dir: Path = _ROOT / "outputs"
        self.papers_dir: Path = _ROOT / "papers"
        self.corpus_dir: Path = (
            _ROOT / "src" / "metaopticsai" / "rag" / "corpus"
        )
        self.material_dir: Path = (
            _ROOT / "src" / "metaopticsai" / "vendor" / "metabox3" / "material_data"
        )
        # Ensure outputs exists; everything else is read-only.
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ── Composite ────────────────────────────────────────────────────────────────

class Settings:
    """Composite holder. Import the module-level `settings` singleton."""

    def __init__(self):
        self.llm = LLMConfig()
        self.rcwa = RCWAConfig()
        self.workflow = WorkflowConfig()
        self.server = ServerConfig()
        self.paths = PathsConfig()

        # Reasonable defaults for TF noise — set early.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


settings = Settings()