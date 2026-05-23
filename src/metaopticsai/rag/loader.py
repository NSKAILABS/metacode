"""Load the metalens-design knowledge corpus from on-disk markdown files."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_corpus(corpus_dir: Path | str) -> dict[str, str]:
    """Return {filename_stem: content} for every .md file under corpus_dir.

    Sorted by filename so numbered files (01_*, 02_*, ...) load in order.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        log.warning("Corpus dir %s does not exist — returning empty corpus.", corpus_dir)
        return {}

    docs: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        try:
            docs[path.stem] = path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Skipping %s: %s", path, e)
    log.info("Loaded %d corpus documents from %s", len(docs), corpus_dir)
    return docs