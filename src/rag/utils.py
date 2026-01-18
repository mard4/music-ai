import logging
from pathlib import Path

logger = logging.getLogger(__name__)
for lib in ["httpcore", "httpx", "clap", "transformers", "pymongo", "numba", "qdrant_client"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def read_prompt(filename: str) -> str:
    path = Path(__file__).parent / "prompts" / filename
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Impossibile leggere il prompt {filename}: {e}")
        return ""
