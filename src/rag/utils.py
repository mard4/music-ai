import logging
from pathlib import Path

logger = logging.getLogger(__name__)
for lib in ["httpcore", "httpx", "clap", "transformers", "pymongo", "numba", "qdrant_client"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def read_prompt(filename: str) -> str:
    """
    Legge un file prompt dalla cartella 'rag/prompts' in modo sicuro.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        prompt_path = current_dir / "prompts" / filename

        if not prompt_path.exists():
            logger.error(f"Prompt file not found at: {prompt_path}")
            return ""

        return prompt_path.read_text(encoding="utf-8")

    except Exception as e:
        logger.error(f"Error reading prompt {filename}: {e}")
        return ""