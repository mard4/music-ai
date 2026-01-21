import logging
import shutil
from pathlib import Path
import os
from bson import ObjectId
from core.infrastructure.database.dependecies import get_gridfs_handler

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

class FoundSimilarAudios():
    def __init__(self):
        logger.info("Inizializzazione Workflow AI...")

        self.storage_handler = get_gridfs_handler()

        self.memory = {"last_analysis": None}

    async def prepare_audio_for_web(self, hit: dict) -> str:
        """
        Recupera il file audio (da disco o GridFS tramite Core) e ritorna l'URL.
        """
        try:
            real_name = hit.get('original_filename') or hit.get('filename') or 'audio.wav'
            safe_filename = real_name.replace(" ", "_").replace("/", "_")

            public_dir = os.path.join(os.getcwd(), "public_audio")
            os.makedirs(public_dir, exist_ok=True)
            dest_path = os.path.join(public_dir, safe_filename)
            web_url = f"http://localhost:8000/public/{safe_filename}"

            if os.path.exists(dest_path):
                return web_url

            mongo_id = hit.get('mongo_id')
            if mongo_id:
                try:
                    file_data = await self.storage_handler.download_file(mongo_id)

                    if file_data:
                        with open(dest_path, "wb") as f:
                            f.write(file_data)
                        return web_url
                    else:
                        logger.warning(f"File binario non trovato in GridFS per ID: {mongo_id}")

                except Exception as e:
                    logger.warning(f"Errore download da Storage Core per {mongo_id}: {e}")

            local_source = hit.get('file_path')
            if local_source and os.path.exists(local_source):
                shutil.copy(local_source, dest_path)
                return web_url

            return ""

        except Exception as e:
            logger.error(f"Errore preparazione audio web: {e}")
            return ""