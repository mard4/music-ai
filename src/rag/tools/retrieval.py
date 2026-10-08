import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from openai import OpenAI
from config.settings import settings
from core.infrastructure.database.dependecies import get_vector_repository

logger = logging.getLogger(__name__)

class RetrievalTool:
    """
    TOOL: Audio Database Retrieval.
    Responsabilità: Interfaccia con Qdrant per trovare sample simili.
    """

    def __init__(self):
        # Inietta il repository astratto invece del client concreto
        self.vector_repo = get_vector_repository()
        self.client_openai = OpenAI(api_key=settings.MODEL_API_KEY)


    async def search_similar_audio_text_vector(self, query_text: str) -> List[Dict[str, Any]]:
        """"
        Cerca audio basandosi su una descrizione testuale (es. "Trovami un basso distorto").
        Usa lo spazio 'text_vector'.
        """

        try:
            emb_res = self.client_openai.embeddings.create(
                input=query_text,
                model=settings.MODEL_EMBEDDING_MODEL
            )
            vector = emb_res.data[0].embedding

            results = await self.vector_repo.search_audio(vector)

            return [self._format_result(res) for res in results]

        except Exception as e:
            logger.error(f"Retrieval Text Error: {e}")
            return []

    async def search_similar_audio_audio_vector(self, vector: List[float]) -> List[Dict[str, Any]]:
        """
        Cerca audio basandosi su un vettore audio raw (CLAP).
        Usa lo spazio 'audio_vector'.
        """
        try:

            # Nota: 'vector' qui arriva già calcolato da CLAP (dall'AudioAnalystAgent)
            results = await self.vector_repo.search_audio(
                vector=vector,
                vector_name="audio_vector"
            )

            return [self._format_result(res) for res in results]

        except Exception as e:
            logger.error(f"Retrieval Vector Error: {e}")
            return []

    def _format_result(self, res) -> Dict[str, Any]:
        """Helper per formattare l'output standard verso gli agenti."""
        return {
            "filename": res.filename,
            "original_filename": res.metadata.get("real_filename"),
            "score": res.score,
            "description": res.label,
            "tags": res.categories,
            "quality": res.metadata.get("clap_quality"),
            "mongo_id": res.payload.get("mongo_id"),
            "file_path": res.payload.get("source")
        }
