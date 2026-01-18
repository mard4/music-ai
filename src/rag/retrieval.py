from typing import List, Dict, Any
import logging
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class AudioRAG:
    """
    Agente di Recupero: Catalogatore Audio
    Cerca solo nella collezione dei file audio (audio_vectors).
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_AUDIO_COLLECTION_NAME

        # Connessione diretta a Qdrant
        self.client_qdrant = QdrantClient(
            host=settings.QDRANT_CONNECTION_HOST,
            port=settings.QDRANT_PORT
        )

        # Connessione OpenAI per embedding
        self.client_openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recupera i file audio più simili alla query."""
        try:
            # 1. Generazione Embedding
            response = self.client_openai.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_vector = response.data[0].embedding

            # 2. Ricerca in Qdrant
            search_result = self.client_qdrant.retrieve(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k
            )

            results = []
            for point in search_result:
                # Payload è un dizionario (grazie a QdrantClient)
                payload = point.payload or {}

                results.append({
                    "type": "audio_file",
                    "score": point.score,
                    "filename": payload.get("filename"),
                    "label": payload.get("label"),
                    "bpm": payload.get("bpm"),
                    "key": payload.get("key"),
                    "tags": payload.get("categories"),
                    "mongo_id": payload.get("mongo_id")
                })

            return results

        except Exception as e:
            logger.error(f"Errore Audio RAG: {e}")
            return []