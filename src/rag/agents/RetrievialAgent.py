from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseRetriever:
    """
    Classe base per la gestione delle connessioni Qdrant e OpenAI.
    Gestisce l'infrastruttura comune di embedding e ricerca vettoriale.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        # Connessione condivisa Qdrant
        self.client_qdrant = QdrantClient(
            host=settings.QDRANT_CONNECTION_HOST,
            port=settings.QDRANT_PORT
        )

        # Connessione condivisa OpenAI
        self.client_openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"

    def _get_embedding(self, text: str) -> List[float]:
        """Genera l'embedding per il testo fornito."""
        response = self.client_openai.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _search(self, vector: List[float], limit: int = 5):
        """Esegue la ricerca raw su Qdrant."""
        search_response = self.client_qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True
        )
        # Compatibilità con diverse versioni del client Qdrant
        return getattr(search_response, 'points', search_response)


class AudioRetriever(BaseRetriever):
    """
    Agente di Recupero: Catalogatore Audio.
    Specializzato nella ricerca di file audio (ex AudioRAG).
    """

    def __init__(self):
        super().__init__(settings.QDRANT_AUDIO_COLLECTION_NAME)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_vector = self._get_embedding(query)
            points = self._search(query_vector, limit=k)

            results = []
            for point in points:
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
            logger.error(f"Errore Audio Retrieval: {e}", exc_info=True)
            return []


class ParameterRetriever(BaseRetriever):
    """
    Layer 2: Semantic Parameter Retrieval.
    Specializzato nella ricerca di parametri DSP (ex VectorParameterRetriever).
    """

    def __init__(self):
        super().__init__(settings.QDRANT_PARAMETERS_COLLECTION_NAME)

    def retrieve(self, query: str) -> Optional[Dict]:
        try:
            query_vector = self._get_embedding(query)
            points = self._search(query_vector, limit=1)

            if points:
                best_match = points[0]
                payload = best_match.payload or {}

                logger.info(f"Parametri trovati per '{query}' (Score: {best_match.score:.4f})")

                return {
                    "descriptor": payload.get("descriptor"),
                    "score": best_match.score,
                    "effect_type": payload.get("effect_type"),
                    "params": payload.get("param_values"),
                    "keys": payload.get("param_keys")
                }

            logger.warning(f"Nessun parametro trovato per: '{query}'")
            return None

        except Exception as e:
            logger.error(f"Errore Parameter Retrieval: {e}", exc_info=True)
            return None