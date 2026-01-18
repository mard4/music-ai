from typing import Dict, Optional
import logging
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorParameterRetriever:
    """
    Layer 2: Semantic Parameter Retrieval (Vector Search)
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_PARAMETERS_COLLECTION_NAME

        self.client_qdrant = QdrantClient(
            host=settings.QDRANT_CONNECTION_HOST,
            port=settings.QDRANT_PORT
        )
        self.client_openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"

    def get_parameters(self, query: str) -> Optional[Dict]:
        """
        Esegue la vector search per trovare i parametri.
        """
        try:
            # 1. Embedding
            response = self.client_openai.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_vector = response.data[0].embedding

            # 2. Ricerca Qdrant (limit=1 per trovare il miglior match)
            search_result = self.client_qdrant.retrieve(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=1
            )

            logger.info(f"Query: '{query}' | Risultati trovati: {len(search_result)}")

            if search_result:
                best_match = search_result[0]
                payload = best_match.payload or {}

                return {
                    "descriptor": payload.get("descriptor"),
                    "score": best_match.score,
                    "effect_type": payload.get("effect_type"),
                    "params": payload.get("param_values"),
                    "keys": payload.get("param_keys")
                }

            logger.warning(f"Nessun risultato trovato per: '{query}'")
            return None

        except Exception as e:
            logger.error(f"Errore Vector Search: {e}", exc_info=True)
            return None