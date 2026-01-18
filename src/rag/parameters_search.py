from typing import Dict, Optional
from datapizza.pipeline import DagPipeline
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.vectorstores.qdrant import QdrantVectorstore
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorParameterRetriever:
    """
    Layer 2: Semantic Parameter Retrieval (Vector Search)
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_PARAMETERS_COLLECTION_NAME
        self.openai_client = OpenAIEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )

        self.qdrant = QdrantVectorstore(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_CONNECTION_PORT)
        self.pipeline = DagPipeline()
        self.pipeline.add_module("embedder", self.openai_client)
        self.pipeline.add_module("retriever", self.qdrant.as_retriever(
            collection_name=self.collection_name,
            k=1,
            vector_name=""
        ))

        self.pipeline.connect("embedder", "retriever", target_key="query_vector")

    def get_parameters(self, query: str) -> Optional[Dict]:
        """
        Esegue la vector search per trovare i parametri.
        """
        try:
            result = self.pipeline.run({
                "embedder": {"text": query},
                "retriever": {
                    "collection_name": self.collection_name,
                    "k": 1,
                    "vector_name": ""
                }
            })

            chunks = result["retriever"]
            logger.info(f"Query: '{query}' | Risultati trovati: {len(chunks) if chunks else 0}")
            logger.debug(f"Raw response: {chunks}")

            if chunks and len(chunks) > 0:
                best_match = chunks[0]
                meta = getattr(best_match, "metadata", {})

                return {
                    "descriptor": meta.get("descriptor"),
                    "score": getattr(best_match, "score", None),
                    "effect_type": meta.get("effect_type"),
                    "params": meta.get("param_values"),
                    "keys": meta.get("param_keys")
                }

            logger.warning(f"Nessun risultato trovato per: '{query}'")
            return None

        except Exception as e:
            logger.error(f"Errore Vector Search: {e}", exc_info=True)
            return None