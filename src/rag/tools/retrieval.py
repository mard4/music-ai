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
        self.client_openai = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def search_similar_audio(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            emb_res = self.client_openai.embeddings.create(
                input=query_text,
                model=settings.OPENAI_EMBEDDING_MODEL
            )
            vector = emb_res.data[0].embedding

            results = await self.vector_repo.search_audio(vector, limit=limit)

            # 3. Format per l'Agente
            return [{
                "filename": res.filename,
                "score": res.score,
                "description": res.label,
                "tags": res.categories
            } for res in results]

        except Exception as e:
            logger.error(f"Retrieval Error: {e}")
            return []