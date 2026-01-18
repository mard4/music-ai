from typing import List
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from core.interfaces.vector_store import VectorStoreRepository
from core.domain.search import AudioSearchResult, SocialFXSearchResult
from config.settings import settings


class QdrantVectorRepository(VectorStoreRepository):
    """
    Implementazione concreta di VectorStoreRepository usando Qdrant.
    """

    def __init__(self, client: AsyncQdrantClient):
        self.client = client
        self.audio_collection = settings.QDRANT_AUDIO_COLLECTION_NAME
        self.fx_collection = settings.QDRANT_PARAMETERS_COLLECTION_NAME

    async def search_audio(self, vector: List[float], limit: int = 5) -> List[AudioSearchResult]:
        try:
            results = await self.client.query_points(
                collection_name=self.audio_collection,
                query_vector=vector,
                limit=limit,
                with_payload=True
            )

            # Mappatura da Qdrant Point a Domain Object
            mapped_results = []
            for point in results:
                payload = point.payload or {}
                mapped_results.append(AudioSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=payload,
                    filename=payload.get("filename", "unknown"),
                    label=payload.get("label", ""),
                    categories=payload.get("categories", [])
                ))
            return mapped_results
        except Exception as e:
            return []

    async def search_social_fx(self, vector: List[float], limit: int = 1) -> List[SocialFXSearchResult]:
        try:
            results = await self.client.query_points(
                collection_name=self.fx_collection,
                query_vector=vector,
                limit=limit,
                with_payload=True
            )

            mapped_results = []
            for point in results:
                payload = point.payload or {}
                mapped_results.append(SocialFXSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=payload,
                    descriptor=payload.get("descriptor", ""),
                    effect_type=payload.get("effect_type", ""),
                    param_values=payload.get("param_values", []),
                    param_keys=payload.get("param_keys", [])
                ))
            return mapped_results
        except Exception as e:
            return []