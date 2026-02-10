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
        #self.audio_collection = settings.QDRANT_AUDIO_COLLECTION_NAME
        self.audio_collection = settings.QDRANT_ENRICHED_COLLECTION_NAME
        self.fx_collection = settings.QDRANT_PARAMETERS_COLLECTION_NAME

    async def search_audio(self, vector: List[float],vector_name: str = "text_vector") -> List[AudioSearchResult]:
        try:
            response = await self.client.query_points(
                collection_name=self.audio_collection,
                query=vector,
                using=vector_name,
                with_payload=True
            )

            results = response.points

            # Mappatura da Qdrant Point a Domain Object
            mapped_results = []
            for point in results:
                payload = point.payload or {}

                display_name = payload.get("label", payload.get("original_filename", "Unknown Sample"))
                ai_tags = payload.get("ai_tags", payload.get("original_tags", []))
                quality_score = payload.get("clap_score", 0.0)

                mapped_results.append(AudioSearchResult(
                    id=point.id,
                    score=point.score,
                    payload=payload,
                    filename=display_name,
                    label=payload.get("label", ""),
                    categories=ai_tags,
                    metadata={"clap_quality": quality_score, "real_filename": payload.get("original_filename"), "original_tags": payload.get("original_tags")}
                ))
            return mapped_results
        except Exception as e:
            print(f"ERRORE QDRANT search_audio: {e}")
            return []

    async def search_social_fx(self, vector: List[float], limit: int = 1) -> List[SocialFXSearchResult]:
        try:
            response = await self.client.query_points(
                collection_name=self.fx_collection,
                query=vector,
                limit=limit,
                with_payload=True
            )
            results = response.points

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
            print(f"ERRORE QDRANT search_social_fx: {e}")
            return []