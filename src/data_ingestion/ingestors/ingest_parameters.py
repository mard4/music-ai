import logging
import sys
from pathlib import Path
from typing import List, Any
from data_ingestion.ingestors.base import BaseIngestor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from qdrant_client.models import VectorParams, Distance, PointStruct
from config.settings import settings
from core.infrastructure.database.dependecies import get_audio_repository, get_mongo_client, get_socialfx_repository
from core.domain.audio import AudioFile, SocialFXEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SocialFXIngestor(BaseIngestor):
    """
    Ingestor per i parametri DSP (SocialFXEntry -> Qdrant).
    """

    def __init__(self):
        super().__init__(collection_name=settings.QDRANT_PARAMETERS_COLLECTION_NAME)

    async def run(self):
        logger.info("--- Avvio Ingestion SocialFX Parameters ---")

        repo = get_socialfx_repository()

        docs: List[SocialFXEntry] = await repo.find_all()

        logger.info(f"Trovati {len(docs)} descrittori validi.")
        if not docs:
            return

        self._prepare_collection()

        batch_size = 50

        for batch_start in range(0, len(docs), batch_size):
            batch_end = min(batch_start + batch_size, len(docs))
            batch_docs = docs[batch_start:batch_end]

            descriptors = [doc.descriptor for doc in batch_docs]
            embeddings = self._generate_embeddings(descriptors)

            points = []
            for idx_in_batch, (doc, vector) in enumerate(zip(batch_docs, embeddings)):
                global_idx = batch_start + idx_in_batch + 1

                payload = {
                    "descriptor": doc.descriptor,
                    "effect_type": doc.effect_type,
                    "param_values": doc.parameters.param_values,
                    "param_keys": doc.parameters.param_keys,
                    "source": doc.source
                }

                points.append(PointStruct(
                    id=global_idx,
                    vector=vector,
                    payload=payload
                ))

            await self._upsert_batch(points)

        logger.info("Ingestion Parameters completata.")


