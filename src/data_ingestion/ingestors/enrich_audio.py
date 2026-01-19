import logging
import os
import tempfile
import asyncio
from typing import List

from data_ingestion.ingestors.base import BaseIngestor
from qdrant_client.models import PointStruct
from config.settings import settings
from core.infrastructure.database.dependecies import (
    get_audio_repository,
    get_gridfs_handler
)
from rag.tools.audio_analysis import LabelEnricherTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnrichedCollectionIngestor(BaseIngestor):
    """
    Crea una NUOVA collection 'audio_enriched' contenente:
    - Vector: Embedding della NUOVA descrizione generata dall'AI.
    - Payload: filename, new_label, clap_score, tags, ai_tags.
    """

    def __init__(self):
        collection_name =settings.QDRANT_ENRICHED_COLLECTION_NAME
        super().__init__(collection_name=collection_name)

        # Tool AI
        self.enricher = LabelEnricherTool()

    async def run(self):
        logger.info(f"--- Creazione Collection Arricchita: {self.collection_name} ---")

        repo = get_audio_repository()
        gridfs = get_gridfs_handler()

        self._prepare_collection()

        docs = await repo.find_all()
        logger.info(f"Processando {len(docs)} file da MongoDB...")

        points_batch = []
        batch_size = 20

        for i, doc in enumerate(docs):
            tmp_path = None
            try:
                gridfs_id = doc.gridfs_file_id
                original_filename = doc.sample.file_name
                original_tags = doc.metadata.categories


                if not gridfs_id: continue

                file_bytes = await gridfs.download_file(str(gridfs_id))
                if not file_bytes: continue

                ext = os.path.splitext(original_filename)[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # Analisi AI (LabelEnricher)
                logger.info(f"[{i + 1}/{len(docs)}] Analisi: {original_filename}")
                result = self.enricher.enrich_and_verify(
                    filename=original_filename,
                    audio_path=tmp_path,
                    original_tags= original_tags
                )

                new_label = result["caption"]
                smart_tags = result["smart_tags"]
                clap_score = result["clap_score"]

                # Generazione Embedding (sulla Label generata dall'AI)
                vectors = self._generate_embeddings([new_label])
                if not vectors:
                    logger.warning(f"Embedding fallito per {original_filename}")
                    continue
                vector = vectors[0]

                point = PointStruct(
                    id=i + 1,  # In produzione meglio usare UUID deterministico
                    vector=vector,
                    payload={
                        "original_filename": original_filename,
                        "label": new_label,
                        "clap_score": clap_score,
                        "mongo_id": str(gridfs_id),
                        "original_tags": original_tags,
                        "ai_tags": smart_tags
                    }
                )
                points_batch.append(point)

                if len(points_batch) >= batch_size:
                    await self._upsert_batch(points_batch)
                    points_batch = []

            except Exception as e:
                logger.error(f"Errore {doc.sample.file_name}: {e}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

        if points_batch:
            await self._upsert_batch(points_batch)

        logger.info("Ingestion completata.")


if __name__ == "__main__":
    asyncio.run(EnrichedCollectionIngestor().run())