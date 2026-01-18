import logging
import sys
from pathlib import Path
from typing import List
from data_ingestion.ingestors.base import BaseIngestor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from qdrant_client.models import VectorParams, Distance, PointStruct
from config.settings import settings
from core.infrastructure.database.dependecies import get_audio_repository, get_mongo_client
from core.domain.audio import AudioFile, SocialFXEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioCatalogIngestor(BaseIngestor):
    """
    Ingestor per i file audio (AudioFile -> Qdrant).
    """

    def __init__(self):
        super().__init__(collection_name=settings.QDRANT_AUDIO_COLLECTION_NAME)

    async def run(self):
        logger.info("--- Avvio Ingestion Audio Catalog ---")

        # 1. Recupero dati
        repo = get_audio_repository()
        docs: List[AudioFile] = await repo.find_all()
        logger.info(f"Trovati {len(docs)} sample audio in MongoDB.")

        if not docs:
            logger.warning("Nessun documento trovato. Skip.")
            return

        # 2. Preparazione Qdrant
        self._prepare_collection()

        # 3. Processing
        batch_size = 50
        points_buffer = []

        for i, doc in enumerate(docs):
            try:
                # Costruzione testo semantico
                sample = doc.sample
                meta = doc.metadata

                label = sample.label
                main_cat = meta.main_category or ""
                # Pulizia eventuale '?' residuo
                if main_cat and "?" in main_cat:
                    main_cat = main_cat.split("?")[0]

                categories = meta.categories or []
                tags_str = ", ".join(categories) if isinstance(categories, list) else str(categories)

                full_text = f"{label}. Category: {main_cat}. Tags: {tags_str}"

                # Aggiungiamo metadati temporanei per il batch processing
                # (Nota: in un sistema prod-ready si farebbe batching degli embedding separato)
                points_buffer.append({
                    "text": full_text,
                    "doc": doc,
                    "idx": i + 1
                })

                # Processa il buffer quando è pieno
                if len(points_buffer) >= batch_size:
                    await self._process_buffer(points_buffer)
                    points_buffer = []

            except Exception as e:
                logger.warning(f"Skipping doc {doc.gridfs_file_id}: {e}")

        # Processa residui
        if points_buffer:
            await self._process_buffer(points_buffer)

        logger.info("✓ Ingestion Audio completata.")

    async def _process_buffer(self, buffer: List[dict]):
        texts = [item["text"] for item in buffer]
        embeddings = self._generate_embeddings(texts)

        points = []
        for item, vector in zip(buffer, embeddings):
            doc = item["doc"]
            sample = doc.sample
            meta = doc.metadata

            payload = {
                "filename": sample.file_name,
                "label": sample.label,
                "source": sample.source,
                "bpm": meta.bpm,
                "key": meta.key,
                "duration": meta.duration,
                "main_category": meta.main_category,
                "categories": meta.categories,
                "mongo_id": doc.gridfs_file_id
            }

            points.append(PointStruct(
                id=item["idx"],
                vector=vector,
                payload=payload
            ))

        await self._upsert_batch(points)
