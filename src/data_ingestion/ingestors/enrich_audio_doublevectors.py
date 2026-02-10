import logging
import os
import tempfile
import asyncio
import time
from typing import List

from data_ingestion.ingestors.base import BaseIngestor
from qdrant_client.models import PointStruct, VectorParams, Distance
from config.settings import settings
from core.infrastructure.database.dependecies import (
    get_audio_repository,
    get_gridfs_handler
)
from rag.tools.audio_analysis import LabelEnricherTool
from rag.clap.model_handler import create_clap_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnrichedCollectionIngestor(BaseIngestor):
    """
    Crea una NUOVA collection 'audio_enriched' contenente:
    - Vettori Doppi:
        1. 'text_vector': Embedding della NUOVA descrizione generata dall'AI.
        2. 'audio_vector': Embedding audio generato da CLAP.
    - Payload: filename, new_label, clap_score, tags, ai_tags.
    """

    def __init__(self):
        collection_name = settings.QDRANT_ENRICHED_COLLECTION_NAME
        super().__init__(collection_name=collection_name)

        self.enricher = LabelEnricherTool()
        self.clap = create_clap_model()

    def _prepare_collection(self):
        """
        Override per configurare la collezione con Vettori Nominati (Dual Vector).
        """
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
            logger.info(f"Collezione '{self.collection_name}' esistente eliminata.")
            time.sleep(2)
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_vector": VectorParams(size=1536, distance=Distance.COSINE),  # OpenAI
                "audio_vector": VectorParams(size=512, distance=Distance.COSINE),  # CLAP (HTSAT-base)
            },
        )
        logger.info(f"Collezione '{self.collection_name}' creata con Dual Vector Support.")

    async def run(self):
        logger.info(f"--- Creazione Collection Arricchita (Dual Vector): {self.collection_name} ---")

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
                original_label = doc.sample.label
                original_tags = doc.metadata.categories
                main_category = doc.metadata.main_category
                main_tag = doc.metadata.main_tag

                if not gridfs_id: continue

                # Scarichiamo il file
                file_bytes = await gridfs.download_file(str(gridfs_id))
                if not file_bytes: continue

                ext = os.path.splitext(original_filename)[1] or ".wav"

                # Creiamo file temporaneo (serve sia per l'Enricher che per il vettore CLAP)
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # 1. Analisi AI (LabelEnricher - Generazione Testo)
                logger.info(f"[{i + 1}/{len(docs)}] Analisi: {original_filename}")
                result = self.enricher.enrich_and_verify(
                    filename=original_filename,
                    label = original_label,
                    main_category=main_category,
                    audio_path=tmp_path,
                    original_tags=original_tags
                )

                new_label = result["caption"]
                ai_tags = result["ai_tags"]
                clap_score = result["clap_score"]

                # 2. Generazione Embedding Testuale (OpenAI) sulla Label generata
                text_vectors = self._generate_embeddings([new_label])
                if not text_vectors:
                    logger.warning(f"Embedding testo fallito per {original_filename}")
                    continue
                text_vec = text_vectors[0]

                # 3. Generazione Embedding Audio (CLAP) sul file temporaneo
                # Sfruttiamo il file tmp_path che esiste già
                audio_vec = None
                try:
                    audio_vec_np = self.clap.get_audio_embedding([tmp_path])
                    if len(audio_vec_np) > 0:
                        audio_vec = audio_vec_np[0].tolist()
                except Exception as e:
                    logger.error(f"Embedding audio CLAP fallito per {original_filename}: {e}")

                if audio_vec is None:
                    logger.warning(f"Salta documento {original_filename}: vettore audio mancante.")
                    continue

                # 4. Creazione Punto con Doppio Vettore
                point = PointStruct(
                    id=i + 1,
                    vector={
                        "text_vector": text_vec,
                        "audio_vector": audio_vec
                    },
                    payload={
                        "original_filename": original_filename,
                        "mongo_id": str(gridfs_id),
                        "original_label": original_label,
                        "original_tags": original_tags,
                        "main_category": main_category,
                        "main_tag": main_tag,
                        "ai_label": new_label,  # Label Migliorata dall'AI
                        "ai_tags": ai_tags,
                        "clap_score": round(clap_score, 5)
                    }
                )
                points_batch.append(point)

                if len(points_batch) >= batch_size:
                    await self._upsert_batch(points_batch)
                    points_batch = []

            except Exception as e:
                logger.error(f"Errore {doc.sample.file_name}: {e}")

            finally:
                # Pulizia file temporaneo
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

        if points_batch:
            await self._upsert_batch(points_batch)

        logger.info("Ingestion Arricchita completata.")


if __name__ == "__main__":
    asyncio.run(EnrichedCollectionIngestor().run())