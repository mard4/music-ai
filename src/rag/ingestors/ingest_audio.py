import asyncio
import os
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI

from config.settings import settings
from core.infrastructure.database.dependecies import get_audio_repository
from core.domain.audio import AudioFile
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_audio_catalog():

    repo = get_audio_repository()
    docs: list[AudioFile] = await repo.find_all()
    logger.info(f"Trovati {len(docs)} sample audio in MongoDB (tramite Repository)")

    if not docs:
        logger.warning("Nessun documento trovato. Uscita.")
        return

    qdrant = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    collection_name = settings.QDRANT_AUDIO_COLLECTION_NAME

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
        logger.info(f"Collezione '{collection_name}' esistente eliminata.")

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    logger.info(f"Collezione '{collection_name}' creata.")

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # 3. Batch Embedding e Caricamento
    points = []
    batch_size = 50

    logger.info(f"Inizio embedding di {len(docs)} file...")

    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]

        texts_to_embed = []
        valid_indices = []

        # Preparazione testi
        for i, doc in enumerate(batch_docs):
            try:
                sample = doc.sample
                meta = doc.metadata

                label = sample.label
                main_cat = meta.main_category or ""

                if main_cat and "?" in main_cat:
                    main_cat = main_cat.split("?")[0]

                categories = meta.categories or []
                tags_str = ", ".join(categories) if isinstance(categories, list) else str(categories)

                # Creazione stringa semantica ricca per la ricerca
                full_text = f"{label}. Category: {main_cat}. Tags: {tags_str}"

                texts_to_embed.append(full_text)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Skipping doc {doc.gridfs_file_id}: {e}")

        if not texts_to_embed:
            continue

        try:
            # Generazione Embedding
            response = openai_client.embeddings.create(
                input=texts_to_embed,
                model="text-embedding-3-small"
            )

            # Creazione Punti Qdrant
            for i, embedding_data in enumerate(response.data):
                original_idx = valid_indices[i]
                doc = batch_docs[original_idx]
                sample = doc.sample
                meta = doc.metadata

                # Payload per il frontend/agente
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

                # ID univoco (int) per Qdrant
                point_id = batch_start + original_idx + 1

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding_data.embedding,
                    payload=payload
                ))

            logger.info(f"Batch {batch_start}-{batch_end} processato.")

        except Exception as e:
            logger.error(f"Errore batch: {e}")

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        logger.info(f"✓ Indicizzazione completata: {len(points)} audio file in '{collection_name}'.")


if __name__ == "__main__":
    asyncio.run(ingest_audio_catalog())