import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).parent.parent))
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI

from config.settings import settings
from core.infrastructure.database.dependecies import get_mongo_client
from core.domain.audio import SocialFXEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_socialfx_vectors():
    client_mongo = get_mongo_client()
    db_name = settings.database.mongodb_database_name
    db = client_mongo[db_name]
    collection = db[settings.database.mongodb_socialfx_collection]

    docs: List[SocialFXEntry] = []

    async for doc in collection.find({}):
        try:
            entry = SocialFXEntry(**doc)
            docs.append(entry)
        except Exception as e:
            logger.warning(f"Documento non valido ignorato: {e}")

    logger.info(f"Trovati {len(docs)} descrittori validi in MongoDB")

    if not docs:
        logger.warning("Nessun documento valido trovato. Uscita.")
        return

    # 2. Configurazione Qdrant
    qdrant = QdrantClient(host=settings.QDRANT_CONNECTION_HOST, port=settings.QDRANT_PORT)
    collection_name = settings.QDRANT_PARAMETERS_COLLECTION_NAME

    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
        logger.info(f"Collezione '{collection_name}' eliminata")

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    logger.info(f"Collezione '{collection_name}' ricreata.")

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # 4. Creazione Punti
    points = []
    batch_size = 50

    logger.info(f"Creazione {len(docs)} embedding in batch di {batch_size}...")

    for batch_start in range(0, len(docs), batch_size):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]

        # Accesso ai dati tramite attributi del modello Pydantic
        descriptors = [doc.descriptor for doc in batch_docs]

        try:
            response = openai_client.embeddings.create(
                input=descriptors,
                model="text-embedding-3-small"
            )

            for idx_in_batch, (doc, embedding_data) in enumerate(zip(batch_docs, response.data)):
                idx = batch_start + idx_in_batch

                # Payload strutturato garantito dal modello
                payload = {
                    "descriptor": doc.descriptor,
                    "effect_type": doc.effect_type,
                    "param_values": doc.parameters.param_values,
                    "param_keys": doc.parameters.param_keys,
                    "source": doc.source
                }

                # Uso PointStruct invece di dizionari raw
                points.append(PointStruct(
                    id=idx + 1,
                    vector=embedding_data.embedding,
                    payload=payload
                ))

            logger.info(f"[{batch_end}/{len(docs)}] Batch elaborato")

        except Exception as e:
            logger.error(f"Errore batch embedding: {e}")

    # 5. Caricamento su Qdrant
    if points:
        try:
            qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"✓ Indicizzati {len(points)} parametri su Qdrant.")
        except Exception as e:
            logger.error(f"Errore upsert su Qdrant: {e}")

if __name__ == "__main__":
    asyncio.run(ingest_socialfx_vectors())