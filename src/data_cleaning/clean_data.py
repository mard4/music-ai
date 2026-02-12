import asyncio
import logging
import sys
import os
from pathlib import Path

# Setup path per importare i moduli del progetto
sys.path.append(os.path.abspath("../data_exploration"))

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from core.infrastructure.database.dependecies import get_mongo_client
from config.settings import settings, DatabaseSettings

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
DB_NAME = "audio_db_test"
COLLECTION_NAME = "audio_samples"


async def clean_duplicates():
    logger.info("--- AVVIO PULIZIA DUPLICATI MONGODB ---")

    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    fs = AsyncIOMotorGridFSBucket(db)

    # --- DEBUG: CONTEGGIO INIZIALE ---
    # Conta tutti i documenti presenti nella collezione (query vuota {})
    total_docs_start = await collection.count_documents({})
    logger.info(f"📌 TOTALE DOCUMENTI RILEVATI (START): {total_docs_start}")
    # ---------------------------------

    pipeline = [
        {
            "$group": {
                "_id": {
                    "filename": "$sample.file_name",
                    "duration": "$metadata.duration" # Decommenta se vuoi distinguere file con stesso nome ma durata diversa
                },
                "ids": {"$push": "$_id"},  # Lista degli ID metadata duplicati
                "gridfs_ids": {"$push": "$gridfs_file_id"},  # Lista degli ID binari (GridFS)
                "count": {"$sum": 1}
            }
        },
        {
            "$match": {
                "count": {"$gt": 1}  # Filtra solo i gruppi con duplicati
            }
        }
    ]

    logger.info("Ricerca duplicati in corso...")
    cursor = collection.aggregate(pipeline)

    deleted_docs = 0
    deleted_chunks = 0

    async for doc in cursor:
        filename = doc["_id"]["filename"]
        ids = doc["ids"]
        gridfs_ids = doc["gridfs_ids"]
        count = doc["count"]

        logger.info(f"Trovati {count} duplicati per: {filename}")

        # LOGICA DI ELIMINAZIONE
        # ID da mantenere (il "vincitore" - il primo della lista)
        id_to_keep = ids[0]

        # IDs da eliminare (tutti gli altri)
        ids_to_remove = ids[1:]
        gridfs_to_remove = gridfs_ids[1:]

        # A. Eliminazione Metadata (Documento JSON)
        if ids_to_remove:
            result = await collection.delete_many({"_id": {"$in": ids_to_remove}})
            deleted_docs += result.deleted_count
            logger.info(f" -> Eliminati {result.deleted_count} documenti metadata.")

        # B. Eliminazione Binary (GridFS)
        for g_id in gridfs_to_remove:
            if g_id:
                try:
                    await fs.delete(g_id)
                    deleted_chunks += 1
                except Exception as e:
                    logger.warning(f" -> Impossibile eliminare GridFS ID {g_id}: {e}")

    logger.info("--- PULIZIA COMPLETATA ---")
    logger.info(f"Totale documenti rimossi: {deleted_docs}")
    logger.info(f"Totale file audio (GridFS) rimossi: {deleted_chunks}")

    # --- DEBUG: CONTEGGIO FINALE ---
    total_docs_end = await collection.count_documents({})
    logger.info(f"📌 TOTALE DOCUMENTI RIMASTI (END): {total_docs_end}")
    # -------------------------------


if __name__ == "__main__":
    try:
        asyncio.run(clean_duplicates())
    except KeyboardInterrupt:
        pass