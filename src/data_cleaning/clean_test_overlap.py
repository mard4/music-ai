import asyncio
import logging
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from bson import ObjectId
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRY_RUN = False  # Metti False per cancellare davvero



async def clean_overlaps():
    logger.info(f"Avvio pulizia overlap. DRY_RUN: {DRY_RUN}")

    # 1. Connessione a MongoDB
    client = AsyncIOMotorClient(settings.database.mongodb_connection_string)

    # DB Train (Sorgente di verità - NON TOCCARE)
    db_train_name = settings.database.mongodb_database_name
    db_train = client[db_train_name]
    col_train = db_train[settings.database.mongodb_audio_collection]

    # DB Test (Target da pulire)
    db_test_name = settings.database.mongodb_database_name_test
    db_test = client[db_test_name]
    col_test = db_test[settings.database.mongodb_audio_collection]
    fs_bucket_test = AsyncIOMotorGridFSBucket(db_test, bucket_name=settings.database.mongodb_gridfs_bucket)

    logger.info(f"Connesso. Train DB: '{db_train_name}' | Test DB: '{db_test_name}'")

    # 2. Iterazione su tutti i file del Test DB
    cursor = col_test.find({})
    total_checked = 0
    duplicates_found = 0

    logger.info("Inizio scansione Test DB...")

    async for doc in cursor:
        total_checked += 1

        # Recupera il nome file (identificativo univoco per SampleFocus)
        # Struttura: doc['sample']['file_name']
        try:
            sample_data = doc.get('sample', {})
            filename = sample_data.get('file_name')

            if not filename:
                logger.warning(f"Documento {doc.get('_id')} senza filename. Skipping.")
                continue

            # 3. Controllo esistenza nel Train DB
            # Usiamo find_one diretto per velocità
            exists_in_train = await col_train.find_one({"sample.file_name": filename})

            if exists_in_train:
                duplicates_found += 1
                audio_id = doc.get('_id')
                gridfs_id = doc.get('gridfs_file_id')

                msg = f"[DUPLICATO] '{filename}' trovato in Train."

                if DRY_RUN:
                    logger.info(f"{msg} (DRY RUN: verrebbe eliminato ID {audio_id})")
                else:
                    # 4. Eliminazione (Solo se DRY_RUN = False)
                    logger.info(f"{msg} ELIMINAZIONE IN CORSO...")

                    # A. Elimina file audio fisico da GridFS
                    if gridfs_id:
                        try:
                            # GridFS richiede ObjectId
                            g_id = ObjectId(gridfs_id) if isinstance(gridfs_id, str) else gridfs_id
                            await fs_bucket_test.delete(g_id)
                            logger.info(f" -> GridFS {gridfs_id} eliminato.")
                        except Exception as e:
                            logger.error(f" -> Errore eliminazione GridFS {gridfs_id}: {e}")

                    # B. Elimina metadati dalla collection
                    await col_test.delete_one({"_id": audio_id})
                    logger.info(f" -> Documento {audio_id} eliminato.")

        except Exception as e:
            logger.error(f"Errore processando documento {doc.get('_id', 'unknown')}: {e}")

    logger.info("------------------------------------------------")
    logger.info(f"Scansione completata.")
    logger.info(f"Totale controllati: {total_checked}")
    logger.info(f"Duplicati trovati (Train->Test leak): {duplicates_found}")
    if DRY_RUN:
        logger.info("Nessun file è stato eliminato. Imposta DRY_RUN = False nello script per procedere.")
    else:
        logger.info("Pulizia completata.")


if __name__ == "__main__":
    try:
        asyncio.run(clean_overlaps())
    except KeyboardInterrupt:
        logger.info("Script interrotto manualmente.")