import os
from pathlib import Path
import logging 
import zipfile

from motor.motor_asyncio import AsyncIOMotorGridFSBucket

from commons.data_models.models import MongoDBConfig
from commons.mongodb.mongo_dependecies import get_mongo_client, get_audiofiles_collection, get_mongo_database
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

def _checkOutputDir(output_dir: Path):
    """Controlla se la directory di output esiste e non è vuota"""
    if output_dir.exists():
        if any(output_dir.iterdir()):
            logging.info(f"Directory già esistente e non vuota")
            return True
        else:
            logging.info("Directory vuota, procedo con la creazione.")
            os.makedirs(output_dir, exist_ok=True)
    return False

def _check_and_create_output_dir(output_dir: Path, force_extract: bool = False) -> bool:
    """Helper function to check output directory and create if needed."""
    if output_dir.exists() and any(output_dir.iterdir()) and not force_extract:
        logging.info(f"Directory già esistente e non vuota, salto l'estrazione: {output_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return True

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

async def getAudioFiles(mongo_config: MongoDBConfig):
    """Factory per creare l'extractor con le dipendenze già iniettate"""
    client = get_mongo_client()
    db = get_mongo_database(client=client)
    fs = AsyncIOMotorGridFSBucket(db, bucket_name=mongo_config.fs_collection)

    collection = get_audiofiles_collection(db)
    repository = MongoAudioFilesRepository(collection)

    audio_files = await repository.find_audio_by_filter()

    return audio_files