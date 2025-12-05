from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
from datapizza.modules.prompt import ChatPromptTemplate
import pandas as pd
import os
from datapizza.type import Media, MediaBlock, TextBlock
import dotenv
import asyncio

from commons.data_models.models import MongoDBConfig
from commons.utils import getAudioFiles, read_txt_file, get_mongo_client, get_mongo_database, get_audiofiles_collection
from commons.mongodb.mongo_repositories import MongoAudioFilesRepository
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
import logging
from datetime import datetime
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

CLEAN_LABELS_COLLECTION = "clean_audio_labels"


class CleanLabelDocument:
    def __init__(self, gridfs_file_id: str, original_label: str, clean_label: str,
                 categories: List[str], created_at: datetime = None):
        self.gridfs_file_id = gridfs_file_id
        self.original_label = original_label
        self.clean_label = clean_label
        self.categories = categories

    def to_dict(self):
        return {
            'gridfs_file_id': self.gridfs_file_id,
            'original_label': self.original_label,
            'label': self.clean_label,
            'categories': self.categories
        }


async def prepare_data_with_ids(mongo_config: MongoDBConfig):
    audio_files = await getAudioFiles(mongo_config)
    logger.info(f"Dataset ottenuto: {len(audio_files)} elementi")

    prepared_data = []

    for audio_file in audio_files:
        data = {
            'gridfs_file_id': audio_file.gridfs_file_id,
            'label': audio_file.sample.label,
            'categories': audio_file.metadata.categories,
            'key': audio_file.metadata.key,
            'bpm': audio_file.metadata.bpm,
            'duration': audio_file.metadata.duration,
            'file_name': audio_file.sample.file_name,
            'source': audio_file.sample.source
        }
        prepared_data.append(data)

    return prepared_data


def generate_clean_label(data_item: Dict, client: OpenAIClient) -> str:
    prompt_template = read_txt_file("./prompts/clean_label.txt")

    prompt = prompt_template.format(
        label=data_item['label'],
        categories=', '.join(data_item['categories'])
    )

    response = client.invoke(input=[TextBlock(content=prompt)])

    return response.text.strip()


async def create_clean_labels_collection(db):
    if CLEAN_LABELS_COLLECTION not in await db.list_collection_names():
        await db.create_collection(CLEAN_LABELS_COLLECTION)
        collection = db[CLEAN_LABELS_COLLECTION]
        await collection.create_index("gridfs_file_id", unique=True)
    return db[CLEAN_LABELS_COLLECTION]


async def save_clean_label_to_mongo(collection, clean_label_doc: CleanLabelDocument):
    try:
        existing = await collection.find_one({"gridfs_file_id": clean_label_doc.gridfs_file_id})

        if existing:
            await collection.update_one(
                {"gridfs_file_id": clean_label_doc.gridfs_file_id},
                {"$set": clean_label_doc.to_dict()}
            )
        else:
            await collection.insert_one(clean_label_doc.to_dict())

        return True
    except Exception as e:
        logger.error(f"Errore salvataggio: {e}")
        return False


async def main():
    mongo_config = MongoDBConfig(
        connection_string=os.getenv("MONGODB_CONNECTION_STRING", "mongodb://localhost:27017/"),
        database_name=os.getenv("MONGODB_DATABASE_NAME", "audio_db"),
        audio_collection=os.getenv("MONGODB_AUDIO_COLLECTION", "audio_samples"),
        fs_collection=os.getenv("MONGODB_FS_COLLECTION", "audio_files")
    )

    client = get_mongo_client()
    db = get_mongo_database(client)

    clean_labels_collection = await create_clean_labels_collection(db)
    all_data = await prepare_data_with_ids(mongo_config)

    openai_client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4.1")
    )

    results = []

    for i, data_item in enumerate(all_data):
        logger.info(f"\n[{i + 1}/{len(all_data)}] Processando: {data_item['label']}")

        try:
            clean_label = generate_clean_label(data_item, openai_client)
            logger.info(f"  Originale: {data_item['label']}")
            logger.info(f"  Pulito: {clean_label}")

            clean_label_doc = CleanLabelDocument(
                gridfs_file_id=data_item['gridfs_file_id'],
                original_label=data_item['label'],
                clean_label=clean_label,
                categories=data_item['categories']
            )

            success = await save_clean_label_to_mongo(clean_labels_collection, clean_label_doc)

            results.append({
                'gridfs_file_id': data_item['gridfs_file_id'],
                'original_label': data_item['label'],
                'clean_label': clean_label,
                'saved_to_mongo': success
            })

        except Exception as e:
            logger.error(f"Errore: {e}")
            results.append({
                'gridfs_file_id': data_item['gridfs_file_id'],
                'original_label': data_item['label'],
                'error': str(e),
                'saved_to_mongo': False
            })

    client.close()
    return results


if __name__ == "__main__":
    results = asyncio.run(main())

    success_count = sum(1 for r in results if r.get('saved_to_mongo'))
    logger.info(f"\nProcessamento completato: {success_count}/{len(results)} successi")