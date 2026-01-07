import logging
import os
import asyncio

from pymongo import MongoClient

from data_retrievial.socialfx.socialfx_extractor import SocialFXKnowledgeBase, create_socialfx_extractor

if __name__ == "__main__":

    async def main():
        # Mock config per test locale
        local_config = {
            "connection_string": "mongodb://localhost:27017",
            "database_name": "rag_audio_db",
            "collection_name": "socialfx_kb"
        }

        try:
            kb_extractor = await create_socialfx_extractor(local_config)
            await kb_extractor.save_kb_to_mongo()
        except Exception as e:
            logging.error(f"Errore nel main: {e}")

    asyncio.run(main())