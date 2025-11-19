import asyncio
import logging
from commons.data_models.models import MongoDBConfig
from ProcessAudio import create_processor_extractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logging.getLogger('pymongo').setLevel(logging.WARNING)


async def main():
    """Main function to process audio files"""
    mongo_config = MongoDBConfig(
        connection_string="mongodb://localhost:27017/",
        database_name="music_ai",
        audio_collection="audio_samples",
        fs_collection="audio_files"
    )
        
    try:
        processor = await create_processor_extractor(mongo_config)
        await processor.process_audio_files_from_mongo()
        logging.info("Audio processing completed successfully!")
    except Exception as e:
        logging.error(f"Failed to process audio files: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())