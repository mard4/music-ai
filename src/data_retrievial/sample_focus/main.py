from SampleFocusExtractor import SampleFocusExtractor, download_by_category_to_mongo
from commons.data_models.models import MongoDBConfig

async def main():
    mongo_config = MongoDBConfig(
        connection_string="mongodb://localhost:27017/",
        database_name="music_ai",
        audio_collection="audio_samples",
        fs_collection="audio_files"
    )
    
    categories = {
            "bass": "https://samplefocus.com/categories/bass",
            "vocals": "https://samplefocus.com/categories/vocals",
            "drums": "https://samplefocus.com/categories/drums", 
            "synths": "https://samplefocus.com/categories/synths",
        }

    results = await download_by_category_to_mongo(
        category_url=categories["drums"],
        max_samples=100,
        mongo_config=mongo_config
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
    
    
