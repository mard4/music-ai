import asyncio
import os
from SampleFocusExtractor import download_by_category_to_mongo
from config.settings import settings, mongo_config

async def main():
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

    print(f"Processati {len(results)} samples. Successi: {sum(1 for r in results if r)}")


if __name__ == "__main__":
    asyncio.run(main())